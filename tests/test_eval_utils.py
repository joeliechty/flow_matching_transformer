"""Sanity checks for the pose metrics: each must score real data as perfect and flag the
failure it is meant to catch.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import torch

from utils.eval_utils import (_rotvec, energy_distance, goal_mode_poses_from_config,
                              mode_balance_kl, nearest_mode_indices, path_straightness,
                              per_mode_energy_distance, pose_bias_spread, pose_mode_distance,
                              sample_goal_mixture, sample_goal_poses)
from utils.tf_utils import compute_twist_between_poses
from utils.tf_utils import _quat_to_rot_mat, add_twist_to_pose, convert_twist_to_pose

# The pose trainer's default goal distribution: 4 modes, sigma = 0.1 per twist dimension.
GOALS = {
    'mu': [[[5, 5, 5, 0, 0, 1.5708]], [[5, 5, -5, 0, 0, -1.5708]],
           [[5, -5, 5, 0, 0, 3.14159]], [[5, -5, -5, 0, 0, 3.14159]]],
    'sigma': [[[0.1] * 6]] * 4,
}


class PoseDistributionMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modes = goal_mode_poses_from_config(GOALS)
        cls.ref, cls.ref_idx = sample_goal_poses(GOALS, 1024, seed=1)
        cls.real, cls.real_idx = sample_goal_poses(GOALS, 256, seed=2)

    def stats(self, poses, assigned):
        return pose_bias_spread(poses, assigned, self.modes, self.ref, self.ref_idx)

    def ed(self, poses, assigned):
        return per_mode_energy_distance(poses, assigned, self.ref, self.ref_idx)

    def test_real_samples_score_as_data(self):
        s = self.stats(self.real, self.real_idx)
        self.assertLess(s['bias_trans'], 0.05)
        self.assertLess(s['bias_rot'], 0.05)
        self.assertAlmostEqual(s['spread_ratio_trans'], 1.0, delta=0.1)
        self.assertAlmostEqual(s['spread_ratio_rot'], 1.0, delta=0.1)
        self.assertLess(abs(self.ed(self.real, self.real_idx)), 0.02)

    def test_collapse_onto_mode_centres_is_flagged(self):
        # Twist distance rewards this (it scores below real data); these metrics must not.
        collapsed = self.modes[self.real_idx]
        self.assertLess(pose_mode_distance(collapsed, self.modes, self.real_idx),
                        pose_mode_distance(self.real, self.modes, self.real_idx))
        s = self.stats(collapsed, self.real_idx)
        self.assertLess(s['spread_ratio_trans'], 0.01)
        self.assertLess(s['spread_ratio_rot'], 0.01)
        self.assertGreater(self.ed(collapsed, self.real_idx), 0.1)

    def test_collapse_onto_the_mode_average_is_flagged(self):
        # The optimal one-step sample without OT: every sample at the mean of the modes.
        mean_twist = torch.tensor(GOALS['mu']).reshape(4, 6).mean(0, keepdim=True)
        avg = convert_twist_to_pose(mean_twist.repeat(len(self.real), 1), dt=1.0,
                                    return_representation='quat')
        nearest = nearest_mode_indices(avg, self.modes)
        self.assertGreater(self.stats(avg, nearest)['bias_trans'], 4.0)
        self.assertGreater(self.ed(avg, nearest), 1.0)

    def test_overdispersion_is_flagged(self):
        wide_goals = {'mu': GOALS['mu'], 'sigma': [[[0.2] * 6]] * 4}
        wide, wide_idx = sample_goal_poses(wide_goals, 256, seed=3)
        s = self.stats(wide, wide_idx)
        self.assertAlmostEqual(s['spread_ratio_trans'], 2.0, delta=0.3)
        self.assertAlmostEqual(s['spread_ratio_rot'], 2.0, delta=0.3)
        self.assertGreater(self.ed(wide, wide_idx), 0.02)

    def test_energy_distance_grows_with_shift(self):
        def shifted(dx):
            out = self.real.clone()
            out[:, 0] += dx
            return out
        base = self.ed(self.real, self.real_idx)
        small, large = self.ed(shifted(0.1), self.real_idx), self.ed(shifted(0.5), self.real_idx)
        self.assertGreater(small, base + 0.01)
        self.assertGreater(large, small)

    def test_whole_mixture_energy_distance_dilutes_within_mode_errors(self):
        # Why the metric is per mode: over the mixture, collapse barely registers.
        collapsed = self.modes[self.real_idx]
        mixture_gap = energy_distance(collapsed, self.ref) - energy_distance(self.real, self.ref)
        per_mode_gap = self.ed(collapsed, self.real_idx) - self.ed(self.real, self.real_idx)
        self.assertGreater(per_mode_gap, 2.5 * mixture_gap)


class PathAndRotationTest(unittest.TestCase):
    def test_training_interpolation_is_perfectly_straight(self):
        torch.manual_seed(0)
        start = convert_twist_to_pose(torch.randn(32, 6), dt=1.0, return_representation='quat')
        twist = torch.randn(32, 6)
        traj = torch.stack([add_twist_to_pose(start, twist, torch.full((32, 1), float(t)))
                            for t in torch.linspace(0, 1, 21)], dim=1)
        ratio, chord = path_straightness(traj)
        self.assertAlmostEqual(ratio, 1.0, delta=1e-3)
        self.assertAlmostEqual(chord, twist.norm(dim=-1).mean().item(), delta=1e-3)

    def test_out_and_back_detour_is_long(self):
        # Go far out, then come back to near the start: long path, short chord.
        torch.manual_seed(0)
        a = convert_twist_to_pose(torch.randn(16, 6), dt=1.0, return_representation='quat')
        far = add_twist_to_pose(a, 2 * torch.randn(16, 6), torch.ones(16, 1))
        near_start = add_twist_to_pose(a, 0.1 * torch.randn(16, 6), torch.ones(16, 1))
        back = add_twist_to_pose(far, compute_twist_between_poses(far, near_start, dt=1.0),
                                 torch.ones(16, 1))
        ratio, _ = path_straightness(torch.stack([a, far, back], dim=1))
        self.assertGreater(ratio, 5.0)

    def test_rotvec_recovers_known_rotations(self):
        for angle in (0.0, 0.3, 3.0):
            pose = convert_twist_to_pose(torch.tensor([[0., 0, 0, 0, 0, angle]]), dt=1.0,
                                         return_representation='quat')
            rv = _rotvec(_quat_to_rot_mat(pose[:, 3:]))
            self.assertTrue(torch.allclose(rv, torch.tensor([[0., 0, angle]]), atol=1e-4), rv)

    def test_balance_kl(self):
        balanced = torch.tensor([0, 2] * 32)
        lopsided = torch.tensor([0] * 60 + [2] * 4)
        self.assertLess(mode_balance_kl(balanced, [0, 2]), 1e-6)
        self.assertGreater(mode_balance_kl(lopsided, [0, 2]), 0.3)

    def test_random_mixture_gives_the_balance_noise_floor(self):
        # A perfect sampler's 50/50 split is random, so its balance KL is ~1/(2n), not 0.
        kls = []
        for seed in range(200):
            _, idx = sample_goal_mixture(GOALS, 64, seed, modes=[0, 2])
            self.assertTrue(set(idx.tolist()) <= {0, 2})
            kls.append(mode_balance_kl(idx, [0, 2]))
        self.assertAlmostEqual(sum(kls) / len(kls), 1 / 128, delta=0.003)


if __name__ == '__main__':
    unittest.main()
