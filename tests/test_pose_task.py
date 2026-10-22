"""Task files and the conditions derived from them.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import os
import tempfile
import unittest

import torch

from pose_gen_inference import build_obs_from_actions
from utils.pose_task import load_pose_task, task_conditions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUR_CORNERS = os.path.join(REPO, 'configs', 'pose_tasks', 'four_corners.yaml')

# The pose trainer's hard-coded defaults before task files existed (pose-easy-v2).
LEGACY_START = {'mu': [[0, 0, 0, 0, 0, 0]], 'sigma': [[1, 1, 1, 1, 1, 1]]}
LEGACY_GOAL = {
    'mu': [[[5, 5, 5, 0, 0, 1.5708]], [[5, 5, -5, 0, 0, -1.5708]],
           [[5, -5, 5, 0, 0, 3.14159]], [[5, -5, -5, 0, 0, 3.14159]]],
    'sigma': [[[0.1] * 6]] * 4,
}
_T, _B, _R, _L = [0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]
LEGACY_ACTIONS = {'mu': [[_T, _R], [_B, _R], [_T, _L], [_B, _L]], 'sigma': [[[0.0] * 6, [0.0] * 6]] * 4}
# The evaluation's old hard-coded conditions.
LEGACY_FULL = [("top", "right"), ("bottom", "right"), ("top", "left"), ("bottom", "left")]
LEGACY_PARTIAL = [(("top", "right"), 1, [0, 2]), (("bottom", "right"), 1, [1, 3]),
                  (("top", "right"), 0, [0, 1]), (("top", "left"), 0, [2, 3])]


def write_task(text):
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    f.write(text)
    f.close()
    return f.name


class FourCornersTaskTest(unittest.TestCase):
    def test_task_file_reproduces_the_legacy_trainer_defaults(self):
        task = load_pose_task(FOUR_CORNERS)
        self.assertEqual(task['start_dist_params'], LEGACY_START)
        self.assertEqual(task['goal_dist_params'], LEGACY_GOAL)
        self.assertEqual(task['action_dist_params'], LEGACY_ACTIONS)
        self.assertEqual(task['obs_dim'], 6)
        self.assertEqual(task['mode_names'], ['top_right', 'bottom_right', 'top_left', 'bottom_left'])

    def test_conditions_match_the_legacy_evaluation(self):
        full, partial = task_conditions(LEGACY_ACTIONS)
        self.assertEqual([c.valid_modes for c in full], [[0], [1], [2], [3]])
        for cond, pair in zip(full, LEGACY_FULL):
            self.assertTrue(torch.equal(cond.obs(8, 'cpu'), build_obs_from_actions(list(pair), 8, 'cpu')))
            self.assertIsNone(cond.obs_mask(8, 'cpu'))
        self.assertEqual(len(partial), len(LEGACY_PARTIAL))
        for cond, (pair, masked, valid) in zip(partial, LEGACY_PARTIAL):
            self.assertEqual(cond.valid_modes, valid)
            self.assertTrue(torch.equal(cond.obs(8, 'cpu'), build_obs_from_actions(list(pair), 8, 'cpu')))
            expected_mask = torch.zeros(8, 2, dtype=torch.bool)
            expected_mask[:, masked] = True
            self.assertTrue(torch.equal(cond.obs_mask(8, 'cpu'), expected_mask))


class CornersTwoOrientationsTaskTest(unittest.TestCase):
    def test_corners_are_bimodal_in_rotation_only(self):
        from utils.eval_utils import _rotvec, goal_mode_poses_from_config
        from utils.tf_utils import _quat_to_rot_mat
        task = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', 'corners_two_orientations.yaml'))
        base = load_pose_task(FOUR_CORNERS)
        full, partial = task_conditions(task['action_dist_params'])
        self.assertEqual([c.valid_modes for c in full], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual([c.valid_modes for c in partial],
                         [[0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
        poses = goal_mode_poses_from_config(task['goal_dist_params'])
        corners = goal_mode_poses_from_config(base['goal_dist_params'])
        for k in range(8):
            corner = corners[k // 2]
            self.assertTrue(torch.allclose(poses[k, :3], corner[:3]))  # same position as the corner
            rel = _quat_to_rot_mat(corner[None, 3:])[0].T @ _quat_to_rot_mat(poses[k, None, 3:])[0]
            expected = 0.25 if k % 2 == 0 else -0.25                    # ccw, then cw, about z
            self.assertTrue(torch.allclose(_rotvec(rel[None])[0], torch.tensor([0., 0., expected]), atol=1e-4))


class Corners3SigmaTaskTest(unittest.TestCase):
    def test_every_mode_is_within_3_sigma_of_every_other(self):
        from utils.eval_utils import _rotvec, goal_mode_poses_from_config
        from utils.tf_utils import _quat_to_rot_mat
        task = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', 'corners_3sigma.yaml'))
        sigma = 0.1
        poses = goal_mode_poses_from_config(task['goal_dist_params'])
        positions = torch.cdist(poses[:, :3], poses[:, :3])
        R = _quat_to_rot_mat(poses[:, 3:])
        angles = torch.stack([_rotvec(R[i].T @ R)[:, 2].abs() for i in range(8)])  # yaw between modes
        self.assertAlmostEqual(positions.max().item(), 3 * sigma, delta=1e-4)          # the diagonal
        self.assertAlmostEqual(angles.max().item(), 3 * sigma, delta=1e-4)
        corners = positions[positions > 1e-6].unique()
        self.assertAlmostEqual(corners.min().item(), 3 * sigma / 2 ** 0.5, delta=1e-4)  # adjacent corners
        full, partial = task_conditions(task['action_dist_params'])
        self.assertEqual([c.valid_modes for c in full], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(len(partial), 4)


class ContinuousGoalsTaskTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', 'continuous_goals.yaml'))
        self.task = self.spec['task']

    def test_goals_sit_at_their_condition_with_two_orientations(self):
        from utils.eval_utils import _rotvec
        from utils.tf_utils import _quat_to_rot_mat
        torch.manual_seed(0)
        c = self.task.sample_conditions(2000)
        self.assertLessEqual(c.norm(dim=1).max().item(), 5.0)
        self.assertEqual(self.spec['obs_dim'], 2)
        self.assertTrue(torch.allclose(self.task.obs(c)[:, 0], c / 5))
        twists, k = self.task.sample_goals(c)
        self.assertTrue(torch.allclose(twists[:, 1:3], c, atol=0.6))       # 6σ
        self.assertAlmostEqual(twists[:, 0].mean().item(), 5.0, delta=0.02)
        self.assertAlmostEqual(k.float().mean().item(), 0.5, delta=0.05)    # equal odds
        modes = self.task.mode_poses(c[:3])                                  # [3, 2, 7]
        R = _quat_to_rot_mat(modes.reshape(-1, 7)[:, 3:]).reshape(3, 2, 3, 3)
        between = _rotvec(R[:, 0].transpose(-1, -2) @ R[:, 1])
        self.assertTrue(torch.allclose(between, torch.tensor([0., 0., -0.5]).expand(3, 3), atol=1e-4))
        base = self.task.mode_twists(c[:3])[:, :, 5].mean(1)
        self.assertTrue(torch.allclose(base, 1.5708 + 0.5 * c[:3, 0] / 5))

    def test_test_conditions_are_fixed(self):
        self.assertTrue(torch.equal(self.task.test_conditions(), self.task.test_conditions()))
        self.assertEqual(self.task.test_conditions().shape, (16, 2))

    def test_batch_sampler_shapes(self):
        from utils.pose_task import pose_batch_sampler
        start, goal, obs, ids = pose_batch_sampler(self.spec)(32)
        self.assertEqual((start.shape, goal.shape, obs.shape, ids), ((32, 1, 6), (32, 1, 6), (32, 1, 2), None))


class JitterTaskTest(unittest.TestCase):
    def test_same_modes_as_the_clean_task_with_noisy_tokens(self):
        from utils.pose_task import pose_batch_sampler
        for clean_name in ('corners_two_orientations', 'corners_3sigma'):
            with self.subTest(clean_name):
                jitter = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', f'{clean_name}_jitter.yaml'))
                clean = load_pose_task(os.path.join(REPO, 'configs', 'pose_tasks', f'{clean_name}.yaml'))
                self.assertEqual(jitter['goal_dist_params'], clean['goal_dist_params'])
                self.assertEqual(jitter['action_dist_params']['mu'], clean['action_dist_params']['mu'])
                torch.manual_seed(0)
                _, _, obs, ids = pose_batch_sampler(jitter)(4096)
                # the sampler lays the 8 modes out in order, 4096 / 8 rows each
                clean_obs = torch.tensor(clean['action_dist_params']['mu']).repeat_interleave(4096 // 8, dim=0)
                self.assertAlmostEqual((obs - clean_obs).std().item(), 0.1, delta=0.005)
                self.assertEqual(torch.unique(obs.flatten(1), dim=0).shape[0], 4096)  # every condition differs


class GeneralTaskTest(unittest.TestCase):
    def test_modes_sharing_tokens_form_one_multimodal_condition(self):
        path = write_task("""
name: shared
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1, 0], b: [0, 1]}
modes:
  - {name: m0, tokens: [a], mu: [1, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m1, tokens: [a], mu: [-1, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m2, tokens: [b], mu: [0, 1, 0, 0, 0, 0], sigma: 0.2}
""")
        task = load_pose_task(path)
        self.assertEqual(task['obs_dim'], 2)
        self.assertEqual(task['start_dist_params']['sigma'], [[1] * 6])
        self.assertEqual(task['goal_dist_params']['sigma'][2], [[0.2] * 6])
        full, partial = task_conditions(task['action_dist_params'])
        self.assertEqual([c.valid_modes for c in full], [[0, 1], [2]])
        self.assertEqual(partial, [])  # a single token per mode has no one-token subsets

    def test_unknown_tokens_and_ragged_token_lists_are_rejected(self):
        unknown = write_task("""
name: bad
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1]}
modes: [{name: m0, tokens: [z], mu: [0, 0, 0, 0, 0, 0], sigma: 0.1}]
""")
        ragged = write_task("""
name: bad
start: {mu: [0, 0, 0, 0, 0, 0], sigma: 1}
tokens: {a: [1], b: [2]}
modes:
  - {name: m0, tokens: [a], mu: [0, 0, 0, 0, 0, 0], sigma: 0.1}
  - {name: m1, tokens: [a, b], mu: [1, 0, 0, 0, 0, 0], sigma: 0.1}
""")
        for path in (unknown, ragged):
            with self.assertRaises(ValueError):
                load_pose_task(path)


if __name__ == '__main__':
    unittest.main()
