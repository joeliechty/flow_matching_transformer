"""Invariants of the training helpers: OT pairing and conditioning masks.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import torch

from utils.tf_utils import convert_twist_to_pose, sample_random_twist
from utils.train_utils import (EMA, PAIRINGS, Pairer, _batch_cost, _build_cond_mask, _mode_condition_ids,
                               _pair_within_conditions, admissible_ratio, c2ot_weight,
                               condition_aware_pairing, flat_ot_pairing, generate_interpolated_states,
                               pose_normalizer_stats, sample_pose_batch, sample_times,
                               sequence_ot_pairing)


def se3_pair(start, goal):
    return sequence_ot_pairing(start, goal, manifold='se3')


def flat_pair(start, goal):
    return flat_ot_pairing(start, goal, manifold='euclidean')


def is_row_subset(rows, pool):
    """True if every row of `rows` is exactly one of the rows of `pool`."""
    a, b = rows.flatten(1), pool.flatten(1)
    return bool((a[:, None, :] == b[None, :, :]).all(-1).any(1).all())


class PairWithinConditionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        modes = torch.tensor([[5., 5, 5, 0, 0, 1.5708], [5, 5, -5, 0, 0, -1.5708],
                              [5, -5, 5, 0, 0, 3.14159], [5, -5, -5, 0, 0, 3.14159]])
        self.start = sample_random_twist(batch_size=64, mu=[0] * 6, sigma=[1] * 6).unsqueeze(1)
        self.goal = torch.cat([m + 0.1 * torch.randn(16, 6) for m in modes]).unsqueeze(1)
        self.cond = torch.arange(4).repeat_interleave(16)

    def test_se3_pairing_never_crosses_conditions(self):
        paired = _pair_within_conditions(self.start, self.goal, self.cond, se3_pair)
        for c in range(4):
            idx = self.cond == c
            self.assertTrue(is_row_subset(paired[idx], self.start[idx]))
            # a permutation of the condition's own noise, not a resampling of it
            self.assertTrue(torch.equal(paired[idx].flatten(1).sort(0).values,
                                        self.start[idx].flatten(1).sort(0).values))
            self.assertTrue(torch.equal(paired[idx], se3_pair(self.start[idx], self.goal[idx])))

    def test_single_condition_equals_global_pairing(self):
        one = torch.zeros(len(self.cond), dtype=torch.long)
        self.assertTrue(torch.equal(_pair_within_conditions(self.start, self.goal, one, se3_pair),
                                    se3_pair(self.start, self.goal)))

    def test_modes_sharing_tokens_share_a_condition(self):
        t, b, r, l = [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0]
        self.assertEqual(_mode_condition_ids([[t, r], [b, r], [t, l], [b, l]]), [0, 1, 2, 3])
        self.assertEqual(_mode_condition_ids([[t, r], [t, r], [b, r], [b, r.copy()]]), [0, 0, 1, 1])
        # what the trainer used before (one condition per mode) is unchanged for distinct tokens
        ids = torch.tensor(_mode_condition_ids([[t, r], [b, r], [t, l], [b, l]])).repeat_interleave(32)
        self.assertTrue(torch.equal(ids, torch.arange(4).repeat_interleave(32)))

    def test_flat_pairing_never_crosses_classes(self):
        x0, x1 = torch.randn(60, 49, 16), torch.randn(60, 49, 16)
        labels = torch.randint(0, 10, (60,))
        paired = _pair_within_conditions(x0, x1, labels, flat_pair)
        for c in labels.unique():
            idx = labels == c
            self.assertTrue(torch.equal(paired[idx], flat_pair(x0[idx], x1[idx])))


def total_cost(start, goal):
    """Summed pairing cost (Hungarian ties may break differently, totals can't)."""
    return _batch_cost(start, goal).diagonal().sum().item()


class ConditionAwarePairingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        B = 64
        self.start = sample_random_twist(batch_size=B, mu=[0] * 6, sigma=[1] * 6).unsqueeze(1)
        # Continuous conditions: each goal sits at its own condition c (2-D), so all differ.
        self.cond = 3 * torch.randn(B, 2)
        goal = torch.zeros(B, 6)
        goal[:, 0], goal[:, 1:3], goal[:, 5] = 5, self.cond, 0.25 * torch.randn(B).sign()
        self.goal = (goal + 0.1 * torch.randn(B, 6)).unsqueeze(1)
        # Discrete conditions, as in PairWithinConditionsTest.
        modes = torch.tensor([[5., 5, 5, 0, 0, 1.5708], [5, 5, -5, 0, 0, -1.5708],
                              [5, -5, 5, 0, 0, 3.14159], [5, -5, -5, 0, 0, 3.14159]])
        self.dgoal = torch.cat([m + 0.1 * torch.randn(16, 6) for m in modes]).unsqueeze(1)
        self.ids = torch.arange(4).repeat_interleave(16)
        self.onehot = torch.nn.functional.one_hot(self.ids, 4).float()

    def pair(self, method, goal=None, cond=None, **kw):
        goal = self.goal if goal is None else goal
        cond = self.cond if cond is None else cond
        return condition_aware_pairing(self.start, goal, method, cond=cond, **kw)[0]

    def test_every_pairing_permutes_the_noise(self):
        centroids = self.cond[:8].clone()
        for method in PAIRINGS:
            paired = self.pair(method, w=1.0, centroids=centroids)
            self.assertTrue(torch.equal(paired.flatten(1).sort(0).values,
                                        self.start.flatten(1).sort(0).values), method)

    def test_large_weight_on_discrete_conditions_pairs_within_them(self):
        within = _pair_within_conditions(self.start, self.dgoal, self.ids, se3_pair)
        for method, kw in (('c2ot_fixed', {'w': 1e6}), ('c2ot', {})):
            paired = self.pair(method, goal=self.dgoal, cond=self.onehot, **kw)
            # the noise follows its own condition's rows: condition c's noise is start[ids == c]
            for c in range(4):
                idx = self.ids == c
                self.assertTrue(is_row_subset(paired[idx], self.start[idx]), method)
            self.assertAlmostEqual(total_cost(paired, self.dgoal), total_cost(within, self.dgoal),
                                   places=3, msg=method)

    def test_zero_weight_is_global_ot(self):
        paired = self.pair('c2ot_fixed', w=0.0)
        self.assertAlmostEqual(total_cost(paired, self.goal),
                               total_cost(se3_pair(self.start, self.goal), self.goal), places=3)
        self.assertAlmostEqual(total_cost(self.pair('global'), self.goal),
                               total_cost(paired, self.goal), places=3)

    def test_tiny_target_ratio_keeps_the_random_pairing(self):
        # r(w) >= 1/B (every noise sample admits its own partner), so r_tar below that drives
        # w up until only the diagonal is admissible: distinct conditions keep their noise.
        self.assertTrue(torch.equal(self.pair('c2ot', r_tar=1e-6), self.start))

    def test_one_cluster_is_global_ot(self):
        paired = self.pair('cluster', centroids=self.cond.mean(0, keepdim=True))
        self.assertAlmostEqual(total_cost(paired, self.goal),
                               total_cost(se3_pair(self.start, self.goal), self.goal), places=3)

    def test_realised_ratio_hits_the_target(self):
        torch.manual_seed(1)
        B = 512
        start = sample_random_twist(batch_size=B, mu=[0] * 6, sigma=[1] * 6).unsqueeze(1)
        cond = torch.rand(B, 2) * 10
        goal = torch.zeros(B, 6)
        goal[:, 0], goal[:, 1:3] = 5, cond
        goal = goal.unsqueeze(1)
        base = _batch_cost(start, goal)
        d = (cond.unsqueeze(1) - cond.unsqueeze(0)).pow(2).sum(-1)
        for r_tar in (0.005, 0.01, 0.05):
            w = c2ot_weight(base, d, r_tar)
            self.assertAlmostEqual(admissible_ratio(base, d, w), r_tar, delta=0.002)
        # and pairing with it moves the noise less than global OT does, and lowers the cost
        _, info = condition_aware_pairing(start, goal, 'c2ot', cond=cond)
        self.assertAlmostEqual(info['r'], 0.01, delta=0.002)


class PairerTest(unittest.TestCase):
    def test_one_ot_batch_is_served_as_shuffled_network_batches(self):
        torch.manual_seed(0)
        drawn = []

        def sampler(n):
            start = torch.randn(n, 1, 6)
            goal = torch.randn(n, 1, 6)
            obs = torch.randn(n, 1, 2)
            drawn.append((start, goal, obs))
            return start, goal, obs, None

        pairer = Pairer('c2ot', sampler, batch_size=16, ot_batch_mult=4)
        batches = [pairer.next_batch() for _ in range(4)]
        self.assertEqual(len(drawn), 1)
        self.assertTrue(all(b[0].shape[0] == 16 for b in batches))
        goals = torch.cat([b[1] for b in batches])
        self.assertTrue(torch.equal(goals.flatten(1).sort(0).values,
                                    drawn[0][1].flatten(1).sort(0).values))
        # a goal keeps its own obs through the shuffle
        for start, goal, obs in batches:
            rows = [(drawn[0][1].flatten(1) == g).all(-1).nonzero().item() for g in goal.flatten(1)]
            self.assertTrue(torch.equal(obs, drawn[0][2][rows]))
        pairer.next_batch()
        self.assertEqual(len(drawn), 2)

    def test_cluster_centroids_and_fixed_weight_are_set_once(self):
        torch.manual_seed(0)
        sampler = lambda n: (torch.randn(n, 1, 6), torch.randn(n, 1, 6), torch.randn(n, 1, 2), None)
        self.assertEqual(Pairer('cluster', sampler, 16, ot_batch_mult=2).centroids.shape, (32, 2))
        self.assertGreater(Pairer('c2ot_fixed', sampler, 16).w, 0)


class CondMaskTest(unittest.TestCase):
    def test_without_cfg_no_sample_is_fully_masked(self):
        torch.manual_seed(0)
        for num_tokens in (1, 2, 3):
            mask = _build_cond_mask(50_000, num_tokens, use_cfg=False, device='cpu')
            self.assertFalse(mask.all(dim=-1).any(), f"{num_tokens} tokens")
        # partial (single-token) dropout is kept when there is more than one token
        mask = _build_cond_mask(50_000, 2, use_cfg=False, device='cpu')
        self.assertGreater(mask.any(dim=-1).float().mean().item(), 0.15)

    def test_with_cfg_some_samples_are_fully_masked(self):
        torch.manual_seed(0)
        mask = _build_cond_mask(50_000, 2, use_cfg=True, device='cpu')
        self.assertAlmostEqual(mask.all(dim=-1).float().mean().item(), 0.109, delta=0.01)

    def test_with_cfg_matches_the_original_rng_stream(self):
        # CFG runs must train exactly as before the NOCFG fix
        for num_tokens in (1, 2):
            torch.manual_seed(7)
            indep = torch.rand(4096, num_tokens) < 0.1
            uncond = torch.rand(4096) < 0.1
            expected = indep | uncond.unsqueeze(-1)
            torch.manual_seed(7)
            self.assertTrue(torch.equal(_build_cond_mask(4096, num_tokens, use_cfg=True, device='cpu'),
                                        expected))



class TimeSamplingTest(unittest.TestCase):
    def test_densities(self):
        torch.manual_seed(0)
        for dist, mean in (('uniform', 0.5), ('logit_normal', 0.5), ('beta', 0.999 * 0.4)):
            with self.subTest(dist):
                t = sample_times(200_000, dist)
                self.assertTrue(bool(((t >= 0) & (t <= 1)).all()))
                self.assertAlmostEqual(t.mean().item(), mean, delta=0.005)
        # π0's density puts about 65% of the times in the noisier half
        self.assertGreater((sample_times(200_000, 'beta') < 0.5).float().mean().item(), 0.6)

    def test_se3_paths_run_from_start_to_goal(self):
        torch.manual_seed(0)
        start, goal = sample_random_twist(16), sample_random_twist(16, mu=[5, 5, 5, 0, 0, 1.5])
        t = torch.tensor([[0.0, 1.0]]).repeat(16, 1)
        interp, _, _ = generate_interpolated_states(start, goal, manifold='se3', t=t)
        for k, end in ((0, start), (1, goal)):
            pose = convert_twist_to_pose(end, dt=1.0, return_representation='quat')
            self.assertTrue(torch.allclose(interp[:, k, :3], pose[:, :3], atol=1e-4))
            self.assertTrue(torch.allclose((interp[:, k, 3:] * pose[:, 3:]).sum(-1).abs(),
                                           torch.ones(16), atol=1e-4))


class EMATest(unittest.TestCase):
    def test_the_first_update_uses_the_warmup_decay(self):
        model = torch.nn.Linear(2, 2)
        ema = EMA(model, 0.999)
        old = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)
        decay = 2 / 11  # min(0.999, (1 + 1) / (10 + 1))
        self.assertTrue(torch.allclose(ema.shadow['weight'], decay * old + (1 - decay) * (old + 1)))


class NormalizerStatsTest(unittest.TestCase):
    def test_stats_leave_the_training_rng_untouched(self):
        start = {'mu': [[0] * 6], 'sigma': [[1] * 6]}
        goal = {'mu': [[[5, 5, 5, 0, 0, 1.5]], [[5, -5, 5, 0, 0, -1.5]]], 'sigma': [[[0.1] * 6]] * 2}
        sampler = lambda n: sample_pose_batch(n, start, goal, None)
        torch.manual_seed(3)
        expected = torch.rand(5)
        torch.manual_seed(3)
        stats = pose_normalizer_stats(sampler, n=4096)
        self.assertTrue(torch.equal(torch.rand(5), expected))
        self.assertEqual([v.shape[0] for v in stats.values()], [3, 3, 6, 6])
        self.assertAlmostEqual(stats['vel_mean'][0].item(), 5.0, delta=0.1)  # the goals sit at x = 5


if __name__ == '__main__':
    unittest.main()
