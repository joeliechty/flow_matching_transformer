"""Invariants of the training helpers: OT pairing and conditioning masks.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import torch

from utils.tf_utils import sample_random_twist
from utils.train_utils import (_build_cond_mask, _pair_within_conditions, flat_ot_pairing,
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

    def test_flat_pairing_never_crosses_classes(self):
        x0, x1 = torch.randn(60, 49, 16), torch.randn(60, 49, 16)
        labels = torch.randint(0, 10, (60,))
        paired = _pair_within_conditions(x0, x1, labels, flat_pair)
        for c in labels.unique():
            idx = labels == c
            self.assertTrue(torch.equal(paired[idx], flat_pair(x0[idx], x1[idx])))


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


if __name__ == '__main__':
    unittest.main()
