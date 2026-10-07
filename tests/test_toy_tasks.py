"""The 2-D toy samplers match the published toys they reimplement.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import torch

from utils.toy_tasks import sample_8gaussians, sample_fork, sample_moons, sample_targets, toy_batch_sampler


class ToyTaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_8gaussians_match_torchcfm(self):
        x = sample_8gaussians(80_000)
        radius = x.norm(dim=1)
        self.assertAlmostEqual(radius.mean().item(), 5.0, delta=0.1)
        from utils.toy_tasks import _EIGHT_CENTERS
        centers = 5 * _EIGHT_CENTERS
        resid = x - centers[torch.cdist(x, centers).argmin(dim=1)]
        self.assertAlmostEqual(resid.std(0).mean().item(), 0.1 ** 0.25, delta=0.01)

    def test_moons_match_torchdyn_scaled(self):
        x = sample_moons(50_000)
        # torchdyn moons span x in [-1, 2] (+ up to 0.2 shift), times 3 minus 1
        self.assertAlmostEqual(x[:, 0].min().item(), -4.0, delta=0.05)
        self.assertAlmostEqual(x[:, 0].max().item(), 5.6, delta=0.05)
        target, cond = sample_targets('moons', 10)
        self.assertTrue(torch.equal(cond[:, 0], target[:, 0]))

    def test_fork_branches(self):
        xy = sample_fork(20_000)
        x, y = xy[:, 0], xy[:, 1]
        self.assertTrue(torch.equal(y[x <= 0], torch.zeros_like(y[x <= 0])))
        self.assertTrue(torch.allclose(y[x > 0].abs(), x[x > 0]))
        self.assertAlmostEqual((y[x > 0] > 0).float().mean().item(), 0.5, delta=0.02)

    def test_batch_sampler_shapes(self):
        for toy, dim in (('moons', 2), ('fork', 1)):
            x0, x1, obs, ids = toy_batch_sampler(toy)(16)
            self.assertEqual((x0.shape, x1.shape, obs.shape, ids), ((16, 1, dim), (16, 1, dim), (16, 1, 1), None))


if __name__ == '__main__':
    unittest.main()
