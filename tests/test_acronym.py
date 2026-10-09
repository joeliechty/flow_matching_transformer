"""ACRONYM task, condition encoders and pose-recreation metrics.

Run from the repo root: python -m unittest discover -s tests -t .
The sampler tests need the example cache (prepare_acronym.py build --dataset
configs/datasets/acronym_examples.yaml --mesh_root <acronym repo>/data/examples) and are
skipped without it.
"""
import math
import os
import unittest

import torch

from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from utils.acronym import REPO, AcronymGraspTask, random_rotations
from utils.grasp_metrics import one_nn_accuracy, pose_distances, recreation_metrics
from utils.tf_utils import _quaternion_multiply, compute_twist_between_poses

TCP = 0.089
EXAMPLES = os.path.join(REPO, 'data', 'acronym', 'acronym_examples.pt')


def _grasps(n, seed, spread=0.05):
    """Random grasp poses [n, 7] in metres: positions N(0, spread²), uniform rotations."""
    gen = torch.Generator().manual_seed(seed)
    return torch.cat([spread * torch.randn(n, 3, generator=gen), random_rotations(n, gen)], -1)


class PoseDistanceTest(unittest.TestCase):
    def test_jaw_swap_is_the_same_grasp(self):
        g = _grasps(64, 0)
        flipped = g.clone()
        flipped[:, 3:] = _quaternion_multiply(g[:, 3:], torch.tensor([0., 0., 0., 1.]).expand(64, 4))
        self.assertLess(pose_distances(g, flipped, TCP).diagonal().max().item(), 1e-4)

    def test_a_shift_is_measured_in_cm(self):
        g = _grasps(16, 1)
        shifted = g.clone()
        shifted[:, 0] += 0.01
        self.assertTrue(torch.allclose(pose_distances(g, shifted, TCP).diagonal(),
                                       torch.ones(16), atol=1e-4))


class RecreationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.real = _grasps(400, 2)

    def test_identical_sets_are_perfect(self):
        m = recreation_metrics(self.real, self.real, TCP)
        self.assertLess(m['recreation_err'], 1e-4)
        self.assertLess(m['fidelity_err'], 1e-4)
        self.assertEqual(m['coverage_1cm'], 1.0)
        self.assertEqual(m['precision_1cm'], 1.0)

    def test_one_nna_tells_shifted_sets_apart(self):
        same = recreation_metrics(_grasps(400, 3), self.real, TCP)['one_nna']
        shifted = _grasps(400, 3)
        shifted[:, :3] += 0.2
        apart = recreation_metrics(shifted, self.real, TCP)['one_nna']
        self.assertLess(abs(same - 0.5), 0.08)
        self.assertGreater(apart, 0.95)

    def test_collapse_and_over_dispersion_fail_different_errors(self):
        fresh = recreation_metrics(_grasps(400, 4), self.real, TCP)
        collapsed = recreation_metrics(self.real[:1].repeat(400, 1), self.real, TCP)
        dispersed = recreation_metrics(_grasps(400, 4, spread=0.15), self.real, TCP)
        self.assertGreater(collapsed['recreation_err'], 2 * fresh['recreation_err'])
        self.assertGreater(dispersed['fidelity_err'], 1.5 * fresh['fidelity_err'])
        self.assertLess(collapsed['spread_ratio_trans'], 0.1)
        self.assertGreater(dispersed['spread_ratio_trans'], 2.0)

    def test_one_nn_accuracy_uses_equal_set_sizes(self):
        d = torch.rand(10, 30)
        self.assertGreaterEqual(one_nn_accuracy(d, torch.rand(10, 10), torch.rand(30, 30)), 0.0)


class UniformPriorTest(unittest.TestCase):
    def test_rotation_angle_density(self):
        # uniform on SO(3): the angle has density (1 - cos θ) / π on [0, π], mean π/2 + 2/π
        q = random_rotations(200_000, torch.Generator().manual_seed(0))
        pose = torch.cat([torch.zeros(len(q), 3), q], -1)
        angle = compute_twist_between_poses(pose, None)[:, 3:].norm(dim=-1)
        self.assertLessEqual(angle.max().item(), math.pi + 1e-4)
        self.assertAlmostEqual(angle.mean().item(), math.pi / 2 + 2 / math.pi, delta=0.01)


def _model(obs_encoder, cond_mode, **kw):
    return ConditionalFlowMatchingTransformerModel(
        input_dim=7, output_dim=6, hidden_dim=32, num_layers=2, num_heads=4, phase_dim=32,
        max_seq_len=1, cond_mode=cond_mode, obs_encoder=obs_encoder, **kw)


class ConditionEncoderTest(unittest.TestCase):
    def test_every_encoder_and_pathway_runs(self):
        x = torch.cat([torch.randn(6, 1, 3), random_rotations(6).unsqueeze(1)], -1)
        t = torch.rand(6)
        for cond_mode in ('adaln', 'cross_attn', 'joint'):
            for enc, obs, kw in (('embedding', torch.randint(10, (6, 2)), {'num_obs_tokens': 2, 'obs_vocab': 10}),
                                 ('point_patch', torch.randn(6, 256, 3), {'num_obs_tokens': 8, 'obs_dim': 3})):
                model = _model(enc, cond_mode, **kw)
                self.assertEqual(model(x, obs, t).shape, (6, 1, 6))
                self.assertEqual(model.inference(x, obs, num_steps=2, cfg_scale=2.0).shape, (6, 1, 7))
                self.assertEqual(model.inference(x, None, num_steps=1).shape, (6, 1, 7))

    def test_shared_contexts_match_per_sample_tokens(self):
        # keys and values computed once per cloud give the same velocity as per-sample tokens,
        # with and without nulled samples, in every pathway
        torch.manual_seed(0)
        x = torch.cat([torch.randn(8, 1, 3), random_rotations(8).unsqueeze(1)], -1)
        t, clouds = torch.rand(8), torch.randn(2, 256, 3).repeat_interleave(4, dim=0)
        mask = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool).unsqueeze(1).expand(8, 8)
        for cond_mode in ('adaln', 'cross_attn', 'joint'):
            model = _model('point_patch', cond_mode, num_obs_tokens=8, obs_dim=3)
            for p in model.parameters():  # leave the zero-initialised start, so the condition matters
                torch.nn.init.normal_(p, std=0.05)
            shared = model.encode_obs(clouds)
            per_sample = shared.unique[shared.index]
            for m in (None, mask):
                a, b = model._predict(x, shared, t, m), model._predict(x, per_sample, t, m)
                self.assertTrue(torch.allclose(a, b, atol=1e-5), cond_mode)

    def test_identical_clouds_are_encoded_once(self):
        model = _model('point_patch', 'cross_attn', num_obs_tokens=8, obs_dim=3)
        cloud = torch.randn(2, 256, 3)
        shared = model.encode_obs(cloud.repeat_interleave(3, dim=0))
        self.assertEqual(len(shared.unique), 2)
        tokens = shared.unique[shared.index]
        self.assertTrue(torch.equal(tokens[0], tokens[2]))
        self.assertFalse(torch.equal(tokens[0], tokens[3]))


@unittest.skipUnless(os.path.exists(EXAMPLES), 'no example cache (see module docstring)')
class AcronymSamplerTest(unittest.TestCase):
    def setUp(self):
        spec = {'name': 'examples', 'dataset': 'configs/datasets/acronym_examples.yaml',
                'condition': 'points', 'batch': {'objects': 2, 'grasps_per_object': 16}}
        self.task = AcronymGraspTask(spec)

    def test_batches_hold_k_grasps_of_each_object(self):
        torch.manual_seed(0)
        start, goal, obs, ids = self.task.batch_sampler()(32)
        self.assertEqual((start.shape, goal.shape, obs.shape), ((32, 1, 6), (32, 1, 6), (32, 2048, 3)))
        self.assertEqual(sorted(ids.unique_consecutive().tolist()), [0, 1])
        for block in ids.split(16):
            self.assertEqual(block.unique().numel(), 1)
        self.assertTrue(torch.equal(obs[0], obs[15]))

    def test_goals_are_training_grasps(self):
        torch.manual_seed(0)
        _, goal, _, ids = self.task.batch_sampler()(32)
        pose = torch.cat([goal[:, 0, :3], torch.zeros(32, 4)], -1)  # positions only
        train = self.task.poses[self.task.split == 0, :3]
        nearest = torch.cdist(pose[:, :3], train, compute_mode='donot_use_mm_for_euclid_dist').min(1).values
        self.assertLess(nearest.max().item(), 1e-4)

    def test_reference_and_floor_are_disjoint(self):
        ref, floor = self.task.reference_and_floor(0, 'heldout_grasps', 64)
        self.assertEqual(floor.shape[0], 64)
        self.assertGreater(torch.cdist(ref[:, :3], floor[:, :3]).min().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
