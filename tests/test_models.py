"""Invariants of the flow-matching transformers: adaLN-Zero initialisation, the condition
pathways, the SE(3) input and normalisation, checkpoints, ODE solvers and guidance intervals.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import os
import tempfile
import unittest

import torch

from models.conditional_flow_matching_transformer import COND_MODES, ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from models.support_models import TransformerBlock
from utils.tf_utils import convert_twist_to_pose, sample_random_twist
from utils.train_utils import EMA

SMALL = dict(hidden_dim=32, num_layers=2, num_heads=4, phase_dim=32, max_seq_len=1)


def poses(n):
    """[n, 1, 7] random quaternion poses."""
    return convert_twist_to_pose(sample_random_twist(n), dt=1.0, return_representation='quat').unsqueeze(1)


def conditional(cond_mode='adaln', **kwargs):
    return ConditionalFlowMatchingTransformerModel(7, 6, obs_dim=6, cond_mode=cond_mode,
                                                   num_obs_tokens=2, **SMALL, **kwargs)


def perturb(model, scale=0.05):
    """Move every weight, the zero-initialised ones included, off its initial value."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))


class InitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_a_block_starts_as_the_identity(self):
        block = TransformerBlock(32, 4, cond_dim=16, cross_attn=True)
        block.zero_init()
        x = torch.randn(3, 5, 32)
        self.assertTrue(torch.equal(block(x, torch.randn(3, 16), context=torch.randn(3, 2, 32)), x))

    def test_the_initial_velocity_is_the_mean_velocity(self):
        model = conditional()
        vel_mean = torch.arange(6.)
        model.set_normalizer(torch.zeros(3), torch.ones(3), vel_mean, 2 * torch.ones(6))
        v = model(poses(4), torch.randn(4, 2, 6), torch.rand(4))
        self.assertTrue(torch.allclose(v, vel_mean.expand_as(v)))


class ConditionPathwayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x, self.t = poses(4), torch.rand(4)
        self.obs = torch.randn(4, 2, 6)

    def test_every_cond_mode_sees_the_tokens_unless_they_are_nulled(self):
        all_null = torch.ones(4, 2, dtype=torch.bool)
        for mode in COND_MODES:
            with self.subTest(mode):
                model = conditional(mode)
                perturb(model)
                model.eval()
                v = model(self.x, self.obs, self.t)
                self.assertEqual(v.shape, (4, 1, 6))
                self.assertFalse(torch.allclose(v, model(self.x, self.obs + 1, self.t)))
                self.assertTrue(torch.allclose(model(self.x, self.obs, self.t, cond_mask=all_null),
                                               model(self.x, self.obs + 1, self.t, cond_mask=all_null)))

    def test_nulling_one_slot_differs_from_nulling_the_other(self):
        model = conditional()
        perturb(model)
        first = torch.tensor([[True, False]]).repeat(4, 1)
        tokens = torch.randn(1, 1, 6).repeat(4, 2, 1)  # the same token in both slots
        self.assertFalse(torch.allclose(model(self.x, tokens, self.t, cond_mask=first),
                                        model(self.x, tokens, self.t, cond_mask=~first)))


class SE3InputTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_q_and_minus_q_give_the_same_velocity(self):
        model = FlowMatchingTransformerModel(7, 6, **SMALL)
        perturb(model)
        x, t = poses(5), torch.rand(5)
        flipped = x.clone()
        flipped[..., 3:] *= -1
        self.assertTrue(torch.allclose(model(x, t), model(flipped, t), atol=1e-6))

    def test_the_loss_is_in_normalised_units(self):
        model = FlowMatchingTransformerModel(7, 6, **SMALL)
        vel_mean, vel_std = torch.randn(6), torch.rand(6) + 0.5
        model.set_normalizer(torch.zeros(3), torch.ones(3), vel_mean, vel_std)
        z = torch.randn(8, 1, 6)
        # the zero-initialised network predicts 0 in normalised units, so the loss is mean(z²)
        loss = model.cfm_loss(poses(8), torch.rand(8), vel_mean + vel_std * z)
        self.assertAlmostEqual(loss.item(), z.pow(2).mean().item(), places=5)


class CheckpointTest(unittest.TestCase):
    def test_a_round_trip_loads_the_ema_weights(self):
        torch.manual_seed(0)
        model = conditional('cross_attn', time_emb='fourier')
        ema = EMA(model, 0.5)
        perturb(model)
        ema.update(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.pt')
            model.save_checkpoint(path, epoch=1, ema_state_dict=ema.state_dict(model))
            loaded, _ = ConditionalFlowMatchingTransformerModel.load_checkpoint(path)
            raw, _ = ConditionalFlowMatchingTransformerModel.load_checkpoint(path, use_ema=False)
        self.assertEqual(loaded.model_config(), model.model_config())
        for name, value in ema.state_dict(model).items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], value), name)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(raw.state_dict()[name], value), name)


class SamplingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_solvers_agree_on_a_constant_field(self):
        # A zero-initialised network outputs the mean velocity everywhere
        for manifold, model, x0 in (
            ('euclidean', FlowMatchingTransformerModel(2, 2, manifold='euclidean', **SMALL), torch.randn(5, 2)),
            ('se3', FlowMatchingTransformerModel(7, 6, **SMALL), poses(5)[:, 0]),
        ):
            with self.subTest(manifold):
                model.set_normalizer(torch.zeros(3), torch.ones(3), 0.5 * torch.randn(model.output_dim),
                                     torch.ones(model.output_dim))
                euler = model.inference(x0, num_steps=4, method='euler')
                for method in ('midpoint', 'heun'):
                    self.assertTrue(torch.allclose(model.inference(x0, num_steps=4, method=method),
                                                   euler, atol=1e-5), method)

    def test_guidance_intervals(self):
        model = conditional()
        perturb(model)
        x0, obs = poses(6)[:, 0], torch.randn(6, 2, 6)
        sample = lambda **kw: model.inference(x0, obs, num_steps=5, **kw)
        guided = sample(cfg_scale=3.0)
        self.assertTrue(torch.equal(sample(cfg_scale=3.0, cfg_interval=(0.0, 1.0)), guided))
        self.assertTrue(torch.equal(sample(cfg_scale=3.0, cfg_interval=(2.0, 3.0)), sample(cfg_scale=1.0)))
        self.assertFalse(torch.allclose(sample(cfg_scale=3.0, cfg_interval=(0.5, 1.0)), guided))


if __name__ == '__main__':
    unittest.main()
