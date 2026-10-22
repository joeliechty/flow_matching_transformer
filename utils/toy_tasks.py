"""2-D toy tasks with continuous conditions, for checking the noise-data pairings in
`utils.train_utils.PAIRINGS` against published results before the pose experiments.

moons: 8 Gaussians -> two moons, conditioned on the target point's x-coordinate. C²OT's
       continuous-condition toy (Cheng & Schwing 2025), with TorchCFM's samplers.
fork:  N(0, 1) -> y, conditioned on x ~ U(-1, 1): y = 0 for x <= 0, otherwise +x or -x with
       equal odds. COT Policy's fork toy (Sochopoulos et al. 2025): for x > 0 the
       conditional is bimodal, so few-step samples that average the branches land between them.

Every sampler takes an optional torch.Generator; without one it uses the global RNG, as the
trainer does.
"""
import math

import torch

TOYS = ('moons', 'fork')
DIM = {'moons': 2, 'fork': 1}

_EIGHT_CENTERS = torch.tensor([(1, 0), (-1, 0), (0, 1), (0, -1),
                               (1 / math.sqrt(2), 1 / math.sqrt(2)),
                               (1 / math.sqrt(2), -1 / math.sqrt(2)),
                               (-1 / math.sqrt(2), 1 / math.sqrt(2)),
                               (-1 / math.sqrt(2), -1 / math.sqrt(2))], dtype=torch.float32)


def sample_8gaussians(n, generator=None):
    """TorchCFM's `sample_8gaussians`: centres on a circle of radius 5, covariance √0.1·I
    (TorchCFM passes sqrt(var) as the covariance, so the std is 0.1 ** 0.25 ≈ 0.56)."""
    idx = torch.randint(8, (n,), generator=generator)
    return 5 * _EIGHT_CENTERS[idx] + 0.1 ** 0.25 * torch.randn(n, 2, generator=generator)


def sample_moons(n, generator=None):
    """TorchCFM's `sample_moons`: torchdyn's two moons with noise 0.2, times 3, minus 1.

    torchdyn spaces the angles evenly and adds the same U(0, 0.2) shift to both coordinates;
    here the angles are uniform draws, the same distribution for a fresh batch every step.
    """
    upper = torch.rand(n, generator=generator) < 0.5
    theta = math.pi * torch.rand(n, generator=generator)
    x = torch.where(upper, torch.cos(theta), 1 - torch.cos(theta))
    y = torch.where(upper, torch.sin(theta), 0.5 - torch.sin(theta))
    shift = 0.2 * torch.rand(n, 1, generator=generator)
    return 3 * (torch.stack([x, y], dim=-1) + shift) - 1


def sample_fork(n, generator=None):
    """(x, y): x ~ U(-1, 1); y = 0 for x <= 0, else ±x with equal odds. Returns [n, 2]."""
    x = 2 * torch.rand(n, generator=generator) - 1
    sign = torch.where(torch.rand(n, generator=generator) < 0.5, 1.0, -1.0)
    y = torch.where(x > 0, sign * x, torch.zeros_like(x))
    return torch.stack([x, y], dim=-1)


def sample_targets(toy, n, generator=None):
    """Target samples [n, D] and their conditions [n, 1]."""
    if toy == 'moons':
        x1 = sample_moons(n, generator)
        return x1, x1[:, :1].clone()
    if toy == 'fork':
        xy = sample_fork(n, generator)
        return xy[:, 1:], xy[:, :1]
    raise ValueError(f"Unknown toy: {toy!r}. Expected one of {TOYS}.")


def sample_source(toy, n, generator=None):
    """Source (noise) samples [n, D]."""
    if toy == 'moons':
        return sample_8gaussians(n, generator)
    if toy == 'fork':
        return torch.randn(n, 1, generator=generator)
    raise ValueError(f"Unknown toy: {toy!r}. Expected one of {TOYS}.")


def joint(toy, samples, cond):
    """Points to compare distributions on: the 2-D point for moons (its x is the condition),
    (condition, y) for the fork."""
    return samples if toy == 'moons' else torch.cat([cond, samples], dim=-1)


def toy_batch_sampler(toy, device='cpu'):
    """`Pairer` sampler: n -> (source [n, 1, D], target [n, 1, D], obs [n, 1, 1], None)."""
    def sampler(n):
        x0 = sample_source(toy, n)
        x1, cond = sample_targets(toy, n)
        return (x0.unsqueeze(1).to(device), x1.unsqueeze(1).to(device),
                cond.unsqueeze(1).to(device), None)
    return sampler
