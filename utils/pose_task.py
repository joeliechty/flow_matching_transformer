"""Pose-generation task definitions.

A task is a YAML file (see `configs/pose_tasks/`) giving the start distribution, a vocabulary
of conditioning tokens, and the goal modes: each a Gaussian in twist space named by a list
of tokens. `load_pose_task` turns it into the distribution dicts the trainer uses.
`task_conditions` recovers the conditions from a run's saved `action_dist_params`, so
evaluation works for any task, including runs that predate task files.

A file with `type: continuous` instead describes goals that vary continuously with the
condition (`ContinuousGoalTask`), so no two training samples share a condition, and one with
`type: acronym` real grasp poses conditioned on the object (`utils.acronym`).
"""
import math
from dataclasses import dataclass
from typing import List

import torch
from omegaconf import OmegaConf

from utils.tf_utils import convert_twist_to_pose, sample_random_twist

TEST_CONDITION_SEED = 2024  # fixed evaluation conditions of continuous tasks


def _vec(value, dim):
    """A scalar broadcast to `dim` entries, or a list of length `dim` as given."""
    if isinstance(value, (int, float)):
        return [value] * dim
    if len(value) != dim:
        raise ValueError(f"expected {dim} values, got {len(value)}: {value}")
    return list(value)


def load_pose_task(path, device='cpu'):
    """Read a task file into the trainer's distribution dicts.

    Returns a dict with 'name', 'mode_names', 'tokens' (the vocabulary), 'obs_dim',
    'start_dist_params', 'goal_dist_params' and 'action_dist_params', laid out exactly as
    the pose trainer has always built them ([K, 1, 6] goal twists, [K, M, obs_dim] tokens).
    An ACRONYM task's data goes to `device`.
    """
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if spec.get('type') == 'continuous':
        return _load_continuous_task(spec)
    if spec.get('type') == 'acronym':
        from utils.acronym import load_acronym_task
        return load_acronym_task(spec, device)
    tokens, modes = spec['tokens'], spec['modes']
    obs_dim = len(next(iter(tokens.values())))
    for mode in modes:
        unknown = [t for t in mode['tokens'] if t not in tokens]
        if unknown:
            raise ValueError(f"mode {mode['name']!r} uses unknown tokens {unknown}")
    num_tokens = {len(m['tokens']) for m in modes}
    if len(num_tokens) != 1:
        raise ValueError("every mode must be named by the same number of tokens")
    M = num_tokens.pop()
    token_sigma = spec.get('token_sigma', 0.0)
    return {
        'name': spec['name'],
        'type': 'discrete',
        'mode_names': [m['name'] for m in modes],
        'tokens': tokens,
        'obs_dim': obs_dim,
        'start_dist_params': {'mu': [_vec(spec['start']['mu'], 6)],
                              'sigma': [_vec(spec['start']['sigma'], 6)]},
        'goal_dist_params': {'mu': [[_vec(m['mu'], 6)] for m in modes],
                             'sigma': [[_vec(m['sigma'], 6)] for m in modes]},
        'action_dist_params': {'mu': [[tokens[t] for t in m['tokens']] for m in modes],
                               'sigma': [[[float(token_sigma)] * obs_dim] * M] * len(modes)},
    }


def _load_continuous_task(spec):
    task = ContinuousGoalTask(spec)
    return {
        'name': spec['name'],
        'type': 'continuous',
        'spec': spec,
        'task': task,
        'mode_names': None,
        'tokens': None,
        'obs_dim': task.obs_dim,
        'start_dist_params': {'mu': [_vec(spec['start']['mu'], 6)],
                              'sigma': [_vec(spec['start']['sigma'], 6)]},
        'goal_dist_params': None,
        'action_dist_params': None,
    }


class ContinuousGoalTask:
    """Goal poses whose distribution varies continuously with the condition.

    The condition c is a point drawn uniformly from a disk of radius R in the y-z plane. A goal
    sits at position (x, c_y, c_z) and turns about z by base_yaw + yaw_gain · c_y / R plus one
    of the `orientations` offsets, picked with equal odds. Every twist dimension then gets
    Gaussian noise of std sigma. The model sees c / R as a single obs token.
    """

    def __init__(self, spec):
        cond, goal = spec['condition'], spec['goal']
        self.radius = float(cond['radius'])
        self.x, self.sigma = float(goal['x']), float(goal['sigma'])
        self.base_yaw, self.yaw_gain = float(goal['base_yaw']), float(goal['yaw_gain'])
        self.offsets = torch.tensor(goal['orientations'], dtype=torch.float32)
        self.num_test_conditions = int(spec.get('test_conditions', 16))
        self.obs_dim = 2

    @property
    def num_modes(self):
        return len(self.offsets)

    def sample_conditions(self, n, generator=None, device='cpu'):
        """[n, 2] conditions, uniform on the disk."""
        r = self.radius * torch.rand(n, generator=generator).sqrt()
        phi = 2 * math.pi * torch.rand(n, generator=generator)
        return torch.stack([r * phi.cos(), r * phi.sin()], dim=-1).to(device)

    def test_conditions(self, device='cpu'):
        """The fixed evaluation conditions [num_test_conditions, 2]."""
        gen = torch.Generator().manual_seed(TEST_CONDITION_SEED)
        return self.sample_conditions(self.num_test_conditions, gen, device)

    def obs(self, c):
        """The model's obs tokens [n, 1, 2] for conditions c [n, 2]."""
        return (c / self.radius).unsqueeze(1)

    def mode_twists(self, c):
        """Mean twist of every mode under each condition [n, K, 6]."""
        mu = torch.zeros(c.shape[0], self.num_modes, 6, device=c.device)
        mu[..., 0] = self.x
        mu[..., 1:3] = c.unsqueeze(1)
        mu[..., 5] = (self.base_yaw + self.yaw_gain * c[:, :1] / self.radius
                      + self.offsets.to(c.device))
        return mu

    def mode_poses(self, c):
        """Mode centres as poses [n, K, 7]."""
        return convert_twist_to_pose(self.mode_twists(c), dt=1.0, return_representation='quat')

    def sample_goals(self, c, generator=None):
        """Goal twists [n, 6] given conditions c [n, 2], and the mode each came from [n]."""
        n = c.shape[0]
        k = torch.randint(self.num_modes, (n,), generator=generator).to(c.device)
        eps = torch.randn(n, 6, generator=generator).to(c.device)
        return self.mode_twists(c)[torch.arange(n, device=c.device), k] + self.sigma * eps, k

    def batch_sampler(self, start_dist_params, seq_len=1, device='cpu'):
        """`Pairer` sampler: n -> (start [n, 1, 6], goal [n, 1, 6], obs [n, 1, 2], None)."""
        if seq_len != 1:
            raise ValueError("continuous pose tasks support seq_len = 1 only")

        def sampler(n):
            start = sample_random_twist(batch_size=n, mu=start_dist_params['mu'][0],
                                        sigma=start_dist_params['sigma'][0], device=device)
            c = self.sample_conditions(n, device=device)
            goal, _ = self.sample_goals(c)
            return start.unsqueeze(1), goal.unsqueeze(1), self.obs(c), None
        return sampler


def pose_batch_sampler(task, seq_len=1, device='cpu'):
    """`Pairer` sampler n -> (start, goal, obs, cond_ids) for a conditional task from
    `load_pose_task` (either kind)."""
    if task['type'] == 'continuous':
        return task['task'].batch_sampler(task['start_dist_params'], seq_len, device)
    if task['type'] == 'acronym':
        return task['task'].batch_sampler(seq_len)
    from utils.train_utils import sample_pose_batch
    return lambda n: sample_pose_batch(n, task['start_dist_params'], task['goal_dist_params'],
                                       task['action_dist_params'], seq_len=seq_len, device=device)


@dataclass
class Condition:
    """One conditioning input: the action tokens fed to the model, which of them are replaced
    by the null token, and the goal modes consistent with the tokens left visible."""
    tokens: torch.Tensor    # [M, obs_dim]
    mask: torch.Tensor      # [M] bool, True = replaced by the null token
    valid_modes: List[int]  # ascending mode indices

    def obs(self, batch_size, device):
        return self.tokens.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    def obs_mask(self, batch_size, device):
        """[batch, M] mask for inference, or None when every token is visible."""
        if not self.mask.any():
            return None
        return self.mask.unsqueeze(0).repeat(batch_size, 1).to(device)


def task_conditions(action_dist_params):
    """(full, partial) conditions of a conditional pose task.

    full:    one per distinct token list, in order of first appearance over the modes. Its
             valid modes are every mode named by that list; several make it multimodal.
    partial: one-token conditions, with one token position kept and the rest nulled, as in
             training-time token dropout. One per distinct token value, kept only when it
             is consistent with two or more modes (otherwise it adds nothing over `full`).
    """
    tokens = torch.tensor(action_dist_params['mu'], dtype=torch.float32)  # [K, M, obs_dim]
    K, M, _ = tokens.shape

    full = []
    for k in range(K):
        if any(torch.equal(tokens[k], c.tokens) for c in full):
            continue
        valid = [i for i in range(K) if torch.equal(tokens[i], tokens[k])]
        full.append(Condition(tokens[k], torch.zeros(M, dtype=torch.bool), valid))

    partial = []
    for j in range(M if M > 1 else 0):
        seen = []
        for k in range(K):
            if any(torch.equal(tokens[k, j], s) for s in seen):
                continue
            seen.append(tokens[k, j])
            valid = [i for i in range(K) if torch.equal(tokens[i, j], tokens[k, j])]
            if len(valid) < 2:
                continue
            mask = torch.ones(M, dtype=torch.bool)
            mask[j] = False
            partial.append(Condition(tokens[k], mask, valid))
    return full, partial
