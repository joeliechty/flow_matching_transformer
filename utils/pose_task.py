"""Pose-generation task definitions.

A task is a YAML file (see `configs/pose_tasks/`) giving the start distribution, a vocabulary
of conditioning tokens, and the goal modes: each a Gaussian in twist space named by a list
of tokens. `load_pose_task` turns it into the distribution dicts the trainer uses.
`task_conditions` recovers the conditions from a run's saved `action_dist_params`, so
evaluation works for any task, including runs that predate task files.
"""
from dataclasses import dataclass
from typing import List

import torch
from omegaconf import OmegaConf


def _vec(value, dim):
    """A scalar broadcast to `dim` entries, or a list of length `dim` as given."""
    if isinstance(value, (int, float)):
        return [value] * dim
    if len(value) != dim:
        raise ValueError(f"expected {dim} values, got {len(value)}: {value}")
    return list(value)


def load_pose_task(path):
    """Read a task file into the trainer's distribution dicts.

    Returns a dict with 'name', 'mode_names', 'tokens' (the vocabulary), 'obs_dim',
    'start_dist_params', 'goal_dist_params' and 'action_dist_params', laid out exactly as
    the pose trainer has always built them ([K, 1, 6] goal twists, [K, M, obs_dim] tokens).
    """
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
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
