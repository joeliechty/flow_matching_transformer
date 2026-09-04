"""Quantitative metrics for the flow-matching ablation study.

Pose metrics work in SE(3) Lie-algebra (twist) space — sample-to-mode distance is
the norm of the twist required to move between them, matching the geodesic
distance the model is trained against in `geodesic_optimal_transport_pairing`.
MNIST metrics rely on a small CNN oracle trained on real MNIST (see
`utils.mnist_classifier`).
"""
from collections import Counter
from typing import Tuple

import torch
import torch.nn.functional as F

from utils.tf_utils import compute_twist_between_poses, convert_twist_to_pose


# -- pose mode bookkeeping ----------------------------------------------------

# Index → name. Must match the goal_dist_params order in pose_gen_trainer.py:34-48.
MODE_NAMES = ("top_right", "bottom_right", "top_left", "bottom_left")

# (top/bottom, left/right) → mode index, matching the action_dist_params layout
# in pose_gen_trainer.py:56-64.
ACTION_TO_MODE = {
    ("top", "right"): 0,
    ("bottom", "right"): 1,
    ("top", "left"): 2,
    ("bottom", "left"): 3,
}


def goal_mode_poses_from_config(goal_dist_params, device='cpu') -> torch.Tensor:
    """Convert the K goal modes (stored as twist means in the config) to quaternion poses [K, 7].

    `goal_dist_params['mu']` is shape `[K, 1, 6]` (seq_len=1) per the pose trainer.
    """
    mus = torch.tensor(goal_dist_params['mu'], device=device, dtype=torch.float32)
    mode_twists = mus.squeeze(1) if mus.dim() == 3 else mus  # [K, 6]
    return convert_twist_to_pose(mode_twists, dt=1.0, return_representation='quat')  # [K, 7]


def actions_to_mode_indices(action_pairs) -> torch.Tensor:
    """Map a list of (vertical, horizontal) action token pairs to mode indices.

    `action_pairs`: iterable of 2-tuples like ('top', 'right'). Returns a LongTensor [N].
    """
    return torch.tensor([ACTION_TO_MODE[tuple(p)] for p in action_pairs], dtype=torch.long)


# -- core twist-space distance ------------------------------------------------

def _twist_distance_matrix(sample_poses: torch.Tensor, mode_poses: torch.Tensor) -> torch.Tensor:
    """Norm of the twist from each sample to each mode. [N, 7], [K, 7] -> [N, K]."""
    N, K = sample_poses.shape[0], mode_poses.shape[0]
    s = sample_poses.unsqueeze(1).expand(N, K, 7).reshape(N * K, 7)
    m = mode_poses.unsqueeze(0).expand(N, K, 7).reshape(N * K, 7)
    twists = compute_twist_between_poses(s, m, dt=1.0)  # [N*K, 6]
    return twists.norm(dim=-1).reshape(N, K)            # [N, K]


def nearest_mode_indices(sample_poses: torch.Tensor, mode_poses: torch.Tensor) -> torch.Tensor:
    """Assign each sample to its nearest mode in twist space. Returns LongTensor [N]."""
    return _twist_distance_matrix(sample_poses, mode_poses).argmin(dim=-1)


# -- pose metrics -------------------------------------------------------------

def pose_mode_accuracy(sample_poses: torch.Tensor,
                      target_mode_idx: torch.Tensor,
                      mode_poses: torch.Tensor) -> float:
    """Fraction of samples whose nearest mode (in twist space) matches the conditioning."""
    assigned = nearest_mode_indices(sample_poses, mode_poses)
    return (assigned == target_mode_idx.to(assigned.device)).float().mean().item()


def pose_mode_coverage_kl(sample_poses: torch.Tensor,
                         mode_poses: torch.Tensor) -> Tuple[float, Counter]:
    """KL(empirical mode histogram || uniform). Lower = more uniform coverage.

    Also returns a Counter of per-mode hit counts so a missed mode is visible.
    """
    K = mode_poses.shape[0]
    assigned = nearest_mode_indices(sample_poses, mode_poses).cpu().tolist()
    counts = Counter(assigned)
    total = len(assigned)
    # Add-one smoothing so a zero-hit mode doesn't break the log.
    probs = torch.tensor([(counts.get(k, 0) + 1) / (total + K) for k in range(K)])
    uniform = torch.full((K,), 1.0 / K)
    kl = (probs * (probs.log() - uniform.log())).sum().item()
    return kl, counts


# -- MNIST metrics ------------------------------------------------------------

@torch.no_grad()
def _classify(samples_28x28: torch.Tensor, classifier) -> torch.Tensor:
    """Run the classifier and return predicted class indices [N]."""
    device = next(classifier.parameters()).device
    x = samples_28x28.to(device)
    if x.dim() == 3:           # [N, 28, 28] -> [N, 1, 28, 28]
        x = x.unsqueeze(1)
    logits = classifier(x)
    return logits.argmax(dim=-1)


def mnist_class_accuracy(samples_28x28: torch.Tensor,
                        labels: torch.Tensor,
                        classifier) -> float:
    preds = _classify(samples_28x28, classifier)
    return (preds == labels.to(preds.device)).float().mean().item()


def mnist_class_marginal_kl(samples_28x28: torch.Tensor,
                           classifier,
                           num_classes: int = 10) -> Tuple[float, Counter]:
    """KL(empirical class histogram || uniform). Lower = better class coverage."""
    preds = _classify(samples_28x28, classifier).cpu().tolist()
    counts = Counter(preds)
    total = len(preds)
    probs = torch.tensor([(counts.get(k, 0) + 1) / (total + num_classes) for k in range(num_classes)])
    uniform = torch.full((num_classes,), 1.0 / num_classes)
    kl = (probs * (probs.log() - uniform.log())).sum().item()
    return kl, counts
