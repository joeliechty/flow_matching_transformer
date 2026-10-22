"""Quantitative metrics for the flow-matching ablation study.

Pose metrics work in SE(3) Lie-algebra (twist) space — sample-to-mode distance is
the norm of the twist required to move between them, matching the geodesic
distance the model is trained against in `geodesic_optimal_transport_pairing`.
The distribution-level pose metrics (bias/spread, energy distance) instead keep
translation and rotation apart or embed them explicitly, and compare against real
goal samples drawn by `sample_goal_poses`.
MNIST metrics rely on a small CNN oracle trained on real MNIST (see
`utils.mnist_classifier`).
"""
from collections import Counter
from typing import Tuple

import torch
import torch.nn.functional as F

from utils.tf_utils import (_quat_to_rot_mat, _rot_mat_to_quat, compute_twist_between_poses,
                            convert_twist_to_pose)


# -- pose mode bookkeeping ----------------------------------------------------
# Modes and the conditions that name them come from the task (`utils.pose_task`); here a
# mode is just its index into the training config's goal_dist_params.

def goal_mode_poses_from_config(goal_dist_params, device='cpu') -> torch.Tensor:
    """Convert the K goal modes (stored as twist means in the config) to quaternion poses [K, 7].

    `goal_dist_params['mu']` is shape `[K, 1, 6]` (seq_len=1) per the pose trainer.
    """
    mus = torch.tensor(goal_dist_params['mu'], device=device, dtype=torch.float32)
    mode_twists = mus.squeeze(1) if mus.dim() == 3 else mus  # [K, 6]
    return convert_twist_to_pose(mode_twists, dt=1.0, return_representation='quat')  # [K, 7]


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


def pose_mode_distance(sample_poses: torch.Tensor,
                       mode_poses: torch.Tensor,
                       target_mode_idx: torch.Tensor = None) -> float:
    """Mean twist-space distance from each sample to its target mode (or to its nearest
    mode when no target is given). Lower is better. Unlike mode accuracy it keeps
    resolving differences once every sample lands in the right basin.
    """
    dist = _twist_distance_matrix(sample_poses, mode_poses)  # [N, K]
    if target_mode_idx is None:
        return dist.min(dim=-1).values.mean().item()
    idx = target_mode_idx.to(dist.device).view(-1, 1)
    return dist.gather(1, idx).mean().item()


def mode_balance_kl(assigned: torch.Tensor, modes) -> float:
    """KL(histogram over `modes` || uniform), add-one smoothed like `pose_mode_coverage_kl`.

    Only samples assigned to one of `modes` are counted; how many landed elsewhere is a
    separate question (validity).
    """
    counts = [(assigned == m).sum().item() for m in modes]
    total, K = sum(counts), len(modes)
    probs = torch.tensor([(c + 1) / (total + K) for c in counts])
    return (probs * (probs.log() - torch.log(torch.tensor(1.0 / K)))).sum().item()


# -- distribution-level pose metrics --------------------------------------------

def sample_goal_poses(goal_dist_params, n_per_mode, seed, device='cpu'):
    """Real goal poses: `n_per_mode` draws from each mode of the training distribution
    (Gaussian in twist space, as in `sample_random_twist`).

    Uses its own generator, so the global RNG (and with it the model-sampling noise) is
    untouched. Returns poses [K * n_per_mode, 7] and their mode indices [K * n_per_mode].
    """
    mu, sigma = _goal_params(goal_dist_params)
    K = mu.shape[0]
    eps = torch.randn(K, n_per_mode, 6, generator=torch.Generator().manual_seed(seed))
    twists = (mu[:, None] + sigma[:, None] * eps).reshape(-1, 6)
    poses = convert_twist_to_pose(twists, dt=1.0, return_representation='quat')
    return poses.to(device), torch.arange(K).repeat_interleave(n_per_mode).to(device)


def sample_goal_mixture(goal_dist_params, n, seed, modes=None, device='cpu'):
    """`n` real goal poses, each from a mode drawn uniformly from `modes` (default: all).

    The split across modes is random, as from a perfect sampler, rather than exactly
    balanced, so balance/coverage KLs get their true sampling-noise floor. Own generator,
    as in `sample_goal_poses`. Returns poses [n, 7] and mode indices [n].
    """
    mu, sigma = _goal_params(goal_dist_params)
    gen = torch.Generator().manual_seed(seed)
    modes = torch.arange(mu.shape[0]) if modes is None else torch.as_tensor(modes).cpu()
    idx = modes[torch.randint(len(modes), (n,), generator=gen)]
    twists = mu[idx] + sigma[idx] * torch.randn(n, 6, generator=gen)
    poses = convert_twist_to_pose(twists, dt=1.0, return_representation='quat')
    return poses.to(device), idx.to(device)


def _goal_params(goal_dist_params):
    """Per-mode twist mean and std, [K, 6] each (seq_len = 1)."""
    mu = torch.tensor(goal_dist_params['mu'], dtype=torch.float32).reshape(-1, 6)
    sigma = torch.tensor(goal_dist_params['sigma'], dtype=torch.float32).reshape(-1, 6)
    return mu, sigma


def _pose_parts(poses):
    """[N, 7] (x, y, z, qw, qx, qy, qz) -> positions [N, 3], rotation matrices [N, 3, 3]."""
    return poses[:, :3], _quat_to_rot_mat(poses[:, 3:])


def _rotvec(R):
    """Rotation matrices [N, 3, 3] -> axis-angle vectors [N, 3] in radians."""
    q = _rot_mat_to_quat(R)                    # w first
    q = torch.where(q[:, :1] < 0, -q, q)       # q and -q are the same rotation
    w, v = q[:, 0], q[:, 1:]
    n = v.norm(dim=-1)
    angle = 2 * torch.atan2(n, w)
    scale = torch.where(n > 1e-8, angle / n.clamp_min(1e-12), torch.full_like(n, 2.0))
    return v * scale.unsqueeze(-1)


def mode_offsets(sample_poses, mode_pose):
    """Offsets of samples from one mode centre in separate units: translation in world
    coordinates [N, 3], and rotation as the axis-angle of R_mode^T R_sample [N, 3]."""
    p, R = _pose_parts(sample_poses)
    pm, Rm = _pose_parts(mode_pose.reshape(1, 7))
    return p - pm, _rotvec(Rm.transpose(-1, -2) @ R)


def _bias_and_spread(offsets):
    """Norm of the mean offset, and the RMS per-axis standard deviation around it."""
    mean = offsets.mean(dim=0)
    spread = ((offsets - mean).pow(2).sum(dim=-1).mean() / offsets.shape[-1]).sqrt()
    return mean.norm().item(), spread.item()


def pose_bias_spread(sample_poses, assigned, mode_poses, ref_poses, ref_assigned):
    """Bias and spread of samples around their mode centre, split into translation and
    rotation, averaged over modes weighted by sample count.

    `assigned` gives each sample's mode (the target for conditional models, the nearest
    mode otherwise); `ref_poses` / `ref_assigned` are real goal samples. Spread is
    reported relative to the real samples': 1 = matches the data, < 1 = collapsed.
    """
    keys = ('bias_trans', 'bias_rot', 'spread_ratio_trans', 'spread_ratio_rot')
    stats, weights = [], []
    for k in range(mode_poses.shape[0]):
        sel = assigned == k
        if sel.sum() < 2:
            continue
        dp, dr = mode_offsets(sample_poses[sel], mode_poses[k])
        rp, rr = mode_offsets(ref_poses[ref_assigned == k], mode_poses[k])
        (bt, st), (br, sr) = _bias_and_spread(dp), _bias_and_spread(dr)
        stats.append((bt, br, st / _bias_and_spread(rp)[1], sr / _bias_and_spread(rr)[1]))
        weights.append(sel.sum().item())
    if not stats:
        return dict.fromkeys(keys)
    w = torch.tensor(weights, dtype=torch.float64)
    vals = (torch.tensor(stats, dtype=torch.float64) * w[:, None]).sum(0) / w.sum()
    return dict(zip(keys, vals.tolist()))


def _pose_features(poses):
    """Euclidean embedding [position, vec(R) / √2]. For small rotations the rotation part
    of a distance is ≈ the angle in radians, so it weighs in like a twist norm."""
    p, R = _pose_parts(poses)
    return torch.cat([p, R.flatten(1) / 2 ** 0.5], dim=-1)


def energy_distance(x_poses, y_poses) -> float:
    """Energy distance between two pose sets: unbiased estimate of
    2·E|X−Y| − E|X−X'| − E|Y−Y'| in the `_pose_features` embedding.

    0 when the distributions match (small negative values are sampling noise). It grows
    with any mismatch in location or spread, so collapse is penalised as well as error.
    """
    fx, fy = _pose_features(x_poses), _pose_features(y_poses)
    exact = 'donot_use_mm_for_euclid_dist'  # the matmul shortcut is off by ~1e-3 on identical rows

    def within(f):
        n = f.shape[0]
        return torch.cdist(f, f, compute_mode=exact).sum() / (n * (n - 1))

    cross = torch.cdist(fx, fy, compute_mode=exact).mean()
    return (2 * cross - within(fx) - within(fy)).item()


def per_mode_energy_distance(sample_poses, assigned, ref_poses, ref_assigned):
    """`energy_distance` between samples and real goal samples within each mode, averaged
    over modes weighted by sample count (modes with < 2 samples are skipped).

    Per mode because over the whole mixture the large distances between modes swamp
    within-mode differences (collapse or excess spread shrink by roughly 1/K). How mass
    is split across modes is measured separately (coverage KL, partial-condition balance).
    """
    vals, weights = [], []
    for k in assigned.unique().tolist():
        sel = assigned == k
        if sel.sum() < 2:
            continue
        vals.append(energy_distance(sample_poses[sel], ref_poses[ref_assigned == k]))
        weights.append(sel.sum().item())
    if not vals:
        return None
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def path_straightness(trajectory) -> Tuple[float, float]:
    """Geometry of sampling paths [N, T+1, 7].

    Returns (mean path length / start-to-end distance, mean start-to-end distance), both as
    twist norms: the geometry of the training interpolation, along which a path scores
    exactly 1. Above 1 means the sampler's path curves.
    """
    N, T1, _ = trajectory.shape
    a = trajectory[:, :-1].reshape(-1, 7)
    b = trajectory[:, 1:].reshape(-1, 7)
    length = compute_twist_between_poses(a, b, dt=1.0).norm(dim=-1).reshape(N, T1 - 1).sum(dim=1)
    chord = compute_twist_between_poses(trajectory[:, 0], trajectory[:, -1], dt=1.0).norm(dim=-1)
    return (length / chord.clamp_min(1e-8)).mean().item(), chord.mean().item()


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
