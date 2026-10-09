"""Pose-recreation metrics for grasp poses: how closely generated grasps reproduce a held-out
set of real grasps of the same object.

Every metric uses one distance between two grasps: the mean Euclidean distance between the
Franka hand's 5 control points (base, finger bases and fingertips [the ACRONYM gripper marker;
Mousavian et al. 2019]), minimised over the jaw swap, since a parallel-jaw grasp turned half a
turn about its approach axis is the same grasp. It is in cm and joins position and orientation
in one physical unit: 1 cm means the gripper geometry is off by about 1 cm.

  energy_distance    the whole-distribution match (unbiased), as in the synthetic study
  recreation_err     mean distance from each real grasp to its nearest sample: how well the
                     real grasps are reproduced (collapse scores badly)
  fidelity_err       mean distance from each sample to its nearest real grasp: are the
                     samples realistic (over-dispersion scores badly)
  coverage_<τ>cm     share of real grasps with a sample within τ
  precision_<τ>cm    share of samples with a real grasp within τ
  one_nna            1-nearest-neighbour two-sample accuracy on equal-size sets; 0.5 means
                     the sets can't be told apart [Lopez-Paz & Oquab 2017]
  spread_ratio_trans / _rot   mean pairwise translation / rotation distance of the samples
                     over that of the real grasps (1 = the data's spread)
"""
import torch

from utils.tf_utils import _quat_to_rot_mat

# Franka hand control points (m) in the hand's base frame; the fingers lie along ±x
HAND_POINTS = ((0.0, 0.0, 0.0), (0.041, 0.0, 0.066), (-0.041, 0.0, 0.066),
               (0.041, 0.0, 0.112), (-0.041, 0.0, 0.112))
JAW_SWAP = (0, 2, 1, 4, 3)
TAUS_CM = (1.0, 2.0)
CM = 100.0


def control_points(poses, tcp_offset):
    """Poses [N, 7] in metres, whose frame sits tcp_offset along z from the hand's base ->
    control points [N, 5, 3] in metres."""
    pts = torch.tensor(HAND_POINTS, device=poses.device, dtype=poses.dtype)
    pts = pts - pts.new_tensor([0.0, 0.0, tcp_offset])
    R = _quat_to_rot_mat(poses[:, 3:7])
    return torch.einsum('nij,kj->nki', R, pts) + poses[:, None, :3]


def pose_distances(a, b, tcp_offset, chunk=256):
    """Jaw-swap-invariant control-point distances [N, M] in cm between poses a [N, 7] and
    b [M, 7] (metres)."""
    ca, cb = control_points(a, tcp_offset), control_points(b, tcp_offset)
    cb_swapped = cb[:, list(JAW_SWAP)]
    out = []
    for i in range(0, ca.shape[0], chunk):
        x = ca[i:i + chunk, None]                                   # [n, 1, 5, 3]
        d = (x - cb[None]).norm(dim=-1).mean(-1)
        d_swapped = (x - cb_swapped[None]).norm(dim=-1).mean(-1)
        out.append(torch.minimum(d, d_swapped))
    return torch.cat(out) * CM


def _rotation_distances(a, b):
    """Jaw-swap-invariant rotation angles [N, M] in radians between poses a and b."""
    Ra, Rb = _quat_to_rot_mat(a[:, 3:7]), _quat_to_rot_mat(b[:, 3:7])
    tr = torch.einsum('nij,mij->nm', Ra, Rb)
    # the half turn about z negates the x and y columns of Rb
    tr_swapped = torch.einsum('nij,mij->nm', Ra[..., 2:], Rb[..., 2:]) * 2 - tr
    cos = (torch.maximum(tr, tr_swapped) - 1) / 2
    return torch.acos(cos.clamp(-1.0, 1.0))


def _mean_off_diagonal(d):
    n = d.shape[0]
    return (d.sum() - d.diagonal().sum()) / (n * (n - 1))


def energy_distance(dxy, dxx, dyy):
    """Unbiased energy distance 2·E|X−Y| − E|X−X'| − E|Y−Y'| from distance matrices."""
    return (2 * dxy.mean() - _mean_off_diagonal(dxx) - _mean_off_diagonal(dyy)).item()


def one_nn_accuracy(dxy, dxx, dyy):
    """Leave-one-out 1-NN accuracy of telling X from Y on the first min(N, M) of each."""
    n = min(dxx.shape[0], dyy.shape[0])
    dxy, dxx, dyy = dxy[:n, :n], dxx[:n, :n].clone(), dyy[:n, :n].clone()
    dxx.fill_diagonal_(float('inf'))
    dyy.fill_diagonal_(float('inf'))
    x_right = dxx.min(1).values < dxy.min(1).values
    y_right = dyy.min(1).values < dxy.min(0).values
    return (x_right.float().sum() + y_right.float().sum()).item() / (2 * n)


def recreation_metrics(samples, reference, tcp_offset):
    """Every metric of the module docstring for one object: samples [N, 7] against its real
    reference grasps [M, 7], both in metres."""
    dxy = pose_distances(samples, reference, tcp_offset)
    dxx = pose_distances(samples, samples, tcp_offset)
    dyy = pose_distances(reference, reference, tcp_offset)
    nearest_sample, nearest_real = dxy.min(0).values, dxy.min(1).values
    out = {'energy_distance': energy_distance(dxy, dxx, dyy),
           'recreation_err': nearest_sample.mean().item(),
           'fidelity_err': nearest_real.mean().item()}
    for tau in TAUS_CM:
        out[f'coverage_{tau:g}cm'] = (nearest_sample < tau).float().mean().item()
        out[f'precision_{tau:g}cm'] = (nearest_real < tau).float().mean().item()
    out['one_nna'] = one_nn_accuracy(dxy, dxx, dyy)
    trans_x = torch.cdist(samples[:, :3], samples[:, :3])
    trans_y = torch.cdist(reference[:, :3], reference[:, :3])
    out['spread_ratio_trans'] = (_mean_off_diagonal(trans_x) / _mean_off_diagonal(trans_y)).item()
    out['spread_ratio_rot'] = (_mean_off_diagonal(_rotation_distances(samples, samples))
                               / _mean_off_diagonal(_rotation_distances(reference, reference))).item()
    return out


RECREATION_METRICS = ('energy_distance', 'recreation_err', 'fidelity_err',
                      *[f'{k}_{t:g}cm' for t in TAUS_CM for k in ('coverage', 'precision')],
                      'one_nna', 'spread_ratio_trans', 'spread_ratio_rot')


def mean_over_objects(per_object):
    """Unweighted mean of each metric over a list of per-object metric dicts."""
    return {k: sum(m[k] for m in per_object) / len(per_object) for k in per_object[0]}
