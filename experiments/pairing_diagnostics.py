"""Pairing diagnostics, no training: does a pairing shorten paths, and does it skew the noise?

For each pairing, over many OT batches of a task:
  cost_ratio     mean pairing cost after ÷ before (1 = unchanged, i.e. random pairing)
  cond_shift     mean squared distance between the condition a noise sample started with
                 and the one it ends up paired with, ÷ the mean over random pairs
                 (0 = stays with its condition, 1 = as if conditions were ignored)
  prior_skew_r2  held-out R² of a quadratic regression predicting the data's condition from
                 the paired noise. ~0 means each condition still sees the whole noise
                 distribution; higher means the noise a condition trains on depends on the
                 condition, which sampling (noise drawn independently of the condition) never
                 reproduces.
  branch_agreement  for tasks with two branches per condition (the pose tasks' two
                 orientations, the fork's two signs): the share of pairs whose noise points to
                 the data's branch (sign of the noise's z rotation, or of the fork noise). 0.5 is
                 random pairing; this is what lets few-step samples keep both branches.
  w, r, gamma    the chosen condition weights (c2ot: w and the realised admissible ratio r).

    python experiments/pairing_diagnostics.py --task moons
    python experiments/pairing_diagnostics.py --task configs/pose_tasks/continuous_goals.yaml --ot_batch 128 512 1280
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import write_csv
from utils.pose_task import load_pose_task, pose_batch_sampler, task_conditions
from utils.toy_tasks import TOYS, toy_batch_sampler
from utils.train_utils import Pairer, _batch_cost, _sq_dists, condition_aware_pairing

TOY_METHODS = ('independent', 'global', 'c2ot', 'c2ot_fixed', 'cluster')


def branch_agreement_fn(task, spec):
    """(paired noise, goal, obs, cond_ids) -> share of pairs whose noise points to the data's
    branch, for tasks with two branches per condition; None for the others."""
    if task == 'fork':
        def fork(x0, x1, obs, ids):
            pos = obs[:, 0, 0] > 0.2
            return ((x0[pos, 0, 0] > 0) == (x1[pos, 0, 0] > 0)).float().mean().item()
        return fork
    if spec is None:
        return None
    if spec['type'] == 'continuous':
        t = spec['task']
        mid = lambda obs, ids: t.base_yaw + t.yaw_gain * obs[:, 0, 0]       # obs = c / R
    else:
        full, _ = task_conditions(spec['action_dist_params'])
        if any(len(c.valid_modes) != 2 for c in full):
            return None
        yaw = torch.tensor(spec['goal_dist_params']['mu'])[:, 0, 5]
        # condition ids index the distinct token lists in order, as `task_conditions` does
        mids = torch.stack([yaw[c.valid_modes].mean() for c in full])
        mid = lambda obs, ids: mids.to(ids.device)[ids]

    def yaw_branch(x0, x1, obs, ids):
        return ((x0[:, 0, 5] > 0) == (x1[:, 0, 5] > mid(obs, ids))).float().mean().item()
    return yaw_branch


def make_sampler(task, device):
    """(name, sampler, manifold, ot_mode, methods, default OT batch, branch agreement) for a toy
    name or a pose task file."""
    if task in TOYS:
        return (f'toy_{task}', toy_batch_sampler(task, device), 'euclidean', 'flat', TOY_METHODS,
                1024, branch_agreement_fn(task, None))
    spec = load_pose_task(task)
    # discrete-token tasks also have true condition ids, so per-condition OT ('ot') is defined
    methods = TOY_METHODS if spec['type'] == 'continuous' else ('ot',) + TOY_METHODS
    return (f"pose_{spec['name']}", pose_batch_sampler(spec, device=device), 'se3', 'per_frame',
            methods, 512, branch_agreement_fn(task, spec))


def quadratic(x):
    """Features [x, x_i x_j (i <= j)], so the skew fit also catches symmetric dependence (the
    fork's noise magnitude, not its sign, tells the condition)."""
    i, j = torch.triu_indices(x.shape[1], x.shape[1])
    return torch.cat([x, x[:, i] * x[:, j]], dim=1)


def r_squared(x_train, y_train, x_test, y_test):
    """Mean over condition dims of held-out R² for a linear fit (with intercept) of y on x."""
    ones = lambda x: torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    x_train, x_test = quadratic(x_train.double()), quadratic(x_test.double())
    beta = torch.linalg.lstsq(ones(x_train), y_train.double()).solution
    resid = y_test.double() - ones(x_test) @ beta
    var = (y_test.double() - y_test.double().mean(0)).pow(2).sum(0)
    keep = var > 1e-12
    return (1 - resid.pow(2).sum(0)[keep] / var[keep]).mean().item()


@torch.no_grad()
def diagnose(method, sampler, ot_batch, manifold, ot_mode, batches, r_tar, branch_fn=None):
    pairer = Pairer(method, sampler, ot_batch, manifold=manifold, ot_mode=ot_mode, r_tar=r_tar)
    w = pairer.w
    ratios, shifts, infos, xs, cs, agree = [], [], [], [], [], []
    for _ in range(batches):
        start, goal, obs, cond_ids = sampler(ot_batch)
        cond = obs.flatten(1)
        paired, info = condition_aware_pairing(
            start, goal, method, cond=cond, cond_ids=cond_ids, manifold=manifold,
            ot_mode=ot_mode, r_tar=r_tar, w=w, centroids=pairer.centroids)
        if method == 'c2ot':
            w = info['w']
        before = _batch_cost(start, goal, manifold, ot_mode).diagonal().mean()
        after = _batch_cost(paired, goal, manifold, ot_mode).diagonal().mean()
        ratios.append((after / before).item())
        # which noise row each data row got: rows are exact copies of start rows
        src = torch.cdist(paired.flatten(1), start.flatten(1)).argmin(dim=1)
        d = _sq_dists(cond, cond)
        shifts.append((d[src, torch.arange(len(src), device=d.device)].mean() / d.mean()).item())
        infos.append(info)
        xs.append(paired.flatten(1))
        cs.append(cond)
        if branch_fn is not None:
            agree.append(branch_fn(paired, goal, obs, cond_ids))
    split = int(0.8 * batches)
    row = {'method': method, 'r_tar': r_tar if method == 'c2ot' else None,
           'ot_batch': ot_batch, 'cost_ratio': statistics.fmean(ratios),
           'cond_shift': statistics.fmean(shifts),
           'prior_skew_r2': r_squared(torch.cat(xs[:split]), torch.cat(cs[:split]),
                                      torch.cat(xs[split:]), torch.cat(cs[split:])),
           'branch_agreement': statistics.fmean(agree) if agree else None}
    for key in ('w', 'r', 'gamma'):
        vals = [i[key] for i in infos if key in i]
        if vals:
            row[key] = statistics.fmean(vals)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help=f'toy name {TOYS} or a pose task file')
    parser.add_argument('--ot_batch', type=int, nargs='+', default=None,
                        help='OT batch sizes (default: 1024 for toys, 512 for poses)')
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--r_tar', type=float, nargs='+', default=[0.01],
                        help='c2ot target ratios (one c2ot row each)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='default: experiments/results/<task>/pairing_diagnostics.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name, sampler, manifold, ot_mode, methods, default_batch, branch_fn = make_sampler(args.task, device)
    rows = []
    for ot_batch in args.ot_batch or [default_batch]:
        for method in methods:
            for r_tar in (args.r_tar if method == 'c2ot' else args.r_tar[:1]):
                torch.manual_seed(args.seed)
                row = {'task': name, **diagnose(method, sampler, ot_batch, manifold, ot_mode,
                                                 args.batches, r_tar, branch_fn)}
                rows.append(row)
                print(f"{name} B={ot_batch:>5} {method:>11}: " + "  ".join(
                    f"{k} {v:.3g}" for k, v in row.items()
                    if k not in ('task', 'method', 'ot_batch') and v is not None))
    output = args.output or REPO_ROOT / 'experiments' / 'results' / name / 'pairing_diagnostics.csv'
    write_csv(rows, output, fieldnames=['task', 'method', 'r_tar', 'ot_batch', 'cost_ratio',
                                        'cond_shift', 'prior_skew_r2', 'branch_agreement',
                                        'w', 'r', 'gamma'])


if __name__ == '__main__':
    main()
