"""Evaluate toy checkpoints (toy_gen_trainer.py): W₂² to the target at several sampling
budgets, plus branch metrics for the fork.

For each checkpoint in --checkpoint_dir: conditions come from real target draws (set A), the
model generates one sample per condition from fresh source noise, and W₂² compares the joint
samples (see `utils.toy_tasks.joint`) with an independent real set B by exact assignment.
A 'data' row scores set A itself, the finite-sample floor. Samplers: Euler at several step
counts and adaptive RK45 (scipy, NFE recorded).

    python experiments/evaluate_toys.py --checkpoint_dir checkpoints/toy_moons/seed_1 \
        --output experiments/results/toy_moons/seed_1/toy_metrics.csv
    python experiments/evaluate_toys.py --aggregate experiments/results/toy_moons
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import write_csv
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from utils.logging_utils import git_commit
from utils.toy_tasks import joint, sample_source, sample_targets

EULER_STEPS = (1, 2, 4, 8, 100)
NUM_SAMPLES = 4096
SEED_A, SEED_B, NOISE_SEED = 11, 12, 13  # own generators: every model sees the same draws
METRICS = ('w2', 'nfe', 'branch_balance', 'on_branch')
VARIANT_ORDER = ('data', 'NOOT', 'GOT', 'C2OT', 'C2OTFIX', 'CLUSTER')


def w2_squared(x, y):
    """Exact squared 2-Wasserstein distance between two equal-size point sets."""
    cost = torch.cdist(x.double(), y.double()).pow(2).cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def fork_metrics(cond, y):
    """Branch balance: share of samples above 0 where x > 0.2 (the data's is 0.5). On-branch:
    share within 0.1 of a branch, over |x| > 0.2 (y = 0 for x < 0, |y| = x for x > 0)."""
    x, y = cond[:, 0], y[:, 0]
    pos, neg = x > 0.2, x < -0.2
    on = torch.cat([((y[pos].abs() - x[pos]).abs() < 0.1), (y[neg].abs() < 0.1)])
    return (y[pos] > 0).float().mean().item(), on.float().mean().item()


@torch.no_grad()
def generate_euler(model, x0, obs, steps):
    return model.inference(x0.unsqueeze(1), obs, num_steps=steps, cfg_scale=1.0,
                           manifold='euclidean').squeeze(1)


@torch.no_grad()
def generate_rk45(model, x0, obs, tol=1e-5):
    """Adaptive RK45 on the whole batch as one ODE system; returns samples and NFE."""
    n, dim = x0.shape

    def velocity(t, y):
        x = torch.as_tensor(y, dtype=torch.float32, device=x0.device).view(n, 1, dim)
        v = model(x, obs, torch.full((n,), t, device=x0.device))
        return v.flatten().double().cpu().numpy()

    sol = solve_ivp(velocity, (0.0, 1.0), x0.flatten().double().cpu().numpy(), method='RK45',
                    rtol=tol, atol=tol)
    return torch.as_tensor(sol.y[:, -1], dtype=torch.float32, device=x0.device).view(n, dim), sol.nfev


def score(toy, samples, cond, ref):
    row = {'w2': w2_squared(joint(toy, samples, cond), ref)}
    if toy == 'fork':
        row['branch_balance'], row['on_branch'] = fork_metrics(cond, samples)
    return row


def evaluate_dir(ckpt_dir, device, num_samples=NUM_SAMPLES):
    rows = []
    for path in sorted(ckpt_dir.glob('toy_*_final.pt')):
        config = OmegaConf.load(str(path).replace('_final.pt', '_training_config.yaml'))
        toy, variant = config.training.toy, path.name[len('toy_'):-len('_final.pt')].split('_', 1)[1]
        target_a, cond = sample_targets(toy, num_samples, torch.Generator().manual_seed(SEED_A))
        target_b, cond_b = sample_targets(toy, num_samples, torch.Generator().manual_seed(SEED_B))
        ref = joint(toy, target_b, cond_b).to(device)
        cond = cond.to(device)
        x0 = sample_source(toy, num_samples, torch.Generator().manual_seed(NOISE_SEED)).to(device)
        base = {'toy': toy, 'seed': config.training.seed, 'git_commit': git_commit()}
        if not rows:
            rows.append({**base, 'variant': 'data', 'solver': 'data', 'num_steps': None,
                         **score(toy, target_a.to(device), cond, ref)})
        model, _ = ConditionalFlowMatchingTransformerModel.load_checkpoint(str(path), device=device)
        model.eval()
        obs = cond.unsqueeze(1)
        for steps in EULER_STEPS:
            row = {**base, 'variant': variant, 'solver': 'euler', 'num_steps': steps, 'nfe': steps,
                   **score(toy, generate_euler(model, x0, obs, steps), cond, ref)}
            rows.append(row)
            print({k: v for k, v in row.items() if v is not None})
        samples, nfe = generate_rk45(model, x0, obs)
        row = {**base, 'variant': variant, 'solver': 'rk45', 'num_steps': None, 'nfe': nfe,
               **score(toy, samples, cond, ref)}
        rows.append(row)
        print({k: v for k, v in row.items() if v is not None})
    return rows


def aggregate(results_dir):
    """Mean and std over seeds per (variant, solver, steps) -> toy_metrics_summary.csv."""
    groups = {}
    for path in sorted(results_dir.glob('seed_*/toy_metrics.csv')):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r['toy'], r['variant'], r['solver'], r['num_steps'])
                groups.setdefault(key, []).append(r)
    out = []
    for (toy, variant, solver, steps), members in groups.items():
        s = {'toy': toy, 'variant': variant, 'solver': solver, 'num_steps': steps,
             'n_seeds': len(members)}
        for m in METRICS:
            vals = [float(r[m]) for r in members if r.get(m)]
            if vals:
                s[f'{m}_mean'] = statistics.fmean(vals)
                s[f'{m}_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(s)
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    out.sort(key=lambda s: (order.get(s['variant'], 99), s['solver'] != 'euler',
                            int(s['num_steps'] or 0)))
    fields = ['toy', 'variant', 'solver', 'num_steps', 'n_seeds'] + [
        f'{m}_{stat}' for m in METRICS for stat in ('mean', 'std') if any(f'{m}_mean' in s for s in out)]
    write_csv(out, results_dir / 'toy_metrics_summary.csv', fieldnames=fields)
    for s in out:
        print(f"{s['variant']:>8} {s['solver']:>5} {s['num_steps'] or '':>4}  "
              + "  ".join(f"{m} {s[f'{m}_mean']:.4g}±{s[f'{m}_std']:.2g}"
                          for m in METRICS if f'{m}_mean' in s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--aggregate', type=str, help='results dir with seed_*/toy_metrics.csv')
    args = parser.parse_args()
    if args.aggregate:
        aggregate(Path(args.aggregate))
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = evaluate_dir(Path(args.checkpoint_dir), device, args.num_samples)
    write_csv(rows, args.output)


if __name__ == '__main__':
    main()
