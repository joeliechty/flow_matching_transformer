"""Sweep the number of ODE integration steps used at sampling time.

Fewer steps = a coarser discretisation of each sampling path. OT pairing is
meant to straighten the learned paths, so OT variants should lose less quality as
the step count shrinks. Every model in the directory is swept; CFG variants are
sampled at --cfg_scale and NOCFG variants unguided (see `effective_cfg_scale`).
--method picks the ODE solver; the rows' `nfe` column counts network evaluations, so
solvers compare at equal cost. Output: experiments/results/steps_sweep.csv
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import (
    discover_checkpoints, evaluate, load_classifier_if_needed, variant_name, write_csv,
)
from models.flow_transformer_base import SOLVER_EVALS


DEFAULT_STEPS = (1, 2, 3, 5, 9, 20, 50, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Evaluate the checkpoints saved at this epoch (default: latest)')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--steps', type=int, nargs='+', default=list(DEFAULT_STEPS),
                        help='Network-evaluation budgets per sample (= Euler steps)')
    parser.add_argument('--method', type=str, nargs='+', default=['euler'], choices=tuple(SOLVER_EVALS),
                        help='ODE solvers to sweep; each runs --steps divided by its evaluations '
                             'per step, so every solver is compared at the same network evaluations')
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='experiments/results/steps_sweep.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    metas = list(discover_checkpoints(Path(args.checkpoint_dir), args.epoch))
    classifier = load_classifier_if_needed(metas, args.classifier_path, device)

    # (solver, steps) for every budget that is a whole number of the solver's steps
    runs = [(method, nfe // SOLVER_EVALS[method]) for method in args.method for nfe in args.steps
            if nfe % SOLVER_EVALS[method] == 0]
    rows = []
    for meta in metas:
        for method, steps in runs:
            tag = f"{meta['task']} {variant_name(meta)} @ {method}, num_steps={steps}"
            print(f"\n=== {tag} ===")
            try:
                row = evaluate(meta, device, classifier,
                               num_samples=args.num_samples, num_steps=steps,
                               cfg_scale=args.cfg_scale, eval_seed=args.eval_seed, method=method)
            except Exception as e:
                print(f"ERROR evaluating {tag}: {e}")
                continue
            if row is None:
                continue
            rows.append(row)
            print({k: v for k, v in row.items() if v is not None})

    write_csv(rows, args.output)


if __name__ == '__main__':
    main()
