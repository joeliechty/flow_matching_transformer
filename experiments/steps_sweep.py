"""Sweep the number of ODE integration steps used at sampling time.

Fewer steps = a coarser Euler discretisation of each sampling path. OT pairing is
meant to straighten the learned paths, so OT variants should lose less quality as
the step count shrinks. Every model in the directory is swept; CFG variants are
sampled at --cfg_scale and NOCFG variants unguided (see `effective_cfg_scale`).
Output: experiments/results/steps_sweep.csv
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


# Pose training only ever shows the model t = k/9 (a 10-point linspace grid), so 1, 3
# and 9 steps query trained time values only; the other counts interpolate in t.
DEFAULT_STEPS = (1, 2, 3, 5, 9, 20, 50, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--steps', type=int, nargs='+', default=list(DEFAULT_STEPS))
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='experiments/results/steps_sweep.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    metas = list(discover_checkpoints(Path(args.checkpoint_dir)))
    classifier = load_classifier_if_needed(metas, args.classifier_path, device)

    rows = []
    for meta in metas:
        for steps in args.steps:
            tag = f"{meta['task']} {variant_name(meta)} @ num_steps={steps}"
            print(f"\n=== {tag} ===")
            try:
                row = evaluate(meta, device, classifier,
                               num_samples=args.num_samples, num_steps=steps,
                               cfg_scale=args.cfg_scale, eval_seed=args.eval_seed)
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
