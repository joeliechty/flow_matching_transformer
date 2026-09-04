"""Sweep `cfg_scale` at inference time for every CFG-trained conditional model.

Produces the classic CFG quality/diversity tradeoff curve. Reuses the metric
functions from `utils.eval_utils` and the per-task evaluators from
`experiments.evaluate_all`. Output: experiments/results/cfg_sweep.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import (
    discover_checkpoints, evaluate_image, evaluate_pose,
)
from utils.mnist_classifier import load_classifier


DEFAULT_SWEEP = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--scales', type=float, nargs='+', default=list(DEFAULT_SWEEP))
    parser.add_argument('--output', type=str, default='experiments/results/cfg_sweep.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    classifier = None
    if Path(args.classifier_path).exists():
        classifier = load_classifier(args.classifier_path, device=device)
    else:
        print(f"WARNING: no classifier at {args.classifier_path} — MNIST CFG sweep will be skipped.")

    rows = []
    for meta in discover_checkpoints(Path(args.checkpoint_dir)):
        if meta['prefix'] != 'cond_' or meta['cfg'] != '_CFG':
            continue  # CFG sweep only meaningful for conditional CFG-trained models
        tag = f"cond_{meta['task']}{meta['ot']}{meta['cfg']}"
        for scale in args.scales:
            print(f"\n=== {tag} @ cfg_scale={scale} ===")
            try:
                if meta['task'] == 'pose':
                    row = evaluate_pose(meta, device,
                                        num_samples=args.num_samples,
                                        num_steps=args.num_steps,
                                        cfg_scale=scale)
                else:
                    if classifier is None:
                        continue
                    row = evaluate_image(meta, device, classifier,
                                         num_samples=args.num_samples,
                                         num_steps=args.num_steps,
                                         cfg_scale=scale)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            row['variant'] = tag
            rows.append(row)
            print({k: v for k, v in row.items() if v is not None})

    if not rows:
        print("No rows produced.")
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()
