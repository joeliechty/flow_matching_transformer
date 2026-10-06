"""Sweep `cfg_scale` at inference time for every CFG-trained conditional model.

Produces the classic CFG quality/diversity tradeoff curve. Reuses the metric
functions from `utils.eval_utils` and the per-task evaluators from
`experiments.evaluate_all`. Output: experiments/results/cfg_sweep.csv
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


DEFAULT_SWEEP = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Evaluate the checkpoints saved at this epoch (default: latest)')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--scales', type=float, nargs='+', default=list(DEFAULT_SWEEP))
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='experiments/results/cfg_sweep.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # CFG sweep only meaningful for conditional CFG-trained models.
    metas = [m for m in discover_checkpoints(Path(args.checkpoint_dir), args.epoch)
             if m['prefix'] == 'cond_' and m['cfg'] == '_CFG']
    classifier = load_classifier_if_needed(metas, args.classifier_path, device)

    rows = []
    for meta in metas:
        for scale in args.scales:
            tag = f"{meta['task']} {variant_name(meta)} @ cfg_scale={scale}"
            print(f"\n=== {tag} ===")
            try:
                row = evaluate(meta, device, classifier,
                               num_samples=args.num_samples, num_steps=args.num_steps,
                               cfg_scale=scale, eval_seed=args.eval_seed)
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
