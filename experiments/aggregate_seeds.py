"""Combine per-seed results into mean ± std summaries and plots.

Reads <results_dir>/seed_*/{metrics,cfg_sweep,steps_sweep}.csv (written by
evaluate_all.py, cfg_sweep.py and steps_sweep.py) and writes into <results_dir>:

  metrics_summary.csv, cfg_sweep_summary.csv, steps_sweep_summary.csv
  cfg_sweep.png, steps_sweep.png        (mean line, ±1 std band across seeds)
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import write_csv


# metric -> (panel title, how to read it). Real goal samples sit at their own spread
# from the mode centre, so mode_distance below that means samples are collapsing.
METRICS = {
    'mode_accuracy': ('Mode accuracy', 'higher is better'),
    'mode_distance': ('Twist distance to target mode', "lower is better, down to the data's spread"),
    'mode_coverage_kl': ('Mode coverage KL', 'lower is better'),
    'class_accuracy': ('Class accuracy', 'higher is better'),
    'class_marginal_kl': ('Class marginal KL', 'lower is better'),
}

# variant -> (legend label, colour, linestyle). Colour follows the variant across every
# panel and figure; linestyle repeats the OT split (solid = OT, dashed = no OT) so that
# distinction never rests on hue alone.
VARIANT_STYLE = {
    'cond_OT_CFG':     ('cond · OT + CFG',      '#2a78d6', '-'),
    'cond_OT_NOCFG':   ('cond · OT, no CFG',    '#eb6834', '-'),
    'cond_NOOT_CFG':   ('cond · no OT + CFG',   '#1baf7a', '--'),
    'cond_NOOT_NOCFG': ('cond · no OT, no CFG', '#eda100', '--'),
    'uncond_OT':       ('uncond · OT',          '#e87ba4', '-'),
    'uncond_NOOT':     ('uncond · no OT',       '#008300', '--'),
}
_VARIANT_ORDER = list(VARIANT_STYLE)


def load_rows(results_dir: Path, filename: str):
    rows = []
    for seed_dir in sorted(results_dir.glob('seed_*')):
        path = seed_dir / filename
        if not path.exists():
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                r['seed'] = seed_dir.name.split('_', 1)[1]
                rows.append(r)
    return rows


def summarize(rows, keys):
    """Group rows by `keys` and reduce every metric to mean/std over seeds."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k, '') for k in keys), []).append(r)

    out = []
    for key, members in groups.items():
        s = dict(zip(keys, key))
        s['n_seeds'] = len({m['seed'] for m in members})
        for metric in METRICS:
            vals = [float(m[metric]) for m in members if m.get(metric)]
            if vals:
                s[f'{metric}_mean'] = statistics.fmean(vals)
                s[f'{metric}_std'] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(s)

    def sort_key(s):
        v = s.get('variant')
        rank = _VARIANT_ORDER.index(v) if v in _VARIANT_ORDER else len(_VARIANT_ORDER)
        sweep_x = (float(s[k]) for k in keys if k not in ('task', 'epoch', 'variant'))
        return (s.get('task', ''), float(s.get('epoch') or 0), rank, *sweep_x)
    return sorted(out, key=sort_key)


def summary_fieldnames(summary, keys):
    stats = [f'{m}_{stat}' for m in METRICS for stat in ('mean', 'std')
             if any(f'{m}_mean' in s for s in summary)]
    return list(keys) + ['n_seeds'] + stats


def plot_sweep(summary, x_key, xlabel, title, output, log_x=False):
    metrics = [m for m in METRICS if any(f'{m}_mean' in s for s in summary)]
    if not metrics:
        return
    n_seeds = max(s['n_seeds'] for s in summary)
    task = summary[0].get('task', '')
    epochs = sorted({s['epoch'] for s in summary if s.get('epoch')}, key=float)
    if epochs:
        task = f"{task}, epoch {'/'.join(epochs)}"

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.8), squeeze=False)
    handles = {}
    for ax, metric in zip(axes[0], metrics):
        for variant, (label, color, linestyle) in VARIANT_STYLE.items():
            pts = sorted((float(s[x_key]), s[f'{metric}_mean'], s[f'{metric}_std'])
                         for s in summary
                         if s.get('variant') == variant and f'{metric}_mean' in s)
            if not pts:
                continue
            xs, ys, sds = (np.array(v) for v in zip(*pts))
            ax.fill_between(xs, ys - sds, ys + sds, color=color, alpha=0.15, linewidth=0)
            (line,) = ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2,
                              marker='o', markersize=6, label=label)
            handles.setdefault(variant, line)

        name, hint = METRICS[metric]
        ax.set_title(f"{name}\n{hint}", fontsize=11)
        ax.set_xlabel(xlabel)
        if log_x:
            ax.set_xscale('log')
            ax.set_xticks(sorted({float(s[x_key]) for s in summary}))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.3, linewidth=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    ordered = [handles[v] for v in _VARIANT_ORDER if v in handles]
    fig.legend(ordered, [h.get_label() for h in ordered],
               loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"{task} — {title} (mean ± 1 std over {n_seeds} "
                 f"seed{'s' if n_seeds != 1 else ''})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def print_summary(summary):
    metrics = [m for m in METRICS if any(f'{m}_mean' in s for s in summary)]
    print(f"\n{'variant':<18}{'n':>3}  " + ''.join(f"{m:>22}" for m in metrics))
    for s in summary:
        cells = ''.join(
            f"{s[f'{m}_mean']:>13.4f} ± {s[f'{m}_std']:<6.4f}" if f'{m}_mean' in s else f"{'-':>22}"
            for m in metrics)
        print(f"{s['variant']:<18}{s['n_seeds']:>3}  {cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='experiments/results/pose/epoch_100/',
                        help='Directory holding seed_*/ result folders for one task and epoch')
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    metrics_rows = load_rows(results_dir, 'metrics.csv')
    if not metrics_rows:
        raise SystemExit(f"No seed_*/metrics.csv under {results_dir}")
    keys = ('task', 'epoch', 'variant')
    summary = summarize(metrics_rows, keys)
    write_csv(summary, results_dir / 'metrics_summary.csv', summary_fieldnames(summary, keys))
    print_summary(summary)

    for name, x_key, xlabel, title, log_x in (
        ('cfg_sweep', 'cfg_scale_at_inference', 'cfg_scale', 'CFG sweep', False),
        ('steps_sweep', 'num_steps', 'integration steps (log scale)', 'sampling-steps sweep', True),
    ):
        keys = ('task', 'epoch', 'variant', x_key)
        sweep = summarize(load_rows(results_dir, f'{name}.csv'), keys)
        if not sweep:
            continue
        write_csv(sweep, results_dir / f'{name}_summary.csv', summary_fieldnames(sweep, keys))
        plot_sweep(sweep, x_key, xlabel, title, results_dir / f'{name}.png', log_x=log_x)


if __name__ == '__main__':
    main()
