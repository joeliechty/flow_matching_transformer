"""Figures and decision rules for the ACRONYM grasp-pose ablations (acronym_ablations_summary.md).

Reads what ./pose_ablations.sh writes for an ACRONYM task and writes PNGs to docs/acronym/:

    python experiments/acronym_figures.py --part ids               # Part 2 (discrete conditioning)
    python experiments/acronym_figures.py --part points            # Part 1 (continuous conditioning)
    python experiments/acronym_figures.py --part ids --only size   # one figure, or a rule:
    python experiments/acronym_figures.py --part ids --only rules   # prints the size and baseline rules

The primary metric is the 1-NN two-sample accuracy against held-out real grasps (0.5 = the
samples can't be told from real ones); recreation and fidelity error (cm) are the readable pair.
Both rules were set before the results:
  size      the smallest backbone whose score is within one seed SD of the best size's
  baseline  among the conditional no-CFG variants, the lowest score
where score = the seed-mean of (1-NNA at 3 Euler steps + 1-NNA at 100) / 2, on the part's
primary level (held-out grasps for ids, held-out objects for point clouds).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.summary_figures import (BLUE, INK2, MUTED, ORANGE, SURFACE, data_line,  # noqa: E402
                                         save, setup_matplotlib)

PARTS = {  # part -> (results folder, primary level, title)
    'ids': ('sota/pose_acronym_10cat_ids', 'heldout_grasps', 'ID conditioning'),
    'points': ('sota/pose_acronym_10cat_points', 'heldout_objects', 'point-cloud conditioning'),
}
LEVEL_TITLE = {'heldout_grasps': 'held-out grasps of training objects',
               'heldout_objects': 'held-out objects', 'heldout_categories': 'held-out categories'}
EPOCH = 100
STEPS = (1, 2, 3, 5, 9, 20, 50, 100)
SIZES = [('S', '256 wide, 6 blocks'), ('M', '384 wide, 8 blocks'), ('L', '512 wide, 12 blocks'),
         ('XL', '768 wide, 16 blocks')]
PRIMARY = 'one_nna'
GUIDANCE = 3.0  # the scale evaluate_all samples CFG-trained models at
METRICS = {  # metric -> axis label
    'one_nna': '1-NN accuracy vs. held-out grasps\n(0.5 = indistinguishable), lower is better',
    'recreation_err': 'recreation error (cm)\nreal grasp → nearest sample',
    'fidelity_err': 'fidelity error (cm)\nsample → nearest real grasp',
    'energy_distance': 'energy distance (cm)',
}
VARIANTS = {  # variant -> (label, colour, linestyle, marker); dashed = CFG-trained
    'cond_OT_NOCFG': ('per-object OT', BLUE, '-', 'o'),
    'cond_NOOT_NOCFG': ('random pairing', ORANGE, '-', 's'),
    'cond_GOT_NOCFG': ('global OT', '#1baf7a', '-', '^'),
    'cond_OT_CFG': ('per-object OT + CFG', BLUE, '--', 'o'),
    'cond_NOOT_CFG': ('random pairing + CFG', ORANGE, '--', 's'),
    'cond_GOT_CFG': ('global OT + CFG', '#1baf7a', '--', '^'),
    'uncond_OT': ('unconditional, OT', '#e87ba4', ':', 'D'),
    'uncond_NOOT': ('unconditional, random', '#008300', ':', 'v'),
}
SIZE_COLORS = ('#eda100', ORANGE, BLUE, '#1baf7a')
SHORT = {'cond_OT_NOCFG': 'per-object\nOT', 'cond_NOOT_NOCFG': 'random\npairing', 'cond_GOT_NOCFG': 'global\nOT',
         'cond_OT_CFG': 'per-object\nOT + CFG', 'cond_NOOT_CFG': 'random\n+ CFG', 'cond_GOT_CFG': 'global\nOT + CFG',
         'uncond_OT': 'uncond.\nOT', 'uncond_NOOT': 'uncond.\nrandom'}


def heading(part):
    title = PARTS[part][2]
    return title[0].upper() + title[1:]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_dir(part, *sub, level=None, seed=None):
    """experiments/results/<task>[/sub...]/epoch_<EPOCH>/<level>[/seed_<N>]."""
    task, primary, _ = PARTS[part]
    d = REPO_ROOT / 'experiments/results' / task
    for s in sub:
        d = d / s
    d = d / f'epoch_{EPOCH}' / (level or primary)
    return d / f'seed_{seed}' if seed is not None else d


def seeds_in(d):
    return sorted(int(p.name.split('_')[1]) for p in d.glob('seed_*') if (p / 'metrics.csv').exists())


def values(part, variant, metric, steps, *sub, level=None):
    """One value per seed: `metric` of `variant` at `steps` Euler steps (100 = the main eval)."""
    d = run_dir(part, *sub, level=level)
    out = []
    for s in seeds_in(d):
        rows = read(d / f'seed_{s}' / ('metrics.csv' if steps == 100 else 'steps_sweep.csv'))
        row = [r for r in rows if r['variant'] == variant and int(r['num_steps']) == steps
               and r.get('method', 'euler') in ('euler', '')]
        if len(row) != 1:
            raise ValueError(f'{d} {variant} {steps} steps: {len(row)} rows')
        out.append(float(row[0][metric]))
    return np.array(out)


def data_value(part, metric, *sub, level=None):
    d = run_dir(part, *sub, level=level)
    rows = read(d / f'seed_{seeds_in(d)[0]}' / 'metrics.csv')
    return float(next(r for r in rows if r['variant'] == 'data')[metric])


def score(part, variant, *sub):
    """The rules' score per seed: (1-NNA at 3 Euler steps + 1-NNA at 100) / 2."""
    return (values(part, variant, PRIMARY, 3, *sub) + values(part, variant, PRIMARY, 100, *sub)) / 2


def size_rule(part, variant='cond_OT_NOCFG'):
    """(chosen size, {size: (mean score, seed SD)}) over the sizes that have results."""
    scores = {}
    for size, _ in SIZES:
        if run_dir(part, 'size', size).exists():
            s = score(part, variant, 'size', size)
            scores[size] = (s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0)
    best = min(scores, key=lambda k: scores[k][0])
    for size, _ in SIZES:  # smallest first
        if size in scores and scores[size][0] <= scores[best][0] + scores[best][1]:
            return size, scores
    return best, scores


def baseline_rule(part):
    """(chosen variant, {variant: mean score}) among the conditional no-CFG variants."""
    rows = read(run_dir(part, seed=seeds_in(run_dir(part))[0]) / 'metrics.csv')
    candidates = sorted({r['variant'] for r in rows if r['variant'].startswith('cond_')
                         and r['variant'].endswith('_NOCFG')})
    scores = {v: float(score(part, v).mean()) for v in candidates}
    return min(scores, key=scores.get), scores


# -- figures ----------------------------------------------------------------------------------

def _steps_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(STEPS)
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.minorticks_off()
    ax.set_xlabel('Euler steps at sampling (log)')


def _band(ax, x, vals, color, label, ls='-', marker='o'):
    ax.fill_between(x, vals.min(0), vals.max(0), color=color, alpha=0.12, lw=0)
    ax.plot(x, vals.mean(0), color=color, ls=ls, marker=marker, mfc=color, mec=color, lw=1.8, ms=6,
            label=label)


def fig_size(plt, out, part, variant='cond_OT_NOCFG'):
    """Backbone size: the three pose-recreation metrics against sampling steps, per size."""
    chosen, scores = size_rule(part)
    metrics = ('one_nna', 'recreation_err', 'fidelity_err')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for (size, desc), color in zip(SIZES, SIZE_COLORS):
        if size not in scores:
            continue
        params = _params(part, 'size', size)
        label = f'{size}: {desc}, {params}' + ('  ← chosen' if size == chosen else '')
        for ax, metric in zip(axes, metrics):
            vals = np.stack([values(part, variant, metric, s, 'size', size) for s in STEPS], 1)
            _band(ax, STEPS, vals, color, label)
    for ax, metric in zip(axes, metrics):
        data_line(ax, data_value(part, metric, 'size', chosen), where='left')
        _steps_axis(ax)
        ax.set_ylabel(METRICS[metric])
    axes[0].set_ylim(0.48, None)
    axes[0].legend(fontsize=9.5, loc='upper right')
    level = PARTS[part][1]
    fig.suptitle(f'Backbone size, {PARTS[part][2]} ({VARIANTS[variant][0]}, no CFG), epoch {EPOCH}: '
                 f'{LEVEL_TITLE[level]}; mean over seeds, shading = seed range', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{part}_size.png')
    plt.close(fig)


def _params(part, *sub):
    """Parameter count of a run, from its training console ('Model parameters: N')."""
    task = PARTS[part][0]
    d = REPO_ROOT / 'checkpoints' / task
    for s in sub:
        d = d / s
    for log in sorted(d.glob('seed_*/*_log.txt')):
        m = re.search(r'Model parameters: ([\d,]+)', log.read_text())
        if m:
            return f'{int(m.group(1).replace(",", "")) / 1e6:.0f}M parameters'
    return '?'


def fig_overview(plt, out, part):
    """Every variant at 1, 3 and 100 Euler steps on the primary metric: one dot per seed."""
    rows = read(run_dir(part, seed=seeds_in(run_dir(part))[0]) / 'metrics.csv')
    variants = [v for v in VARIANTS if any(r['variant'] == v for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, steps in zip(axes, (1, 3, 100)):
        for i, v in enumerate(variants):
            label, color, ls, marker = VARIANTS[v]
            vals = values(part, v, PRIMARY, steps)
            ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=36, marker=marker,
                       facecolors=SURFACE if ls == '--' else color, edgecolors=color, linewidths=1.5, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=color, lw=2.5, zorder=4)
            ax.text(i, vals.max() + 0.01, f'{vals.mean():.3f}', ha='center', va='bottom', fontsize=9, color=INK2)
        data_line(ax, data_value(part, PRIMARY))
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([SHORT[v] for v in variants], fontsize=9)
        ax.grid(axis='x', visible=False)
        ax.set_title(f'{steps} Euler step{"s" if steps > 1 else ""}')
    axes[0].set_ylabel(METRICS[PRIMARY])
    fig.suptitle(f'{heading(part)}, epoch {EPOCH}: {LEVEL_TITLE[PARTS[part][1]]}; '
                 f'one dot per seed, bar = mean (CFG-trained models sampled at guidance {GUIDANCE:g})', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{part}_overview.png')
    plt.close(fig)


def fig_steps(plt, out, part):
    """The pose-recreation metrics against sampling steps, every variant."""
    rows = read(run_dir(part, seed=seeds_in(run_dir(part))[0]) / 'metrics.csv')
    variants = [v for v in VARIANTS if any(r['variant'] == v for r in rows)]
    metrics = ('one_nna', 'recreation_err', 'fidelity_err')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for v in variants:
        label, color, ls, marker = VARIANTS[v]
        for ax, metric in zip(axes, metrics):
            vals = np.stack([values(part, v, metric, s) for s in STEPS], 1)
            _band(ax, STEPS, vals, color, label, ls=ls, marker=marker)
    for ax, metric in zip(axes, metrics):
        data_line(ax, data_value(part, metric), where='left')
        _steps_axis(ax)
        ax.set_ylabel(METRICS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Sampling steps, {PARTS[part][2]}, epoch {EPOCH}: {LEVEL_TITLE[PARTS[part][1]]}; '
                 f'mean over seeds, shading = seed range (CFG-trained models at guidance {GUIDANCE:g})', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, f'{part}_steps_sweep.png')
    plt.close(fig)


LEVEL_COLORS = {'heldout_grasps': BLUE, 'heldout_objects': ORANGE, 'heldout_categories': '#1baf7a'}


def fig_levels(plt, out, part='points'):
    """Generalisation: the point-cloud baseline on each held-out level, against the id baseline
    on held-out grasps (ids can't condition on unseen objects)."""
    base = baseline_rule(part)[0]
    metrics = ('one_nna', 'recreation_err', 'fidelity_err')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for level, color in LEVEL_COLORS.items():
        for ax, metric in zip(axes, metrics):
            vals = np.stack([values(part, base, metric, st, level=level) for st in STEPS], 1)
            _band(ax, STEPS, vals, color, f'point clouds: {LEVEL_TITLE[level]}')
            ax.axhline(data_value(part, metric, level=level), color=color, ls=':', lw=1.2, alpha=0.8)
    if run_dir('ids').exists():
        ids_base = baseline_rule('ids')[0]
        for ax, metric in zip(axes, metrics):
            vals = np.stack([values('ids', ids_base, metric, st) for st in STEPS], 1)
            _band(ax, STEPS, vals, INK2, 'ids: held-out grasps of training objects', ls='--', marker='s')
    for ax, metric in zip(axes, metrics):
        _steps_axis(ax)
        ax.set_ylabel(METRICS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Generalisation, epoch {EPOCH}: the per-object OT baselines (points at M, ids at L) on each '
                 'held-out level; dotted = real data of that level; mean over seeds', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, 'generalisation.png')
    plt.close(fig)


ABLATION_AXES = [  # (title, [(arm, label)]); arm None = the baseline itself
    ('Condition pathway', [(None, 'cross-attention (baseline)'), ('cond_adaln', 'adaLN'),
                           ('cond_joint', 'joint attention')]),
    ('Rotation of the prior', [(None, 'uniform on SO(3) (baseline)'), ('rot_gauss', 'exp of N(0, 1) twist')]),
    ('Grasps per object per batch', [('k4', '4 (256 objects)'), (None, '16 (64 objects, baseline)'),
                                     ('k64', '64 (16 objects)')]),
    ('Training-time density', [(None, 'uniform (baseline)'), ('t_logit_normal', 'logit-normal (SD3)'),
                               ('t_beta', 'Beta, noisy end (π0)')]),
]
ARM_COLORS = (INK2, BLUE, '#eda100')
ARM_MARKERS = ('s', 'o', '^')
SOLVERS = [('euler', 'Euler', INK2, 's'), ('midpoint', 'midpoint', BLUE, 'o'), ('heun', 'Heun', '#eda100', '^')]
NFE = (2, 4, 6, 10, 18, 40, 100)
INTERVALS = [('full', 'whole path', INK2), ('0-0.5', 't < 0.5 (noisy half)', BLUE),
             ('0.25-0.75', '0.25 ≤ t < 0.75', '#1baf7a'), ('0.5-1', 't ≥ 0.5 (data half)', '#eda100')]
GUIDANCE_SCALES = (1.0, 1.5, 3.0)


def fig_ablations(plt, out, part):
    """Single-axis ablations of the baseline: the primary metric against sampling steps, one
    row per held-out level (point clouds: unseen objects, then seen objects' held-out grasps)."""
    base, _ = baseline_rule(part)
    axes_present = [(t, arms) for t, arms in ABLATION_AXES
                    if any(a and run_dir(part, 'ablate', a).exists() for a, _ in arms)]
    if not axes_present:
        raise FileNotFoundError('no ablation results')
    levels = [PARTS[part][1]] + (['heldout_grasps'] if part == 'points' else [])
    fig, axes = plt.subplots(len(levels), len(axes_present), figsize=(5.2 * len(axes_present), 4.8 * len(levels)),
                             sharey='row', squeeze=False)
    for r, level in enumerate(levels):
        for ax, (title, arms) in zip(axes[r], axes_present):
            for (arm, label), color, marker in zip(arms, ARM_COLORS, ARM_MARKERS):
                sub = () if arm is None else ('ablate', arm)
                if arm is not None and not run_dir(part, *sub, level=level).exists():
                    continue
                vals = np.stack([values(part, base, PRIMARY, s, *sub, level=level) for s in STEPS], 1)
                _band(ax, STEPS, vals, color, label, ls='--' if arm is None else '-', marker=marker)
            data_line(ax, data_value(part, PRIMARY, level=level), where='left')
            _steps_axis(ax)
            ax.set_title(f'{title}\n{LEVEL_TITLE[level]}', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
        axes[r, 0].set_ylabel(METRICS[PRIMARY])
    fig.suptitle(f'Single-axis ablations, {PARTS[part][2]}, epoch {EPOCH}: each arm changes one choice of '
                 f'the baseline ({VARIANTS[base][0]}); mean over seeds, shading = seed range', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{part}_ablations.png')
    plt.close(fig)


def _sweep_values(part, name, variant, metric, **match):
    d = run_dir(part)
    out = []
    for seed in seeds_in(d):
        rows = [r for r in read(d / f'seed_{seed}' / f'{name}.csv') if r['variant'] == variant
                and all(str(r[k]) == str(v) or (r[k].replace('.', '', 1).isdigit()
                                                and float(r[k]) == float(v)) for k, v in match.items())]
        if len(rows) != 1:
            raise ValueError(f'{name} {variant} {match}: {len(rows)} rows')
        out.append(float(rows[0][metric]))
    return np.array(out)


def fig_solvers(plt, out, part):
    """ODE solvers at equal network evaluations on the baseline (unguided)."""
    base, _ = baseline_rule(part)
    metrics = ('one_nna', 'recreation_err', 'fidelity_err')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for method, label, color, marker in SOLVERS:
        for ax, metric in zip(axes, metrics):
            vals = np.stack([_sweep_values(part, 'solver_sweep', base, metric, method=method, nfe=n)
                             for n in NFE], 1)
            _band(ax, NFE, vals, color, label, marker=marker)
    for ax, metric in zip(axes, metrics):
        data_line(ax, data_value(part, metric))
        ax.set_xscale('log'); ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE]); ax.minorticks_off()
        ax.set_xlabel('network evaluations per sample (log)')
        ax.set_ylabel(METRICS[metric])
    axes[0].legend(fontsize=10, loc='upper right')
    fig.suptitle(f'ODE solvers at equal cost, {PARTS[part][2]} baseline ({VARIANTS[base][0]}), epoch {EPOCH}; '
                 'mean over seeds, shading = seed range', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{part}_solvers.png')
    plt.close(fig)


def fig_guidance(plt, out, part, variant=None):
    """Guidance scale and interval on the CFG-trained baseline: recreation vs. fidelity."""
    if variant is None:
        variant = baseline_rule(part)[0].replace('_NOCFG', '_CFG')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for steps, ls in ((9, '--'), (100, '-')):
        for interval, label, color in INTERVALS:
            pts = {m: [] for m in ('one_nna', 'recreation_err', 'fidelity_err')}
            for g in GUIDANCE_SCALES:
                iv = 'full' if g == 1.0 else interval
                for m in pts:
                    pts[m].append(_sweep_values(part, 'guidance_sweep', variant, m, num_steps=steps,
                                                cfg_interval=iv, cfg_scale_at_inference=g).mean())
            axes[0].plot(GUIDANCE_SCALES, pts['one_nna'], color=color, ls=ls, marker='o', lw=1.8,
                         label=f'{label}, {steps} steps')
            axes[1].plot(GUIDANCE_SCALES, pts['recreation_err'], color=color, ls=ls, marker='o', lw=1.8)
            axes[2].plot(pts['fidelity_err'], pts['recreation_err'], color=color, ls=ls, marker='o', lw=1.8)
    data_line(axes[0], data_value(part, 'one_nna'))
    data_line(axes[1], data_value(part, 'recreation_err'))
    for ax in axes[:2]:
        ax.set_xticks(GUIDANCE_SCALES); ax.set_xlabel('guidance scale')
    axes[0].set_ylabel(METRICS['one_nna']); axes[1].set_ylabel(METRICS['recreation_err'])
    axes[2].set_xlabel(METRICS['fidelity_err'].replace('\n', ' ')); axes[2].set_ylabel(METRICS['recreation_err'])
    axes[2].set_title('trade-off: each line runs from guidance 1 to 3')
    axes[0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle(f'Guidance, {PARTS[part][2]} ({VARIANTS[variant][0]}), epoch {EPOCH}: mean over seeds', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{part}_guidance.png')
    plt.close(fig)


def fig_training(plt, out, part):
    """The primary metric over training, for the baseline, at three samplers."""
    base, _ = baseline_rule(part)
    root = REPO_ROOT / 'experiments/results' / PARTS[part][0] / 'training_curves'
    samplers = [('steps3', '3 Euler steps', '#eda100', '^'), ('midpoint5', '5 midpoint steps', BLUE, 'o'),
                ('steps100', '100 Euler steps', INK2, 's')]
    seeds = sorted(int(p.name.split('_')[1]) for p in root.glob('seed_*'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, label, color, marker in samplers:
        epochs = sorted({int(re.search(r'epoch(\d+)_', p.name).group(1))
                         for p in root.glob(f'seed_{seeds[0]}/metrics_epoch*_{name}.csv')})
        for ax, metric in zip(axes, ('one_nna', 'recreation_err')):
            vals = np.array([[float(next(r for r in read(root / f'seed_{s}/metrics_epoch{e}_{name}.csv')
                                         if r['variant'] == base)[metric]) for e in epochs] for s in seeds])
            _band(ax, epochs, vals, color, label, marker=marker)
    for ax, metric in zip(axes, ('one_nna', 'recreation_err')):
        data_line(ax, data_value(part, metric))
        ax.set_xlabel('training epoch (300 batches of 1024 each)')
        ax.set_ylabel(METRICS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Training length, {PARTS[part][2]} baseline ({VARIANTS[base][0]}), '
                 f'{LEVEL_TITLE[PARTS[part][1]]}; mean over seeds, shading = seed range', y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, f'{part}_training.png')
    plt.close(fig)


FIGURES = {'size': fig_size, 'overview': fig_overview, 'steps': fig_steps, 'levels': fig_levels,
           'ablations': fig_ablations,
           'solvers': fig_solvers, 'guidance': fig_guidance, 'training': fig_training}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--part', choices=tuple(PARTS), required=True)
    parser.add_argument('--only', choices=(*FIGURES, 'rules'), default=None)
    parser.add_argument('--out', type=str, default='docs/acronym')
    args = parser.parse_args()
    if args.only in (None, 'rules'):
        if (REPO_ROOT / 'experiments/results' / PARTS[args.part][0] / 'size').exists():
            chosen, scores = size_rule(args.part)
            print('size rule:', {k: f'{m:.4f} ± {s:.4f}' for k, (m, s) in scores.items()}, '->', chosen)
        if run_dir(args.part).exists():
            chosen, scores = baseline_rule(args.part)
            print('baseline rule:', {k: round(v, 4) for k, v in scores.items()}, '->', chosen)
        if args.only == 'rules':
            return
    plt = setup_matplotlib()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in FIGURES.items():
        if args.only in (None, name):
            if name == 'levels' and args.part != 'points':
                continue
            try:
                fig(plt, out, args.part)
            except (FileNotFoundError, ValueError, IndexError) as e:
                print(f'skip {name}: {e}')


if __name__ == '__main__':
    main()
