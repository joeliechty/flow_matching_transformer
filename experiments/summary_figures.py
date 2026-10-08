"""Figures for ablations_summary.md: the pose ablations on the two multimodal tasks.

Reads what ./pose_ablations.sh writes for each task (the eval stage at epoch 100, the curves
stage and the training logs), samples the epoch-100 models for the per-sample figures, and
writes the PNGs that ablations_summary.md embeds:

    python experiments/summary_figures.py                    # Part 1 (discrete) -> docs/ablations/
    python experiments/summary_figures.py --part continuous  # Part 2 (continuous conditions)
    python experiments/summary_figures.py --part sota        # Part 3 (current framework, 3σ task)

Parts 1 and 2 read pre-framework results; their per-sample figures load pre-framework checkpoints,
which need tag pose-continuous-v1.

Part 1: colour = OT (blue) or no OT (orange); dashed lines / open markers = CFG-trained models
sampled at guidance 3; grey = real data. Part 2: one colour and marker per pairing
(`PAIRING_STYLE`); dashed = baselines.
"""
import argparse
import ast
import csv
import functools
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import _eval_obs, _token_sigma  # noqa: E402
from experiments.evaluate_toys import NOISE_SEED, SEED_A, generate_euler  # noqa: E402
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel  # noqa: E402
from pose_gen_inference import (_generate, _load_run, _run_config, _sample_start_poses,  # noqa: E402
                                generate_from_start_poses)
from utils.eval_utils import _rotvec, goal_mode_poses_from_config, sample_goal_mixture  # noqa: E402
from utils.pose_task import ContinuousGoalTask, task_conditions  # noqa: E402
from utils.tf_utils import _quat_to_rot_mat, convert_twist_to_pose  # noqa: E402
from utils.toy_tasks import joint, sample_source, sample_targets  # noqa: E402

TASKS = [  # (checkpoint / results folder, panel title)
    ('pose_corners_two_orientations', 'Two orientations per corner (5σ apart)'),
    ('pose_corners_3sigma', 'Every mode within 3σ'),
]
EPOCH = 100
SEEDS = (1, 2, 3, 4, 5)
GUIDANCE = 3.0
STEPS = (1, 2, 3, 5, 9, 20, 50, 100)
GUIDANCE_SWEEP = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)

BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#8a8a85'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

COND = ['cond_OT_NOCFG', 'cond_NOOT_NOCFG', 'cond_OT_CFG', 'cond_NOOT_CFG']
UNCOND = ['uncond_OT', 'uncond_NOOT']
STEM = {
    'cond_OT_NOCFG': 'cond_pose_flow_matching_model_OT_NOCFG',
    'cond_NOOT_NOCFG': 'cond_pose_flow_matching_model_NOOT_NOCFG',
    'cond_OT_CFG': 'cond_pose_flow_matching_model_OT_CFG',
    'cond_NOOT_CFG': 'cond_pose_flow_matching_model_NOOT_CFG',
}
LABEL = {
    'cond_OT_NOCFG': 'OT, no CFG',
    'cond_NOOT_NOCFG': 'no OT, no CFG',
    'cond_OT_CFG': f'OT + CFG (guidance {GUIDANCE:g})',
    'cond_NOOT_CFG': f'no OT + CFG (guidance {GUIDANCE:g})',
    'uncond_OT': 'unconditional, OT',
    'uncond_NOOT': 'unconditional, no OT',
}


def style(variant):
    ot = '_OT' in variant and '_NOOT' not in variant
    color = BLUE if ot else ORANGE
    guided = variant.endswith('_CFG') and variant.startswith('cond')
    return dict(color=color, ls='--' if guided else '-', marker='o' if ot else 's',
                mfc=SURFACE if guided else color, mec=color, lw=1.8, ms=6)


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'figure.titlesize': 14,
        'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'legend.frameon': False,
    })
    return plt


# -- result files -------------------------------------------------------------------------

def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def results(task, name):
    """{seed: rows of experiments/results/<task>/epoch_<EPOCH>/seed_<N>/<name>.csv}"""
    return {s: read(REPO_ROOT / f'experiments/results/{task}/epoch_{EPOCH}/seed_{s}/{name}.csv') for s in SEEDS}


def _matches(cell, value):
    """A CSV cell equals `value`: numerically for numbers, else as text."""
    return float(cell) == float(value) if not isinstance(value, str) else cell == value


def per_seed(rows_by_seed, variant, metric, **match):
    """One value per seed for `variant`, from the row matching `match` (e.g. num_steps=3)."""
    out = []
    for rows in rows_by_seed.values():
        row = [r for r in rows if r['variant'] == variant
               and all(_matches(r[k], v) for k, v in match.items())]
        if len(row) != 1:
            raise ValueError(f'{variant} {match}: {len(row)} rows')
        out.append(float(row[0][metric]))
    return np.array(out)


def data_value(task, metric):
    """The real-data reference row (identical for every seed)."""
    rows = results(task, 'metrics')[SEEDS[0]]
    return float(next(r for r in rows if r['variant'] == 'data')[metric])


def curve(task, variant, metric, steps):
    """[seeds, epochs] of `metric` from the curves stage; returns (epochs, values)."""
    root = REPO_ROOT / f'experiments/results/{task}/training_curves'
    epochs = sorted({int(m.group(1)) for p in root.glob(f'seed_{SEEDS[0]}/metrics_epoch*_steps{steps}.csv')
                     for m in [re.search(r'epoch(\d+)_', p.name)]})
    vals = np.array([[float(next(r for r in read(root / f'seed_{s}/metrics_epoch{e}_steps{steps}.csv')
                                 if r['variant'] == variant)[metric]) for e in epochs] for s in SEEDS])
    return np.array(epochs), vals


def training_loss(task, variant):
    """[seeds, epochs] training loss from each run's log."""
    losses = []
    for s in SEEDS:
        log = REPO_ROOT / f'checkpoints/{task}/seed_{s}/{STEM[variant]}_log.txt'
        losses.append([float(v) for v in re.findall(r'^Epoch \d+: ([0-9.eE+-]+)$', log.read_text(), re.M)])
    return np.arange(1, len(losses[0]) + 1), np.array(losses)


# -- plotting helpers -----------------------------------------------------------------------

def band_line(ax, x, vals, variant, label=None):
    """Mean over seeds with the seed range shaded."""
    st = style(variant)
    ax.fill_between(x, vals.min(0), vals.max(0), color=st['color'], alpha=0.12, lw=0)
    ax.plot(x, vals.mean(0), color=st['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'],
            mfc=st['mfc'], mec=st['mec'], ms=st['ms'], label=label or LABEL[variant])


def data_line(ax, y, text='real data', where='right'):
    ax.axhline(y, color=MUTED, ls=':', lw=1.5, zorder=1)
    x = 0.99 if where == 'right' else 0.01
    ax.text(x, y, f' {text} ', transform=ax.get_yaxis_transform(), ha='right' if where == 'right' else 'left',
            va='bottom', color=INK2, fontsize=9.5)


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')


# -- figures from the result files ----------------------------------------------------------

def fig_overview(plt, out):
    """Energy distance per configuration at 3 and 100 steps: one dot per seed."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey='row')
    for r, (task, title) in enumerate(TASKS):
        steps_rows, main_rows = results(task, 'steps_sweep'), results(task, 'metrics')
        for c, steps in enumerate((3, 100)):
            ax = axes[r, c]
            for i, v in enumerate(COND):
                vals = (per_seed(steps_rows, v, 'energy_distance', num_steps=3) if steps == 3
                        else per_seed(main_rows, v, 'energy_distance'))
                st = style(v)
                ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=42, marker=st['marker'],
                           facecolors=st['mfc'], edgecolors=st['color'], linewidths=1.5, zorder=3)
                ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=st['color'], lw=2.5, zorder=4)
                ax.text(i + 0.28, vals.mean(), f'{vals.mean():.3f}', va='center', fontsize=9.5, color=INK2)
            data_line(ax, data_value(task, 'energy_distance'), where='left')
            ax.set_yscale('log')
            ax.set_xticks(range(len(COND)))
            ax.set_xticklabels(['OT\nno CFG', 'no OT\nno CFG', f'OT + CFG\nguidance {GUIDANCE:g}',
                                f'no OT + CFG\nguidance {GUIDANCE:g}'])
            ax.set_xlim(-0.5, len(COND) - 0.2)
            ax.grid(axis='x', visible=False)
            ax.set_title(f'{title} · {steps} sampling steps')
            if c == 0:
                ax.set_ylabel('energy distance to real data (log)\nlower is better')
    fig.suptitle(f'Conditional models at epoch {EPOCH}: one dot per seed, bar = mean', y=1.0)
    fig.tight_layout()
    save(fig, out, 'overview.png')
    plt.close(fig)


def fig_steps(plt, out):
    """Energy distance vs. number of Euler steps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for c, (task, title) in enumerate(TASKS):
        rows = results(task, 'steps_sweep')
        for r, variants in enumerate((COND, UNCOND)):
            ax = axes[r, c]
            for v in variants:
                band_line(ax, STEPS, np.stack([per_seed(rows, v, 'energy_distance', num_steps=s) for s in STEPS], 1), v)
            data_line(ax, data_value(task, 'energy_distance'))
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS]); ax.minorticks_off()
            ax.set_title(f'{title} · {"conditional" if r == 0 else "unconditional"}')
            if c == 0:
                ax.set_ylabel('energy distance (log), lower is better')
            if r == 1:
                ax.set_xlabel('Euler steps at sampling (log)')
            if c == 1:
                ax.legend(loc='lower left', fontsize=9.5)
    fig.suptitle(f'Sampling steps, epoch {EPOCH}: mean over 5 seeds, shading = seed range', y=1.0)
    fig.tight_layout()
    save(fig, out, 'steps_sweep.png')
    plt.close(fig)


def fig_guidance(plt, out):
    """Guidance scale: distribution quality, corner separation and each condition's orientation split."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True)
    for r, (task, title) in enumerate(TASKS):
        rows, main_rows = results(task, 'cfg_sweep'), results(task, 'metrics')
        g = np.array(GUIDANCE_SWEEP)
        for c, (metric, ylabel) in enumerate((('energy_distance', 'energy distance (log), lower is better'),
                                               ('mode_accuracy', 'corner accuracy'))):
            ax = axes[r, c]
            for v in ('cond_OT_CFG', 'cond_NOOT_CFG'):
                label = LABEL[v].split(' (')[0] + ', guidance on x'
                band_line(ax, g, np.stack([per_seed(rows, v, metric, cfg_scale_at_inference=x) for x in g], 1),
                          v, label=label)
            for v, dx in (('cond_OT_NOCFG', -0.12), ('cond_NOOT_NOCFG', 0.12)):  # trained without CFG
                st = style(v)
                vals = per_seed(main_rows, v, metric)
                ax.errorbar(1 + dx, vals.mean(), yerr=[[vals.mean() - vals.min()], [vals.max() - vals.mean()]],
                            color=st['color'], marker=st['marker'], ms=7, lw=1.5, capsize=3, ls='none',
                            label=f'{LABEL[v]} (trained without CFG)', zorder=4)
            data_line(ax, data_value(task, metric))
            if metric == 'energy_distance':
                ax.set_yscale('log')
            else:
                ax.set_ylim(0.65, 1.02)
            ax.set_ylabel(ylabel)
            ax.set_title(title + (' (saturated: corners 10 units apart)'
                                  if metric == 'mode_accuracy' and data_value(task, metric) == 1.0 else ''))
        # each condition's split between its two orientations
        ax = axes[r, 2]
        conditions, _ = task_conditions(OmegaConf.to_container(
            _run_config(REPO_ROOT / f'checkpoints/{task}/seed_1', STEM['cond_OT_CFG']).training.action_dist_params,
            resolve=True))
        n = None
        for v, dx in (('cond_OT_CFG', -0.1), ('cond_NOOT_CFG', 0.1)):
            st = style(v)
            for x in g:
                shares = []
                for seed_rows in rows.values():
                    row = next(rr for rr in seed_rows if rr['variant'] == v and float(rr['cfg_scale_at_inference']) == x)
                    counts = ast.literal_eval(row['per_mode_counts'])
                    for cond in conditions:
                        a, b = (counts.get(k, 0) for k in cond.valid_modes)
                        shares.append(a / (a + b))
                        n = a + b
                ax.scatter(x + dx + np.linspace(-0.05, 0.05, len(shares)), shares, s=16, marker=st['marker'],
                           facecolors=st['mfc'], edgecolors=st['color'], linewidths=1.0, zorder=3)
        noise = 2 * np.sqrt(0.25 / n)
        ax.axhspan(0.5 - noise, 0.5 + noise, color=MUTED, alpha=0.15, lw=0, zorder=1)
        ax.axhline(0.5, color=MUTED, ls=':', lw=1.5, zorder=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel('share of a condition\'s samples\nin its first orientation')
        ax.set_title(f'{title}: orientation split\n(dot = condition × seed; grey = 95% range for real data, '
                     f'{n} samples)', fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel('guidance scale at sampling')
    for ax in axes.flat:
        ax.set_xticks(GUIDANCE_SWEEP); ax.set_xticklabels([f'{x:g}' for x in GUIDANCE_SWEEP])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'Guidance scale, CFG-trained models, 100 steps, epoch {EPOCH}', y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'guidance_sweep.png')
    plt.close(fig)


def fig_training(plt, out):
    """Training loss and evaluation metrics against training epochs."""
    panels = [('loss', None, 'training loss (log)'),
              ('energy_distance', 100, 'energy distance, 100 steps (log)'),
              ('energy_distance', 3, 'energy distance, 3 steps (log)'),
              ('spread_ratio_trans', 100, 'position spread ÷ real data, 100 steps')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2), sharex=True)
    for r, (task, title) in enumerate(TASKS):
        for c, (metric, steps, ylabel) in enumerate(panels):
            ax = axes[r, c]
            for v in COND:
                if metric == 'loss':
                    x, vals = training_loss(task, v)
                    st = style(v)
                    ax.fill_between(x, vals.min(0), vals.max(0), color=st['color'], alpha=0.12, lw=0)
                    ax.plot(x, vals.mean(0), color=st['color'], ls=st['ls'], lw=1.6, label=LABEL[v])
                else:
                    x, vals = curve(task, v, metric, steps)
                    band_line(ax, x, vals, v)
            if metric == 'energy_distance':
                data_line(ax, data_value(task, metric))
            if metric == 'spread_ratio_trans':
                ax.axhline(data_value(task, 'spread_ratio_trans'), color=MUTED, ls=':', lw=1.5)
                ax.text(0.99, data_value(task, 'spread_ratio_trans'), ' real data ', transform=ax.get_yaxis_transform(),
                        ha='right', va='bottom', color=INK2, fontsize=9.5)
            else:
                ax.set_yscale('log')
            ax.set_ylabel(ylabel)
            ax.set_title(title if c == 0 else '')
            if r == 1:
                ax.set_xlabel('training epoch')
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Training: mean over 5 seeds, shading = seed range (CFG models evaluated at guidance '
                 f'{GUIDANCE:g}; metrics every 10 epochs)', y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'training_curves.png')
    plt.close(fig)


# -- figures from fresh samples -------------------------------------------------------------

class Task:
    """A task's goal modes and conditions, read from a run's training config."""

    def __init__(self, task, stem=STEM['cond_OT_NOCFG']):
        cfg = _run_config(REPO_ROOT / f'checkpoints/{task}/seed_1', stem).training
        self.name = task
        self.goal_dist = OmegaConf.to_container(cfg.goal_dist_params, resolve=True)
        self.start_dist = OmegaConf.to_container(cfg.start_dist_params, resolve=True)
        self.conditions, _ = task_conditions(OmegaConf.to_container(cfg.action_dist_params, resolve=True))
        self.modes = goal_mode_poses_from_config(self.goal_dist)
        self.centre = self.modes[:, :3].mean(0)

    def mid_rotation(self, cond):
        q = self.modes[cond.valid_modes, 3:].clone()
        q = torch.where((q * q[:1]).sum(-1, keepdim=True) < 0, -q, q)
        return _quat_to_rot_mat((q.mean(0) / q.mean(0).norm())[None])[0]

    def yaw_offset(self, poses, cond):
        """Rotation about z relative to the midpoint of the condition's two orientations (rad)."""
        return _rotvec(self.mid_rotation(cond).T @ _quat_to_rot_mat(poses[:, 3:]))[:, 2]

    def outward_offset(self, poses, cond):
        """Position along the corner's outward direction (away from the other corners), relative to the corner."""
        corner = self.modes[cond.valid_modes[0], :3]
        u = (corner - self.centre) / (corner - self.centre).norm()
        return (poses[:, :3] - corner) @ u

    def target_yaws(self, cond):
        return self.yaw_offset(self.modes[cond.valid_modes], cond)

    def real(self, cond, n, seed):
        return sample_goal_mixture(self.goal_dist, n, seed=seed, modes=cond.valid_modes)[0]


@functools.lru_cache(maxsize=None)
def load(task_name, variant, seed, device):
    return _load_run(str(REPO_ROOT / f'checkpoints/{task_name}/seed_{seed}'), STEM[variant], EPOCH, device)[0]


def sample(task, variant, seed, steps, guidance, n, device):
    """n samples per condition from one seed's epoch-<EPOCH> model; the same starts for every model."""
    model = load(task.name, variant, seed, device)
    gen = torch.Generator().manual_seed(1000 + seed)
    out = []
    for cond in task.conditions:
        starts = _sample_start_poses(task.start_dist, n, gen)
        out.append(_generate(model, STEM[variant], starts, cond, steps, guidance, device, trajectory=False))
    return out


def violins(ax, groups, positions, colors):
    parts = ax.violinplot(groups, positions=positions, widths=0.8, showextrema=False, showmedians=False)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.35); body.set_linewidth(1.2)
    for x, g, color in zip(positions, groups, colors):
        q1, med, q3 = np.percentile(g, [25, 50, 75])
        ax.plot([x, x], [q1, q3], color=color, lw=3, solid_capstyle='butt', zorder=3)
        ax.scatter([x], [med], color=SURFACE, edgecolors=color, s=22, zorder=4)


def fig_orientation_violins(plt, out, device, n=128):
    """Per-sample rotation about z relative to the midpoint of each condition's two orientations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, (task_name, title) in zip(axes, TASKS):
        task = Task(task_name)
        groups, colors, labels = [], [], []
        real = torch.cat([task.yaw_offset(task.real(c, n * len(SEEDS), seed=7), c) for c in task.conditions])
        groups.append(real.numpy()); colors.append(MUTED); labels.append('real data')
        for steps in (1, 3, 100):
            for v in ('cond_OT_NOCFG', 'cond_NOOT_NOCFG'):
                yaws = [task.yaw_offset(p, c) for s in SEEDS
                        for p, c in zip(sample(task, v, s, steps, 1.0, n, device), task.conditions)]
                groups.append(torch.cat(yaws).numpy()); colors.append(style(v)['color'])
                labels.append('OT' if '_OT' in v else 'no OT')
        positions = [0] + [1.3 + i + i // 2 * 0.6 for i in range(len(groups) - 1)]
        violins(ax, groups, positions, colors)
        for k, steps in enumerate((1, 3, 100)):
            ax.text((positions[1 + 2 * k] + positions[2 + 2 * k]) / 2, -0.16, f'{steps} step{"s" if steps > 1 else ""}',
                    transform=ax.get_xaxis_transform(), ha='center', va='top', color=INK, fontsize=11)
        for y in task.target_yaws(task.conditions[0]).tolist():
            ax.axhline(y, color=MUTED, ls=':', lw=1.3, zorder=1)
        ax.text(0.995, task.target_yaws(task.conditions[0]).max().item(), ' target orientations ', ha='right',
                va='bottom', transform=ax.get_yaxis_transform(), color=INK2, fontsize=9.5)
        ax.set_xticks(positions); ax.set_xticklabels(labels)
        ax.grid(axis='x', visible=False)
        lim = 3.2 * task.target_yaws(task.conditions[0]).abs().max().item()
        ax.set_ylim(-lim, lim)
        ax.set_ylabel('rotation about z relative to the\nmidpoint of the two orientations (rad)')
        ax.set_title(title)
    fig.suptitle(f'Do samples keep both orientations? Models trained without CFG, epoch {EPOCH} '
                 f'(bar = middle 50%, dot = median)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'orientation_violins.png')
    plt.close(fig)


def fig_guidance_violins(plt, out, device, n=128):
    """Per-sample position along each corner's outward direction, by guidance scale."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, (task_name, title) in zip(axes, TASKS):
        task = Task(task_name)
        groups = [torch.cat([task.outward_offset(task.real(c, n * len(SEEDS), seed=7), c)
                             for c in task.conditions]).numpy()]
        for g in GUIDANCE_SWEEP:
            groups.append(torch.cat([task.outward_offset(p, c) for s in SEEDS
                                     for p, c in zip(sample(task, 'cond_OT_CFG', s, 100, g, n, device),
                                                     task.conditions)]).numpy())
        positions = list(range(len(groups)))
        violins(ax, groups, positions, [MUTED] + [BLUE] * len(GUIDANCE_SWEEP))
        print(f'  {task_name}: median outward offset, real data then guidance {GUIDANCE_SWEEP}: '
              + ', '.join(f'{np.median(g):+.3f}' for g in groups))
        top = 1.15 * np.percentile(np.concatenate(groups), 99)
        if np.concatenate(groups).max() < 2 * top:  # clip only tails long enough to flatten everything else
            top = None
        for x, g in zip(positions, groups if top is not None else []):
            above = int((g > top).sum())
            if above:
                ax.text(x, top, f'{above} of {len(g)}\nabove', ha='center', va='top', fontsize=9, color=INK2)
        ax.set_ylim(None, top)
        ax.axhline(0, color=MUTED, ls=':', lw=1.3, zorder=1)
        ax.text(0.995, 0, ' the corner ', ha='right', va='bottom', transform=ax.get_yaxis_transform(),
                color=INK2, fontsize=9.5)
        inward = -(task.modes[task.conditions[0].valid_modes[0], :3] - task.centre).norm().item()
        if inward > -1:  # the other corners are close enough to show
            ax.axhline(inward, color=MUTED, ls='--', lw=1.3, zorder=1)
            ax.text(0.995, inward, ' centre of the 4 corners ', ha='right', va='bottom',
                    transform=ax.get_yaxis_transform(), color=INK2, fontsize=9.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(['real\ndata'] + [f'guidance\n{g:g}' for g in GUIDANCE_SWEEP])
        ax.grid(axis='x', visible=False)
        ax.set_ylabel('position along the corner\'s outward direction,\nrelative to the corner')
        ax.set_title(title)
    fig.suptitle(f'Where guidance puts samples: OT + CFG models, 100 steps, epoch {EPOCH} '
                 f'(bar = middle 50%, dot = median)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'guidance_violins.png')
    plt.close(fig)


def copy_3d(out):
    """The 3-D sample figures from pose_gen_inference.py --visualize_task, seed 1."""
    for task, src, dst in (('pose_corners_3sigma', 'mappings_3d.png', 'mappings_3d_3sigma.png'),
                           ('pose_corners_two_orientations', 'rotation_paths_3d.png',
                            'rotation_paths_3d_two_orientations.png')):
        path = REPO_ROOT / f'experiments/results/{task}/epoch_{EPOCH}/{src}'
        shutil.copy(path, out / dst)
        print(f'Copied {path} -> {out / dst}')


# -- Part 2: continuous conditions ----------------------------------------------------------

CONT_TASKS = [  # (checkpoint / results folder, panel title)
    ('pose_corners_two_orientations_jitter', 'Noisy tokens: corners, two orientations each'),
    ('pose_continuous_goals', 'Continuous goals: disk, two orientations each'),
]
TOYS = [('moons', '8 Gaussians → moons, conditioned on x'), ('fork', 'Fork: y given x')]
# variant -> (label, colour, linestyle, marker). C²OT takes OT's blue and random pairing no
# OT's orange, as in Part 1; the rest follow the palette order. Dashed / dotted = baselines.
PAIRING_STYLE = {
    'cond_NOOT_NOCFG': ('random pairing (I-CFM)', ORANGE, '--', 's'),
    'cond_GOT_NOCFG': ('global OT (ignores conditions)', '#e87ba4', '--', 'v'),
    'cond_OT_NOCFG': ('per-corner OT (oracle ids)', '#008300', ':', 'P'),
    'cond_C2OT_NOCFG': ('C²OT (adaptive weight)', BLUE, '-', 'o'),
    'cond_C2OTFIX_NOCFG': ('fixed weight (COT-FM)', '#eda100', '-', '^'),
    'cond_CLUSTER_NOCFG': ('cluster (COT Policy)', '#1baf7a', '-', 'D'),
}
SHORT = {'cond_NOOT_NOCFG': 'random\n(I-CFM)', 'cond_GOT_NOCFG': 'global\nOT',
         'cond_OT_NOCFG': 'per-corner\nOT (oracle)', 'cond_C2OT_NOCFG': 'C²OT',
         'cond_C2OTFIX_NOCFG': 'fixed\nweight', 'cond_CLUSTER_NOCFG': 'cluster'}
TOY_SUFFIX = {'cond_NOOT_NOCFG': 'NOOT', 'cond_GOT_NOCFG': 'GOT', 'cond_C2OT_NOCFG': 'C2OT',
              'cond_C2OTFIX_NOCFG': 'C2OTFIX', 'cond_CLUSTER_NOCFG': 'CLUSTER'}
METHOD_VARIANT = {'independent': 'cond_NOOT_NOCFG', 'global': 'cond_GOT_NOCFG', 'ot': 'cond_OT_NOCFG',
                  'c2ot': 'cond_C2OT_NOCFG', 'c2ot_fixed': 'cond_C2OTFIX_NOCFG',
                  'cluster': 'cond_CLUSTER_NOCFG'}


def pstyle(variant):
    label, color, ls, marker = PAIRING_STYLE[variant]
    return dict(label=label, color=color, ls=ls, marker=marker)


def present(task):
    """The pairing variants trained for a task, in PAIRING_STYLE order."""
    rows = results(task, 'metrics')[SEEDS[0]]
    return [v for v in PAIRING_STYLE if any(r['variant'] == v for r in rows)]


def cont_band_line(ax, x, vals, variant):
    st = pstyle(variant)
    ax.fill_between(x, vals.min(0), vals.max(0), color=st['color'], alpha=0.12, lw=0)
    ax.plot(x, vals.mean(0), color=st['color'], ls=st['ls'], lw=1.8, marker=st['marker'], ms=6,
            label=st['label'])


def fig_calibration(plt, out):
    """Branch agreement against prior skew as each pairing's knob loosens (no training)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (task, title) in zip(axes, CONT_TASKS):
        root = REPO_ROOT / f'experiments/results/{task}'
        calib = read(root / 'pairing_calibration.csv')
        diag = [r for r in read(root / 'pairing_diagnostics.csv')
                if r['ot_batch'] == calib[0]['ot_batch'] and r['method'] in ('independent', 'global', 'ot')]
        skew = lambda r: max(float(r['prior_skew_r2']), 0.0)
        chosen_text = []
        for method in ('c2ot', 'c2ot_fixed', 'cluster'):
            rows = [r for r in calib if r['method'] == method]
            st = pstyle(METHOD_VARIANT[method])
            ax.plot([skew(r) for r in rows], [float(r['branch_agreement']) for r in rows], color=st['color'],
                    ls='-', lw=1.6, marker=st['marker'], ms=5, label=st['label'] + ', knob loosening →')
            chosen = next(r for r in rows if r['chosen'] == 'True')
            knob = 'r_tar' if method == 'c2ot' else 'cond_scale'
            ax.scatter([skew(chosen)], [float(chosen['branch_agreement'])], s=150, facecolors='none',
                       edgecolors=st['color'], linewidths=2, zorder=5)
            chosen_text.append(f"{st['label'].split(' (')[0]}: {knob} {float(chosen[knob]):g}")
        for r in diag:
            st = pstyle(METHOD_VARIANT[r['method']])
            ax.scatter([skew(r)], [float(r['branch_agreement'])], s=70, marker=st['marker'], color=st['color'],
                       zorder=4, label=st['label'])
        ax.text(0.98, 0.04, 'chosen (rings):\n' + '\n'.join(chosen_text), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9.5, color=INK2)
        ax.axvline(0.02, color=MUTED, ls=':', lw=1.5)
        ax.text(0.02, 0.02, ' skew bound ', transform=ax.get_xaxis_transform(), color=INK2, fontsize=9.5)
        ax.set_xscale('symlog', linthresh=0.01, linscale=0.6)
        ax.set_xlim(-0.0005, 1.2)
        ax.set_xlabel('prior skew: R² predicting the condition from its paired noise\n(0 = every condition sees all of the noise)')
        ax.set_title(f'{title}\nOT batch {calib[0]["ot_batch"]}')
    axes[0].set_ylabel('branch agreement: share of noise sent to\nthe orientation it points at (0.5 = random)')
    handles, labels = axes[0].get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(*axes[1].get_legend_handles_labels()) if l not in labels]
    handles += [h for h, _ in extra]; labels += [l for _, l in extra]
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Calibrating the pairings without training: each line loosens one knob; '
                 'ring = the loosest setting within the skew bound', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, 'continuous_calibration.png')
    plt.close(fig)


@torch.no_grad()
def fig_toys(plt, out, device, n=2000):
    """One-step samples of each pairing on the two toys (seed 1), against real data."""
    variants = list(TOY_SUFFIX)
    fig, axes = plt.subplots(2, len(variants) + 1, figsize=(18, 6.4), sharex='row', sharey='row')
    for row, (toy, title) in enumerate(TOYS):
        summary = read(REPO_ROOT / f'experiments/results/toy_{toy}/toy_metrics_summary.csv')
        w2 = lambda suf: next(float(r['w2_mean']) for r in summary if r['variant'] == suf
                              and r['solver'] == 'euler' and r['num_steps'] == '1')
        target, cond = sample_targets(toy, n, torch.Generator().manual_seed(SEED_A))
        x0 = sample_source(toy, n, torch.Generator().manual_seed(NOISE_SEED)).to(device)
        real = joint(toy, target, cond)
        ax = axes[row, 0]
        ax.scatter(real[:, 0], real[:, 1], s=3, color=MUTED, alpha=0.6, lw=0)
        ax.set_title('real data' if row == 0 else '', fontsize=11)
        ax.set_ylabel(title)
        for col, v in enumerate(variants, start=1):
            path = REPO_ROOT / f'checkpoints/toy_{toy}/seed_1/toy_{toy}_{TOY_SUFFIX[v]}_final.pt'
            model, _ = ConditionalFlowMatchingTransformerModel.load_checkpoint(str(path), device=device)
            samples = generate_euler(model.eval(), x0, cond.to(device).unsqueeze(1), 1).cpu()
            pts = joint(toy, samples, cond)
            ax = axes[row, col]
            ax.scatter(real[:, 0], real[:, 1], s=3, color=GRID, lw=0)
            ax.scatter(pts[:, 0], pts[:, 1], s=3, color=pstyle(v)['color'], alpha=0.6, lw=0)
            label = pstyle(v)['label'].split(' (')[0]
            ax.set_title(f'{label}\nW₂² {w2(TOY_SUFFIX[v]):.3g} (3 seeds)', fontsize=10.5)
    for ax in axes.flat:
        ax.grid(False)
    axes[1, 0].set_xlabel('condition x'); axes[0, 0].set_xlabel('')
    fig.suptitle('Toys: samples after ONE Euler step (seed 1), real data in grey; W₂² = mean over 3 seeds, '
                 'lower is better', y=1.0)
    fig.tight_layout()
    save(fig, out, 'toy_samples.png')
    plt.close(fig)


def fig_continuous_overview(plt, out):
    """Energy distance per pairing at 1, 3 and 100 steps: one dot per seed."""
    steps_list = (1, 3, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharey='row')
    for r, (task, title) in enumerate(CONT_TASKS):
        steps_rows, main_rows = results(task, 'steps_sweep'), results(task, 'metrics')
        variants = present(task)
        for c, steps in enumerate(steps_list):
            ax = axes[r, c]
            for i, v in enumerate(variants):
                vals = (per_seed(main_rows, v, 'energy_distance') if steps == 100
                        else per_seed(steps_rows, v, 'energy_distance', num_steps=steps))
                st = pstyle(v)
                ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=36, marker=st['marker'],
                           color=st['color'], zorder=3)
                ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=st['color'], lw=2.5, zorder=4)
                ax.text(i, vals.max() * 1.35, f'{vals.mean():.3g}', ha='center', va='bottom', fontsize=9,
                        color=INK2)
            data_line(ax, data_value(task, 'energy_distance'), where='left')
            ax.set_yscale('log')
            ax.set_ylim(None, ax.get_ylim()[1] * 2.5)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([SHORT[v] for v in variants], fontsize=9.5)
            ax.set_xlim(-0.5, len(variants) - 0.2)
            ax.grid(axis='x', visible=False)
            ax.set_title(f'{title}\n{steps} sampling step{"s" if steps > 1 else ""}', fontsize=11)
            if c == 0:
                ax.set_ylabel('energy distance to real data (log)\nlower is better')
    fig.suptitle(f'Pairings for continuous conditions, epoch {EPOCH}: one dot per seed, bar = mean', y=1.0)
    fig.tight_layout()
    save(fig, out, 'continuous_overview.png')
    plt.close(fig)


def fig_continuous_steps(plt, out):
    """Energy distance vs. number of Euler steps, every pairing."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (task, title) in zip(axes, CONT_TASKS):
        rows = results(task, 'steps_sweep')
        for v in present(task):
            cont_band_line(ax, STEPS, np.stack([per_seed(rows, v, 'energy_distance', num_steps=s)
                                                for s in STEPS], 1), v)
        data_line(ax, data_value(task, 'energy_distance'))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS]); ax.minorticks_off()
        ax.set_xlabel('Euler steps at sampling (log)')
        ax.set_title(title)
    axes[0].set_ylabel('energy distance (log), lower is better')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f'Sampling steps, epoch {EPOCH}: mean over 5 seeds, shading = seed range', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, 'continuous_steps_sweep.png')
    plt.close(fig)


class ContinuousView:
    """Samples and yaw offsets of a Part-2 task, from its run configs."""

    def __init__(self, task_name):
        self.name = task_name
        cfg = _run_config(REPO_ROOT / f'checkpoints/{task_name}/seed_1', 'cond_pose_flow_matching_model_NOOT_NOCFG')
        self.config = cfg
        self.start = OmegaConf.to_container(cfg.training.start_dist_params, resolve=True)
        if cfg.training.get('task_type') == 'continuous':
            self.task = ContinuousGoalTask(OmegaConf.to_container(cfg.training.task_spec, resolve=True))
            self.conds = list(self.task.test_conditions())
        else:
            self.task = None
            self.discrete = Task(task_name, stem='cond_pose_flow_matching_model_NOOT_NOCFG')
            self.conds = self.discrete.conditions
            self.token_sigma = _token_sigma(cfg)

    def obs(self, i, n, device):
        if self.task is not None:
            return self.task.obs(self.conds[i].reshape(1, 2)).repeat(n, 1, 1).to(device)
        return _eval_obs(self.conds[i], n, device, self.token_sigma, i)

    def yaw_offset(self, poses, i):
        if self.task is None:
            return self.discrete.yaw_offset(poses, self.conds[i])
        c = self.conds[i].cpu()
        mid = self.task.base_yaw + self.task.yaw_gain * c[0] / self.task.radius
        R_mid = _quat_to_rot_mat(convert_twist_to_pose(torch.tensor([[0, 0, 0, 0, 0, float(mid)]]))[:, 3:])[0]
        return _rotvec(R_mid.T @ _quat_to_rot_mat(poses[:, 3:]))[:, 2]

    def real(self, i, n, seed):
        if self.task is None:
            return self.discrete.real(self.conds[i], n, seed)
        twists, _ = self.task.sample_goals(self.conds[i].cpu().reshape(1, 2).repeat(n, 1),
                                           generator=torch.Generator().manual_seed(seed))
        return convert_twist_to_pose(twists, dt=1.0, return_representation='quat')

    @torch.no_grad()
    def sample(self, variant, seed, steps, n, device):
        """n samples per condition from one seed's model; the same starts for every model."""
        stem = 'cond_pose_flow_matching_model' + variant[len('cond'):]
        model = _load_run(str(REPO_ROOT / f'checkpoints/{self.name}/seed_{seed}'), stem, EPOCH, device)[0]
        gen = torch.Generator().manual_seed(1000 + seed)
        return [generate_from_start_poses(model, _sample_start_poses(self.start, n, gen), self.obs(i, n, device),
                                          num_steps=steps, cfg_scale=1.0, device=device).cpu()
                for i in range(len(self.conds))]


def fig_continuous_violins(plt, out, device):
    """Per-sample rotation about z relative to the midpoint of each condition's two orientations."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5))
    for ax, (task_name, title) in zip(axes, CONT_TASKS):
        view = ContinuousView(task_name)
        n = 512 // len(view.conds)  # samples per condition and seed
        variants = present(task_name)
        groups = [torch.cat([view.yaw_offset(view.real(i, n * len(SEEDS), seed=7 + i), i)
                             for i in range(len(view.conds))]).numpy()]
        colors, labels = [MUTED], ['real data']
        for steps in (1, 3):
            for v in variants:
                yaws = [view.yaw_offset(p, i) for s in SEEDS for i, p in enumerate(view.sample(v, s, steps, n, device))]
                groups.append(torch.cat(yaws).numpy()); colors.append(pstyle(v)['color']); labels.append(SHORT[v])
        k = len(variants)
        positions = [0] + [1.3 + i + (i // k) * 0.8 for i in range(len(groups) - 1)]
        violins(ax, groups, positions, colors)
        for j, steps in enumerate((1, 3)):
            mid = (positions[1 + j * k] + positions[k + j * k]) / 2
            ax.text(mid, -0.2, f'{steps} step{"s" if steps > 1 else ""}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', color=INK, fontsize=11)
        for y in (-0.25, 0.25):
            ax.axhline(y, color=MUTED, ls=':', lw=1.3, zorder=1)
        ax.text(0.995, 0.25, ' target orientations ', ha='right', va='bottom', transform=ax.get_yaxis_transform(),
                color=INK2, fontsize=9.5)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(-0.8, 0.8)
        ax.set_ylabel('rotation about z relative to the\nmidpoint of the two orientations (rad)')
        ax.set_title(title)
    fig.suptitle(f'Do few-step samples keep both orientations? Epoch {EPOCH}, 5 seeds per violin '
                 f'(bar = middle 50%, dot = median)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'continuous_orientation_violins.png')
    plt.close(fig)


def fig_sensitivity(plt, out, task='pose_continuous_goals', variant='cond_C2OTFIX_NOCFG'):
    """The sensitivity sweep of one pairing: energy distance vs. steps for each setting."""
    ramp = ('#86b6ef', '#2a78d6', '#104281')  # one hue, light -> dark = small -> large
    panels = [('OT batch (network batches per assignment)',
               [('ot_batch_x1', '×1 (128)'), (None, '×4 (512), main runs'), ('ot_batch_x10', '×10 (1280)')]),
              ('condition weight cond_scale (smaller = looser)',
               [('cond_scale_0.35', '0.35 (skew 0.035)'), (None, '0.7, calibrated (skew 0.017)'),
                ('cond_scale_1.4', '1.4 (skew 0.002)')])]
    main_rows = results(task, 'steps_sweep')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (title, settings) in zip(axes, panels):
        vals = np.stack([per_seed(main_rows, 'cond_NOOT_NOCFG', 'energy_distance', num_steps=n) for n in STEPS], 1)
        ax.fill_between(STEPS, vals.min(0), vals.max(0), color=ORANGE, alpha=0.12, lw=0)
        ax.plot(STEPS, vals.mean(0), color=ORANGE, ls='--', lw=1.8, marker='s', ms=6,
                label='random pairing (I-CFM), 5 seeds')
        for color, (name, label) in zip(ramp, settings):
            if name is None:
                rows = main_rows
            else:
                d = REPO_ROOT / f'experiments/results/{task}/sensitivity/{name}/epoch_{EPOCH}'
                rows = {s: read(d / f'seed_{s}/steps_sweep.csv') for s in (1, 2, 3)}
            vals = np.stack([per_seed(rows, variant, 'energy_distance', num_steps=n) for n in STEPS], 1)
            ax.fill_between(STEPS, vals.min(0), vals.max(0), color=color, alpha=0.12, lw=0)
            ax.plot(STEPS, vals.mean(0), color=color, lw=1.8, marker='o', ms=6,
                    label=f'{label}, {len(rows)} seeds')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(STEPS); ax.set_xticklabels([str(n) for n in STEPS]); ax.minorticks_off()
        ax.set_xlabel('Euler steps at sampling (log)')
        ax.set_title(title, fontsize=11.5)
        ax.legend(fontsize=9.5, loc='upper right')
    axes[0].set_ylabel('energy distance (log), lower is better')
    fig.suptitle('Fixed-weight pairing on continuous goals: one setting varied at a time '
                 '(mean, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'continuous_sensitivity.png')
    plt.close(fig)


# -- Part 3: the current model and training framework ---------------------------------------

SOTA_TASKS = [  # (results folder, panel title)
    ('sota/pose_corners_3sigma', 'Discrete conditions (clean tokens)'),
    ('sota/pose_corners_3sigma_jitter', 'Continuous conditions (noisy tokens)'),
]
OLD_FRAMEWORK = {'sota/pose_corners_3sigma': 'pose_corners_3sigma'}  # the same task's pre-framework runs
# Phase 3's baseline candidates: conditional, no CFG; for noisy tokens neither global OT nor the
# per-corner oracle, which uses the clean corner ids.
BASELINE_EXCLUDE = ('cond_GOT_NOCFG', 'cond_OT_NOCFG')
ABLATION_AXES = [  # (title, [(arm, label)]); arm None = the baseline itself
    ('Training-time density', [(None, 'uniform (baseline)'), ('t_logit_normal', 'logit-normal (SD3)'),
                               ('t_beta', 'Beta, noisy end (π0)')]),
    ('Time embedding', [(None, 'sinusoids of t (baseline)'), ('time_x1000', 'sinusoids of 1000·t'),
                        ('time_fourier', 'Gaussian Fourier features')]),
    ('Condition pathway', [(None, 'adaLN (baseline)'), ('cond_cross_attn', 'cross-attention'),
                           ('cond_joint', 'joint attention')]),
]
ARM_COLORS = (INK2, BLUE, '#eda100')
ARM_STYLE = (dict(ls='--', marker='s'), dict(ls='-', marker='o'), dict(ls='-', marker='^'))
SOLVERS = [('euler', 'Euler', INK2, 's'), ('midpoint', 'midpoint', BLUE, 'o'), ('heun', 'Heun', '#eda100', '^')]
NFE = (2, 4, 6, 10, 18, 40, 100)
INTERVALS = [('full', 'whole path', INK2), ('0-0.5', 't < 0.5 (noisy half)', BLUE),
             ('0.25-0.75', '0.25 ≤ t < 0.75', '#1baf7a'), ('0.5-1', 't ≥ 0.5 (data half)', '#eda100')]
GUIDANCE_SCALES = (1.0, 1.5, 3.0)


def is_discrete(task):
    return not task.endswith('_jitter')


def vstyle(task, variant):
    """Line style of a variant: Part 1's for clean tokens, Part 2's per pairing for noisy ones."""
    if is_discrete(task):
        st = style(variant)
        return dict(color=st['color'], ls=st['ls'], marker=st['marker'], mfc=st['mfc'], label=LABEL[variant])
    st = pstyle(variant)
    return dict(color=st['color'], ls=st['ls'], marker=st['marker'], mfc=st['color'], label=st['label'])


def sota_variants(task):
    return COND if is_discrete(task) else present(task)


def baseline(task):
    """Phase 3's baseline variant, by the rule set before the results: among the conditional
    no-CFG candidates, the lowest mean over seeds of (log10 ED at 3 steps + log10 ED at 100) / 2.
    Returns (variant, {candidate: score})."""
    steps_rows, main_rows = results(task, 'steps_sweep'), results(task, 'metrics')
    scores = {}
    for v in sota_variants(task):
        if not v.endswith('_NOCFG') or (not is_discrete(task) and v in BASELINE_EXCLUDE):
            continue
        three = np.log10(per_seed(steps_rows, v, 'energy_distance', num_steps=3))
        hundred = np.log10(per_seed(main_rows, v, 'energy_distance'))
        scores[v] = float(((three + hundred) / 2).mean())
    return min(scores, key=scores.get), scores


def ablation_rows(task, arm, name):
    """{seed: rows} of one ablation arm's <name>.csv (arm None = the task's own results)."""
    if arm is None:
        return results(task, name)
    d = REPO_ROOT / f'experiments/results/{task}/ablate/{arm}/epoch_{EPOCH}'
    return {s: read(d / f'seed_{s}/{name}.csv') for s in SEEDS}


def fig_sota_framework(plt, out, task='sota/pose_corners_3sigma'):
    """The new framework against the old one on the same task: sampling steps, spread, training."""
    old = OLD_FRAMEWORK[task]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for v in ('cond_OT_NOCFG', 'cond_NOOT_NOCFG'):
        color = style(v)['color']
        for t, framework, ls, mfc in ((task, 'current', '-', color), (old, 'old', ':', SURFACE)):
            label = f'{LABEL[v]}, {framework} framework'
            kw = dict(color=color, ls=ls, marker=style(v)['marker'], mfc=mfc, mec=color, lw=1.8, ms=6, label=label)
            rows = results(t, 'steps_sweep')
            for ax, metric in zip(axes[:2], ('energy_distance', 'spread_ratio_trans')):
                vals = np.stack([per_seed(rows, v, metric, num_steps=s) for s in STEPS], 1)
                ax.fill_between(STEPS, vals.min(0), vals.max(0), color=color, alpha=0.08, lw=0)
                ax.plot(STEPS, vals.mean(0), **kw)
            x, vals = curve(t, v, 'energy_distance', 100)
            axes[2].fill_between(x, vals.min(0), vals.max(0), color=color, alpha=0.08, lw=0)
            axes[2].plot(x, vals.mean(0), **kw)
    data_line(axes[0], data_value(task, 'energy_distance'))
    data_line(axes[1], data_value(task, 'spread_ratio_trans'))
    data_line(axes[2], data_value(task, 'energy_distance'))
    for ax in axes[:2]:
        ax.set_xscale('log'); ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS]); ax.minorticks_off()
        ax.set_xlabel('Euler steps at sampling (log)')
    axes[0].set_yscale('log'); axes[2].set_yscale('log')
    axes[0].set_ylabel('energy distance (log), lower is better')
    axes[1].set_ylabel('position spread ÷ real data')
    axes[2].set_ylabel('energy distance at 100 steps (log)')
    axes[2].set_xlabel('training epoch')
    axes[0].set_title('Sampling steps, epoch 100'); axes[1].set_title('Spread around the mode, epoch 100')
    axes[2].set_title('Training')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Every mode within 3σ, discrete conditions: current vs. old model and training framework '
                 f'(mean over {len(SEEDS)} seeds, shading = seed range)', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, 'sota_framework.png')
    plt.close(fig)


def fig_sota_overview(plt, out):
    """Energy distance per configuration at 1, 3 and 100 steps, both conditioning types."""
    steps_list = (1, 3, 100)
    fig, axes = plt.subplots(len(SOTA_TASKS), 3, figsize=(16, 4.3 * len(SOTA_TASKS)), squeeze=False)
    for r, (task, title) in enumerate(SOTA_TASKS):
        steps_rows, main_rows = results(task, 'steps_sweep'), results(task, 'metrics')
        variants = sota_variants(task)
        for c, steps in enumerate(steps_list):
            ax = axes[r, c]
            for i, v in enumerate(variants):
                vals = (per_seed(main_rows, v, 'energy_distance') if steps == 100
                        else per_seed(steps_rows, v, 'energy_distance', num_steps=steps))
                st = vstyle(task, v)
                ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=36, marker=st['marker'],
                           facecolors=st['mfc'], edgecolors=st['color'], linewidths=1.5, zorder=3)
                ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=st['color'], lw=2.5, zorder=4)
                ax.text(i, vals.max() * 1.35, f'{vals.mean():.3g}', ha='center', va='bottom', fontsize=9,
                        color=INK2)
            data_line(ax, data_value(task, 'energy_distance'), where='left')
            ax.set_yscale('log')
            ax.set_ylim(None, ax.get_ylim()[1] * 2.5)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([LABEL[v].replace(' (', '\n(').replace(', ', '\n', 1) if is_discrete(task)
                                else SHORT[v] for v in variants], fontsize=9)
            ax.set_xlim(-0.5, len(variants) - 0.5)
            ax.grid(axis='x', visible=False)
            ax.set_title(f'{title}\n{steps} sampling step{"s" if steps > 1 else ""}', fontsize=11)
            if c == 0:
                ax.set_ylabel('energy distance to real data (log)\nlower is better')
    fig.suptitle(f'Every mode within 3σ, current framework, epoch {EPOCH}: one dot per seed, bar = mean '
                 f'(CFG models at guidance {GUIDANCE:g})', y=1.0)
    fig.tight_layout()
    save(fig, out, 'sota_overview.png')
    plt.close(fig)


def fig_sota_steps(plt, out):
    """Energy distance vs. Euler steps, every configuration, both conditioning types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, (task, title) in zip(axes, SOTA_TASKS):
        rows = results(task, 'steps_sweep')
        for v in sota_variants(task):
            st = vstyle(task, v)
            vals = np.stack([per_seed(rows, v, 'energy_distance', num_steps=s) for s in STEPS], 1)
            ax.fill_between(STEPS, vals.min(0), vals.max(0), color=st['color'], alpha=0.10, lw=0)
            ax.plot(STEPS, vals.mean(0), color=st['color'], ls=st['ls'], marker=st['marker'], mfc=st['mfc'],
                    mec=st['color'], lw=1.8, ms=6, label=st['label'])
        data_line(ax, data_value(task, 'energy_distance'))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS]); ax.minorticks_off()
        ax.set_xlabel('Euler steps at sampling (log)')
        ax.set_title(title)
        ax.legend(fontsize=9, loc='lower left')
    axes[0].set_ylabel('energy distance (log), lower is better')
    fig.suptitle(f'Sampling steps, current framework, epoch {EPOCH}: mean over {len(SEEDS)} seeds, shading = seed range',
                 y=1.0)
    fig.tight_layout()
    save(fig, out, 'sota_steps_sweep.png')
    plt.close(fig)


def fig_sota_ablations(plt, out):
    """Single-axis ablations: energy distance vs. Euler steps for the baseline and each arm."""
    fig, axes = plt.subplots(len(SOTA_TASKS), 3, figsize=(17, 4.5 * len(SOTA_TASKS)), sharex=True, squeeze=False)
    for r, (task, title) in enumerate(SOTA_TASKS):
        base, _ = baseline(task)
        for c, (axis_title, arms) in enumerate(ABLATION_AXES):
            ax = axes[r, c]
            for (arm, label), color, st in zip(arms, ARM_COLORS, ARM_STYLE):
                rows = ablation_rows(task, arm, 'steps_sweep')
                vals = np.stack([per_seed(rows, base, 'energy_distance', num_steps=s) for s in STEPS], 1)
                ax.fill_between(STEPS, vals.min(0), vals.max(0), color=color, alpha=0.10, lw=0)
                ax.plot(STEPS, vals.mean(0), color=color, lw=1.8, ms=6, label=label, **st)
            data_line(ax, data_value(task, 'energy_distance'))
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xticks(STEPS); ax.set_xticklabels([str(s) for s in STEPS]); ax.minorticks_off()
            ax.set_title(f'{axis_title}\n{title}', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            if c == 0:
                ax.set_ylabel('energy distance (log), lower is better')
            if r == len(SOTA_TASKS) - 1:
                ax.set_xlabel('Euler steps at sampling (log)')
    fig.suptitle(f'Single-axis ablations, epoch {EPOCH}: each arm changes one choice of the baseline '
                 f'(mean over {len(SEEDS)} seeds, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'sota_ablations.png')
    plt.close(fig)


def fig_sota_solvers(plt, out):
    """ODE solvers at equal network evaluations, on each task's baseline (unguided)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (task, title) in zip(axes, SOTA_TASKS):
        base, _ = baseline(task)
        rows = results(task, 'solver_sweep')
        for method, label, color, marker in SOLVERS:
            vals = np.stack([per_seed(rows, base, 'energy_distance', method=method, nfe=n) for n in NFE], 1)
            ax.fill_between(NFE, vals.min(0), vals.max(0), color=color, alpha=0.10, lw=0)
            ax.plot(NFE, vals.mean(0), color=color, marker=marker, lw=1.8, ms=6, label=label)
        data_line(ax, data_value(task, 'energy_distance'))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(NFE); ax.set_xticklabels([str(n) for n in NFE]); ax.minorticks_off()
        ax.set_xlabel('network evaluations per sample (log)')
        ax.set_title(f"{title}\nbaseline: {vstyle(task, base)['label']}", fontsize=11)
        ax.legend(fontsize=10)
    axes[0].set_ylabel('energy distance (log), lower is better')
    fig.suptitle(f'ODE solvers at equal cost, epoch {EPOCH} (mean over {len(SEEDS)} seeds, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, 'sota_solvers.png')
    plt.close(fig)


def fig_sota_guidance(plt, out):
    """Guidance over the whole path or only within a flow-time interval, CFG-trained models."""
    fig, axes = plt.subplots(len(SOTA_TASKS), 3, figsize=(17, 4.5 * len(SOTA_TASKS)), squeeze=False)
    for r, (task, title) in enumerate(SOTA_TASKS):
        base, _ = baseline(task)
        if is_discrete(task):
            variant, rows = base.replace('_NOCFG', '_CFG'), results(task, 'guidance_sweep')
        else:
            variant, rows = base.replace('_NOCFG', '_CFG'), ablation_rows(task, 'cfg', 'guidance_sweep')
        panels = (('energy_distance', 9), ('energy_distance', 100), ('mode_accuracy', 100))
        for c, (metric, steps) in enumerate(panels):
            ax = axes[r, c]
            unguided = per_seed(rows, variant, metric, num_steps=steps, cfg_interval='full',
                                cfg_scale_at_inference=1.0)
            for interval, label, color in INTERVALS:
                vals = np.stack([unguided] + [per_seed(rows, variant, metric, num_steps=steps,
                                                       cfg_interval=interval, cfg_scale_at_inference=g)
                                              for g in GUIDANCE_SCALES[1:]], 1)
                ax.fill_between(GUIDANCE_SCALES, vals.min(0), vals.max(0), color=color, alpha=0.10, lw=0)
                ax.plot(GUIDANCE_SCALES, vals.mean(0), color=color, marker='o', lw=1.8, ms=6, label=label)
            data_line(ax, data_value(task, metric))
            if metric == 'energy_distance':
                ax.set_yscale('log')
                ax.set_ylabel('energy distance (log), lower is better')
            else:
                ax.set_ylabel('corner accuracy (data = ceiling;\nhigher over-separates)')
            ax.set_xticks(GUIDANCE_SCALES); ax.set_xticklabels([f'{g:g}' for g in GUIDANCE_SCALES])
            ax.set_title(f'{title}\n{steps} steps', fontsize=11)
            if r == len(SOTA_TASKS) - 1:
                ax.set_xlabel('guidance scale')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03),
               title='guidance applied over')
    fig.suptitle(f'Guidance intervals, CFG-trained baselines, epoch {EPOCH} '
                 f'(mean over {len(SEEDS)} seeds, shading = seed range)', y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, 'sota_guidance.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='docs/ablations')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--part', choices=('discrete', 'continuous', 'sota'), default='discrete')
    parser.add_argument('--only', nargs='+', default=None,
                        help='continuous part: just these figures (calibration toys overview steps violins sensitivity); '
                             'sota part: baseline framework overview steps ablations solvers guidance')
    args = parser.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    if args.part == 'continuous':
        figs = {'calibration': lambda: fig_calibration(plt, out),
                'toys': lambda: fig_toys(plt, out, args.device),
                'overview': lambda: fig_continuous_overview(plt, out),
                'steps': lambda: fig_continuous_steps(plt, out),
                'violins': lambda: fig_continuous_violins(plt, out, args.device),
                'sensitivity': lambda: fig_sensitivity(plt, out)}
        for name in args.only or figs:
            figs[name]()
        return
    if args.part == 'sota':
        def print_baselines():
            for task, _ in SOTA_TASKS:
                choice, scores = baseline(task)
                print(f'{task}: baseline {choice}; mean (log10 ED@3 + log10 ED@100) / 2: '
                      + ', '.join(f'{v} {s:.3f}' for v, s in sorted(scores.items(), key=lambda kv: kv[1])))
        figs = {'baseline': print_baselines,
                'framework': lambda: fig_sota_framework(plt, out),
                'overview': lambda: fig_sota_overview(plt, out),
                'steps': lambda: fig_sota_steps(plt, out),
                'ablations': lambda: fig_sota_ablations(plt, out),
                'solvers': lambda: fig_sota_solvers(plt, out),
                'guidance': lambda: fig_sota_guidance(plt, out)}
        for name in args.only or figs:
            figs[name]()
        return
    fig_overview(plt, out)
    fig_steps(plt, out)
    fig_guidance(plt, out)
    fig_training(plt, out)
    with torch.no_grad():
        fig_orientation_violins(plt, out, args.device)
        fig_guidance_violins(plt, out, args.device)
    copy_3d(out)


if __name__ == '__main__':
    main()
