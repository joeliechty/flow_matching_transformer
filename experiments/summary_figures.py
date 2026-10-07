"""Figures for ablations_summary.md: the pose ablations on the two multimodal tasks.

Reads what ./pose_ablations.sh writes for each task (the eval stage at epoch 100, the curves
stage and the training logs), samples the epoch-100 models for the per-sample figures, and
writes the PNGs that ablations_summary.md embeds:

    python experiments/summary_figures.py            # -> docs/ablations/

Colour = OT (blue) or no OT (orange); dashed lines / open markers = CFG-trained models sampled
at guidance 3; grey = real data.
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

from pose_gen_inference import _generate, _load_run, _run_config, _sample_start_poses  # noqa: E402
from utils.eval_utils import _rotvec, goal_mode_poses_from_config, sample_goal_mixture  # noqa: E402
from utils.pose_task import task_conditions  # noqa: E402
from utils.tf_utils import _quat_to_rot_mat  # noqa: E402

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


def per_seed(rows_by_seed, variant, metric, **match):
    """One value per seed for `variant`, from the row matching `match` (e.g. num_steps=3)."""
    out = []
    for rows in rows_by_seed.values():
        row = [r for r in rows if r['variant'] == variant
               and all(float(r[k]) == float(v) for k, v in match.items())]
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

    def __init__(self, task):
        cfg = _run_config(REPO_ROOT / f'checkpoints/{task}/seed_1', STEM['cond_OT_NOCFG']).training
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='docs/ablations')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
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
