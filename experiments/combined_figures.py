"""Figures for ablations_continuous.md and ablations_discrete.md: each topic shown on both
evaluations side by side, the synthetic 3σ task and ACRONYM grasps.

    python experiments/combined_figures.py                     # both documents -> docs/combined/
    python experiments/combined_figures.py --kind continuous --only scorecard

Synthetic (experiments/results/sota_geo/, 5 seeds): energy distance to real data, 0 = matches it.
ACRONYM (experiments/results/sota/, 3 seeds): 1-NN accuracy against held-out real grasps, 0.5 = matches it;
continuous conditioning is scored on unseen objects, discrete on held-out grasps of training objects.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.summary_figures import (BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, data_line,  # noqa: E402
                                         ed_axis, save, setup_matplotlib)

GREEN, YELLOW, PINK = '#1baf7a', '#eda100', '#e87ba4'
EVALS = {  # evaluation -> (results root, title, metric, metric label)
    'syn': (REPO_ROOT / 'experiments/results/sota_geo', 'Synthetic 3σ task', 'energy_distance',
            'energy distance to real data\n(lower is better; 0 = matches the data)'),
    'acr': (REPO_ROOT / 'experiments/results/sota', 'ACRONYM grasps', 'one_nna',
            '1-NN accuracy vs. held-out grasps\n(lower is better; 0.5 = matches the data)'),
}
KINDS = {  # kind -> {evaluation: (task folder, level, baseline variant, condition word)}
    'continuous': {'syn': ('pose_corners_3sigma_jitter', None, 'cond_C2OT_NOCFG', 'noisy corner tokens'),
                   'acr': ('pose_acronym_10cat_points', 'heldout_objects', 'cond_OT_NOCFG', 'point clouds, unseen objects')},
    'discrete': {'syn': ('pose_corners_3sigma', None, 'cond_OT_NOCFG', 'corner tokens'),
                 'acr': ('pose_acronym_10cat_ids', 'heldout_grasps', 'cond_OT_NOCFG', 'object ids, held-out grasps')},
}
SEED_SD_FLOOR = {'energy_distance': 1e-4, 'one_nna': 1e-3, 'mode_accuracy': 1e-3}
STEPS = (1, 2, 3, 5, 9, 20, 50, 100)


# -- reading results ------------------------------------------------------------------------

def _dir(ev, kind, sub=(), level='default'):
    task, lvl, _, _ = KINDS[kind][ev]
    d = EVALS[ev][0] / task
    for s in sub:
        d = d / s
    d = d / 'epoch_100'
    lvl = lvl if level == 'default' else level
    return d / lvl if lvl else d


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _match(row, **match):
    for k, v in match.items():
        cell = row.get(k)
        if cell in (None, ''):
            if k == 'method' and v == 'euler':
                continue
            return False
        if isinstance(v, str) and cell != v:
            return False
        if not isinstance(v, str) and float(cell) != float(v):
            return False
    return True


def per_seed(ev, kind, variant, metric, name='metrics', sub=(), level='default', **match):
    """One value per seed of `metric` for `variant`, from <name>.csv rows matching `match`."""
    d = _dir(ev, kind, sub, level)
    out = []
    for seed_dir in sorted(d.glob('seed_*')):
        path = seed_dir / f'{name}.csv'
        if not path.exists():
            continue
        rows = [r for r in _read(path) if r['variant'] == variant and _match(r, **match)]
        if len(rows) != 1:
            raise ValueError(f'{path}: {variant} {match}: {len(rows)} rows')
        out.append(float(rows[0][metric]))
    if not out:
        raise FileNotFoundError(f'no {name}.csv under {d}')
    return np.array(out)


def at_steps(ev, kind, variant, steps, sub=(), metric=None, level='default'):
    """Per-seed metric at `steps` Euler steps (100 = the main evaluation)."""
    metric = metric or EVALS[ev][2]
    if steps == 100:
        return per_seed(ev, kind, variant, metric, 'metrics', sub, level)
    return per_seed(ev, kind, variant, metric, 'steps_sweep', sub, level, num_steps=steps, method='euler')


def data_value(ev, kind, metric=None, level='default'):
    metric = metric or EVALS[ev][2]
    d = _dir(ev, kind, (), level)
    rows = _read(sorted(d.glob('seed_*'))[0] / 'metrics.csv')
    return float(next(r for r in rows if r['variant'] == 'data')[metric])


def effect(alt, rec, metric):
    """Difference of means in pooled seed SDs (negative = alt is better), with an SD floor."""
    sd = np.sqrt((alt.var(ddof=1) + rec.var(ddof=1)) / 2)
    return (alt.mean() - rec.mean()) / max(sd, SEED_SD_FLOOR[metric])


# -- shared drawing -------------------------------------------------------------------------

def metric_axis(ax, ev):
    if EVALS[ev][2] == 'energy_distance':
        ed_axis(ax)
    ax.set_ylabel(EVALS[ev][3])


def steps_axis(ax, ticks=STEPS, label='Euler steps at sampling (log)'):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(s) for s in ticks])
    ax.minorticks_off()
    ax.set_xlabel(label)


def band(ax, x, vals, color, label, ls='-', marker='o'):
    ax.fill_between(x, vals.min(0), vals.max(0), color=color, alpha=0.12, lw=0)
    ax.plot(x, vals.mean(0), color=color, ls=ls, marker=marker, lw=1.8, ms=6, label=label)


def panel_title(ax, ev, kind, extra=''):
    ax.set_title(f'{EVALS[ev][1]}: {KINDS[kind][ev][3]}{extra}', fontsize=11)


def two_panels(plt, width=15, height=5.2):
    return plt.subplots(1, 2, figsize=(width, height))


# -- figures --------------------------------------------------------------------------------

def scorecard_rows(kind):
    """(label, {ev: ((alt sub, alt variant), (rec sub, rec variant))}) for each alternative."""
    S, A = KINDS[kind]['syn'][2], KINDS[kind]['acr'][2]
    base = lambda v: ((), v)
    arm = lambda a, v: (('ablate', a), v)
    both = lambda a: {'syn': (arm(a, S), base(S)), 'acr': (arm(a, A), base(A))}
    rows = []
    if kind == 'continuous':
        rows += [('Random pairing\n(instead of condition-aware OT)',
                  {'syn': (base('cond_NOOT_NOCFG'), base(S)), 'acr': (base('cond_NOOT_NOCFG'), base(A))}),
                 ('Global OT\n(pairs across conditions)',
                  {'syn': (base('cond_GOT_NOCFG'), base(S)), 'acr': (base('cond_GOT_NOCFG'), base(A))}),
                 ('adaLN conditioning\n(instead of attention)',
                  {'syn': (base(S), arm('cond_joint', S)), 'acr': (arm('cond_adaln', A), base(A))}),
                 ('Joint attention\n(instead of cross-attention)',
                  {'syn': (arm('cond_joint', S), arm('cond_cross_attn', S)), 'acr': (arm('cond_joint', A), base(A))}),
                 ('CFG training,\nguidance 3', {'syn': (arm('cfg', 'cond_C2OT_CFG'), base(S)),
                                               'acr': (base('cond_OT_CFG'), base(A))})]
    else:
        rows += [('Random pairing\n(instead of per-condition OT)',
                  {'syn': (base('cond_NOOT_NOCFG'), base(S)), 'acr': (base('cond_NOOT_NOCFG'), base(A))}),
                 ('adaLN conditioning\n(instead of joint attention)',
                  {'syn': (base(S), arm('cond_joint', S)), 'acr': (arm('cond_adaln', A), arm('cond_joint', A))}),
                 ('Cross-attention\n(instead of joint attention)',
                  {'syn': (arm('cond_cross_attn', S), arm('cond_joint', S)), 'acr': (base(A), arm('cond_joint', A))}),
                 ('CFG training,\nguidance 3', {'syn': (base('cond_OT_CFG'), base(S)),
                                               'acr': (base('cond_OT_CFG'), base(A))})]
    rows += [('Logit-normal training times\n(instead of uniform)', both('t_logit_normal')),
             ('Beta training times, π0\n(instead of uniform)', both('t_beta')),
             ('Time embedding: sinusoids of 1000·t', {'syn': (arm('time_x1000', S), base(S))}),
             ('Time embedding: Fourier features', {'syn': (arm('time_fourier', S), base(S))}),
             ('Gaussian-twist rotation prior\n(instead of uniform on SO(3))', {'acr': (arm('rot_gauss', A), base(A))}),
             ('64 samples per condition per batch\n(instead of 16)', {'acr': (arm('k64', A), base(A))}),
             ('4 samples per condition per batch\n(instead of 16)', {'acr': (arm('k4', A), base(A))})]
    return rows


def fig_scorecard(plt, out, kind, cap=60):
    """Every alternative against the recommended setting, in both evaluations, at 3 and 100 steps."""
    rows = scorecard_rows(kind)
    styles = {'syn': dict(color=BLUE, marker='o', label=EVALS['syn'][1]),
              'acr': dict(color=ORANGE, marker='s', label=EVALS['acr'][1])}
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(rows) + 1.8), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, steps in zip(axes, (3, 100)):
        labelled = set()
        ax.axvspan(-2, 2, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=1)
        for yi, (label, evs) in zip(y, rows):
            for k, ev in enumerate(('syn', 'acr')):
                if ev not in evs:
                    continue
                (asub, avar), (rsub, rvar) = evs[ev]
                metric = EVALS[ev][2]
                z = effect(at_steps(ev, kind, avar, steps, asub), at_steps(ev, kind, rvar, steps, rsub), metric)
                zc = float(np.clip(z, -cap, cap))
                yy = yi + (0.14 if ev == 'syn' else -0.14)
                ax.scatter(zc, yy, s=60, zorder=3, color=styles[ev]['color'], marker=styles[ev]['marker'],
                           label=None if ev in labelled else styles[ev]['label'])
                labelled.add(ev)
                if abs(z) > cap:
                    ax.annotate(f'{z:+.0f}', (zc, yy), xytext=(-6 if z > 0 else 6, 0), textcoords='offset points',
                                ha='right' if z > 0 else 'left', va='center', fontsize=8.5, color=INK2)
        ax.set_xscale('symlog', linthresh=2, linscale=0.6)
        ax.set_xlim(-cap * 1.25, cap * 1.25)
        ax.set_xticks([-50, -10, -2, 0, 2, 10, 50])
        ax.set_xticklabels(['−50', '−10', '−2', '0', '2', '10', '50'])
        ax.set_xlabel('difference from the recommended setting, in seed SDs\n← better            worse →')
        ax.set_title(f'{steps} Euler steps', fontsize=12)
        ax.grid(axis='y', visible=False)
        ax.text(0, y[0] + 0.62, 'no clear difference', ha='center', va='bottom', fontsize=9, color=INK2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=9.5)
    axes[0].set_ylim(-0.7, len(rows) - 0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.6, -0.02))
    fig.suptitle(f'{kind.capitalize()} conditioning: each alternative against the recommended setting, in both '
                 'evaluations (missing marker = not tested there)', y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, f'{kind}_scorecard.png')
    plt.close(fig)


def fig_pairing(plt, out, kind):
    S, A = KINDS[kind]['syn'][2], KINDS[kind]['acr'][2]
    rec = 'C²OT (condition-aware OT)' if kind == 'continuous' else 'per-condition OT'
    lines = {'syn': [((), S, rec, BLUE, '-', 'o'), ((), 'cond_NOOT_NOCFG', 'random pairing', ORANGE, '-', 's')],
             'acr': [((), A, 'per-object OT (16 grasps of each of 64 objects)', BLUE, '-', 'o'),
                     ((), 'cond_NOOT_NOCFG', 'random pairing', ORANGE, '-', 's'),
                     (('ablate', 'k64'), A, 'per-object OT, 64 grasps of each of 16 objects', GREEN, '--', '^'),
                     (('ablate', 'k4'), A, 'per-object OT, 4 grasps of each of 256 objects', YELLOW, '--', 'v')]}
    if kind == 'continuous':
        lines['syn'] += [((), 'cond_GOT_NOCFG', 'global OT (ignores the condition)', PINK, ':', 'D'),
                         ((), 'cond_OT_NOCFG', 'per-corner OT (oracle: knows the corner)', INK2, ':', 'x')]
        lines['acr'] += [((), 'cond_GOT_NOCFG', 'global OT (ignores the object)', PINK, ':', 'D')]
    fig, axes = two_panels(plt, height=5.6)
    for ax, ev in zip(axes, ('syn', 'acr')):
        for sub, var, label, color, ls, marker in lines[ev]:
            vals = np.stack([at_steps(ev, kind, var, s, sub) for s in STEPS], 1)
            band(ax, STEPS, vals, color, label, ls, marker)
        data_line(ax, data_value(ev, kind), 'real data' if ev == 'acr' else 'one real draw', where='left')
        steps_axis(ax)
        metric_axis(ax, ev)
        panel_title(ax, ev, kind)
        ax.legend(fontsize=9, loc='upper right')
    fig.suptitle('Pairing noise with data: quality against sampling steps (mean over seeds, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{kind}_pairing.png')
    plt.close(fig)


def _strip(ax, groups, colors, labels, ref=None, ref_label='real data'):
    for i, (vals, color) in enumerate(zip(groups, colors)):
        ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=36, color=color, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=color, lw=2.5, zorder=4)
    if ref is not None:
        data_line(ax, ref, ref_label, where='left')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(axis='x', visible=False)


def fig_pathway(plt, out, kind):
    """Condition pathway: adaLN, cross-attention, joint attention; one dot per seed."""
    S, A = KINDS[kind]['syn'][2], KINDS[kind]['acr'][2]
    arms = {'syn': [((), S, 'adaLN'), (('ablate', 'cond_cross_attn'), S, 'cross'), (('ablate', 'cond_joint'), S, 'joint')],
            'acr': [(('ablate', 'cond_adaln'), A, 'adaLN\n(pooled)'), ((), A, 'cross'), (('ablate', 'cond_joint'), A, 'joint')]}
    colors = [MUTED, BLUE, GREEN]
    if kind == 'continuous':
        panels = [('syn', 100, 'mode_accuracy'), ('syn', 100, None), ('acr', 3, None), ('acr', 100, None)]
    else:
        panels = [('syn', 3, None), ('syn', 100, None), ('acr', 3, None), ('acr', 100, None)]
    short = {'energy_distance': 'energy distance (lower is better)', 'one_nna': '1-NN accuracy (lower is better;\nreal data 0.50)'}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.6))
    for ax, (ev, steps, metric) in zip(axes, panels):
        groups = [at_steps(ev, kind, v, steps, sub, metric) for sub, v, _ in arms[ev]]
        if metric == 'mode_accuracy':
            _strip(ax, groups, colors, [l for _, _, l in arms[ev]], data_value(ev, kind, metric), 'real data')
            ax.set_ylabel('corner accuracy\n(real data = target; higher over-separates)')
        else:
            _strip(ax, groups, colors, [l for _, _, l in arms[ev]])
            ax.set_ylabel(short[EVALS[ev][2]])
        ax.set_title(f"{EVALS[ev][1]}\n{steps} Euler steps", fontsize=11)
    fig.suptitle(f'How the condition enters the transformer ({kind} conditioning): one dot per seed, bar = mean',
                 y=1.0)
    fig.tight_layout()
    save(fig, out, f'{kind}_pathway.png')
    plt.close(fig)


NFE = (2, 4, 6, 10, 18, 40, 100)


def fig_solvers(plt, out, kind):
    fig, axes = two_panels(plt)
    for ax, ev in zip(axes, ('syn', 'acr')):
        base = KINDS[kind][ev][2]
        for method, label, color, marker in (('euler', 'Euler', INK2, 's'), ('midpoint', 'midpoint', BLUE, 'o'),
                                             ('heun', 'Heun', YELLOW, '^')):
            vals = np.stack([per_seed(ev, kind, base, EVALS[ev][2], 'solver_sweep', method=method, nfe=n)
                             for n in NFE], 1)
            band(ax, NFE, vals, color, label, marker=marker)
        data_line(ax, data_value(ev, kind), 'real data' if ev == 'acr' else 'one real draw', where='left')
        steps_axis(ax, NFE, 'network evaluations per sample (log); midpoint and Heun use 2 per step')
        metric_axis(ax, ev)
        panel_title(ax, ev, kind)
        ax.legend(fontsize=10, loc='upper right')
    fig.suptitle('ODE solvers at equal cost, on the recommended model (mean over seeds, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{kind}_solvers.png')
    plt.close(fig)


GUIDANCE = (1.0, 1.1, 1.2, 1.3, 1.5, 3.0)
SMALL_GUIDANCE = (1.0, 1.1, 1.2, 1.3, 1.5)  # the zoomed row: Ho & Salimans' best FID was at 1.1-1.3
INTERVALS = [('full', 'whole path', INK2, 's'), ('0-0.5', 't < 0.5 (noisy half)', BLUE, 'o'),
             ('0.25-0.75', '0.25 ≤ t < 0.75 (middle)', GREEN, 'D'), ('0.5-1', 't ≥ 0.5 (data half)', YELLOW, '^')]


def guidance_sources(kind):
    """{evaluation: (sub-folder, CFG variant)} for the guidance sweeps."""
    return {'syn': (('ablate', 'cfg'), 'cond_C2OT_CFG') if kind == 'continuous' else ((), 'cond_OT_CFG'),
            'acr': ((), 'cond_OT_CFG')}


def guided(ev, kind, metric, interval, scales):
    """[seeds, scales] of `metric` at 100 steps, guiding within `interval` (scale 1 = unguided)."""
    sub, var = guidance_sources(kind)[ev]
    return np.stack([per_seed(ev, kind, var, metric, 'guidance_sweep', sub, num_steps=100,
                              cfg_interval='full' if g == 1.0 else interval, cfg_scale_at_inference=g)
                     for g in scales], 1)


def fig_guidance(plt, out, kind):
    """Guidance scale and interval at 100 steps, on the CFG-trained recommended model; the bottom row zooms
    into the small scales."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, scales in enumerate((GUIDANCE, SMALL_GUIDANCE)):
        for ax, ev in zip(axes[row], ('syn', 'acr')):
            for interval, label, color, marker in INTERVALS:
                band(ax, scales, guided(ev, kind, EVALS[ev][2], interval, scales), color, f'guidance on {label}',
                     marker=marker)
            if row == 0:
                data_line(ax, data_value(ev, kind), 'real data' if ev == 'acr' else 'one real draw', where='left')
            ticks = scales if row else (1.0, 1.5, 3.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels(['1\n(none)' if g == 1.0 else f'{g:g}' for g in ticks])
            ax.set_xlabel('guidance scale' + (' (zoom on 1–1.5)' if row else ''))
            metric_axis(ax, ev)
            panel_title(ax, ev, kind)
            ax.legend(fontsize=9, loc='upper left')
    fig.suptitle('Classifier-free guidance on the CFG-trained model, 100 Euler steps (mean over seeds, shading = '
                 'seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{kind}_guidance.png')
    plt.close(fig)



def fig_sharpening(plt, out, kind):
    """What whole-path guidance does to the samples: corner separation (synthetic), nearest-neighbour errors
    and spread (ACRONYM), at 100 steps."""
    x = np.arange(len(GUIDANCE))
    labels = ['1\n(none)' if g == 1.0 else f'{g:g}' for g in GUIDANCE]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    band(ax, x, guided('syn', kind, 'mode_accuracy', 'full', GUIDANCE), BLUE, 'generated samples')
    data_line(ax, data_value('syn', kind, 'mode_accuracy'), 'real data', where='right')
    ax.set_ylabel('corner accuracy\n(real data = target; higher over-separates)')
    ax.set_title(f"{EVALS['syn'][1]}: {KINDS[kind]['syn'][3]}", fontsize=11)
    ax.legend(fontsize=9.5, loc='upper left')
    ax = axes[1]
    band(ax, x, guided('acr', kind, 'recreation_err', 'full', GUIDANCE), BLUE,
         'recreation error: real grasp → nearest generated', marker='o')
    band(ax, x, guided('acr', kind, 'fidelity_err', 'full', GUIDANCE), ORANGE,
         'fidelity error: generated → nearest real', marker='s')
    ax.set_ylabel('mean nearest-neighbour distance (cm)\n(lower is better)')
    ax.set_title(f"{EVALS['acr'][1]}: {KINDS[kind]['acr'][3]}", fontsize=11)
    ax.legend(fontsize=9.5, loc='upper left')
    ax = axes[2]
    band(ax, x, guided('acr', kind, 'spread_ratio_trans', 'full', GUIDANCE), BLUE, 'position', marker='o')
    band(ax, x, guided('acr', kind, 'spread_ratio_rot', 'full', GUIDANCE), ORANGE, 'orientation', marker='s')
    data_line(ax, 1.0, 'real data', where='right')
    ax.set_ylabel('spread of generated / spread of real\n(below 1 = narrower than the data)')
    ax.set_title(f"{EVALS['acr'][1]}: {KINDS[kind]['acr'][3]}", fontsize=11)
    ax.legend(fontsize=9.5, loc='lower left')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('guidance scale, whole path (evenly spaced, not to scale)')
    fig.suptitle('What guidance does to the samples, 100 Euler steps (mean over seeds, shading = seed range)', y=1.0)
    fig.tight_layout()
    save(fig, out, f'{kind}_sharpening.png')
    plt.close(fig)

def _curve(ev, kind, tag):
    task, level, base, _ = KINDS[kind][ev]
    root = EVALS[ev][0] / task / 'training_curves'
    seeds = sorted(p.name for p in root.glob('seed_*'))
    epochs = list(range(10, 101, 10))
    vals = np.array([[float(next(r for r in _read(root / s / f'metrics_epoch{e}_{tag}.csv')
                                 if r['variant'] == base)[EVALS[ev][2]]) for e in epochs] for s in seeds])
    return np.array(epochs), vals


def fig_training(plt, out, kind):
    fig, axes = two_panels(plt)
    for ax, ev in zip(axes, ('syn', 'acr')):
        for tag, label, color, marker in (('steps3', '3 Euler steps', YELLOW, '^'),
                                          ('midpoint5', '5 midpoint steps', BLUE, 'o'),
                                          ('steps100', '100 Euler steps', INK2, 's')):
            x, vals = _curve(ev, kind, tag)
            band(ax, x, vals, color, label, marker=marker)
        data_line(ax, data_value(ev, kind), 'real data' if ev == 'acr' else 'one real draw', where='left')
        metric_axis(ax, ev)
        ax.set_xlabel('training epoch (' + ('100 batches of 128' if ev == 'syn' else '300 batches of 1024') + ')')
        panel_title(ax, ev, kind)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Training length: the recommended model sampled every 10 epochs (mean over seeds, shading = seed '
                 'range)', y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, f'{kind}_training.png')
    plt.close(fig)


SIZES = [('S', '256 wide, 6 blocks'), ('M', '384 wide, 8 blocks'), ('L', '512 wide, 12 blocks')]
PARAMS = {'continuous': {'S': 13.5, 'M': 37.6, 'L': 92.9}, 'discrete': {'S': 10.3, 'M': 30.5, 'L': 80.2}}


def fig_size(plt, out, kind):
    """ACRONYM backbone size: 1-NN accuracy at 100 steps against parameters, per held-out level."""
    levels = ([('heldout_grasps', 'held-out grasps of training objects', BLUE, 'o'),
               ('heldout_objects', 'unseen objects', ORANGE, 's'),
               ('heldout_categories', 'unseen categories', GREEN, '^')] if kind == 'continuous'
              else [('heldout_grasps', 'held-out grasps of training objects', BLUE, 'o')])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = [PARAMS[kind][s] for s, _ in SIZES]
    for level, label, color, marker in levels:
        vals = np.stack([per_seed('acr', kind, 'cond_OT_NOCFG', 'one_nna', 'metrics', ('size', s), level)
                         for s, _ in SIZES], 1)
        band(ax, x, vals, color, label, marker=marker)
        ax.axhline(data_value('acr', kind, level=level), color=color, ls=':', lw=1.2)
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n{p:.0f}M' for (s, _), p in zip(SIZES, x)])
    ax.minorticks_off()
    ax.set_xlabel('backbone (parameters)')
    ax.set_ylabel(EVALS['acr'][3])
    ax.legend(fontsize=9.5, loc='center right')
    ax.set_title('ACRONYM, 100 Euler steps; dotted = real data on that level\n(2 seeds, shading = seed range)',
                 fontsize=11)
    fig.tight_layout()
    save(fig, out, f'{kind}_size.png')
    plt.close(fig)


FIGURES = {'scorecard': fig_scorecard, 'pairing': fig_pairing, 'pathway': fig_pathway, 'solvers': fig_solvers,
           'guidance': fig_guidance, 'sharpening': fig_sharpening, 'training': fig_training, 'size': fig_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--kind', choices=('continuous', 'discrete', 'both'), default='both')
    parser.add_argument('--only', nargs='+', choices=tuple(FIGURES), default=None)
    parser.add_argument('--out', default='docs/combined')
    args = parser.parse_args()
    plt = setup_matplotlib()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for kind in (('continuous', 'discrete') if args.kind == 'both' else (args.kind,)):
        for name in args.only or FIGURES:
            FIGURES[name](plt, out, kind)


if __name__ == '__main__':
    main()
