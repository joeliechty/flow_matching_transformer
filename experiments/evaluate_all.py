"""Evaluate every checkpoint in one checkpoint directory and write a single metrics CSV.

Point --checkpoint_dir at one seed's directory (e.g. checkpoints/pose/seed_1/).
For each (task, ablation) combination we generate a fresh batch of samples and compute
the metrics in `utils.eval_utils`. Pose models are evaluated in twist-space; MNIST
models are scored by the small CNN oracle at `eval_assets/mnist_cnn.pt`.

Output: experiments/results/metrics.csv
"""
import argparse
import csv
import functools
import glob
import os
import re
import statistics
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.eval_utils import (
    goal_mode_poses_from_config,
    mnist_class_accuracy,
    mnist_class_marginal_kl,
    mode_balance_kl,
    nearest_mode_indices,
    path_straightness,
    per_mode_energy_distance,
    pose_bias_spread,
    pose_mode_accuracy,
    pose_mode_coverage_kl,
    pose_mode_distance,
    sample_goal_mixture,
    sample_goal_poses,
)
from utils.acronym import LEVELS, load_acronym_task
from utils.grasp_metrics import mean_over_objects, recreation_metrics
from utils.logging_utils import git_commit
from utils.mnist_classifier import load_classifier
from models.flow_transformer_base import SOLVER_EVALS
from utils.pose_task import ContinuousGoalTask, task_conditions
from utils.tf_utils import convert_twist_to_pose
from pose_gen_inference import (
    generate_from_distribution,
    load_model as load_pose_model,
)
from image_gen_inference import (
    build_obs_from_digit,
    generate_from_noise,
    load_model as load_image_model,
)


# Conditions (full and one-token) are read from each run's training config by
# `utils.pose_task.task_conditions`, so any pose task is evaluated the same way.

# Real goal samples for the distribution-level metrics. They have their own seeds, so the
# model-sampling noise (and with it every older metric) is unchanged.
REF_SEED, REF_PER_MODE = 1234, 1024
DATA_SEED = 4321  # independent draws scored as if from a perfect sampler (the 'data' row)
TOKEN_SEED = 5678  # eval-time noise on the obs tokens of tasks trained with token_sigma > 0


# Filename schema:
#   {prefix}{_OT|_NOOT}{_CFG|_NOCFG}_epoch_{N}.pt
# prefix ∈ {pose_flow_matching_model, cond_pose_flow_matching_model,
#           image_flow_matching_model, cond_image_flow_matching_model}
_CKPT_RE = re.compile(
    r"^(?P<prefix>cond_)?(?P<task>pose|image)_flow_matching_model"
    r"(?P<ot>_OT|_NOOT|_GOT|_C2OTFIX|_C2OT|_CLUSTER)(?P<cfg>_CFG|_NOCFG)_epoch_(?P<epoch>\d+)\.pt$"
)
# Checkpoint suffix -> noise-data pairing (`utils.train_utils.PAIRINGS`, pose_gen_trainer.py).
SUFFIX_PAIRING = {'_OT': 'ot', '_NOOT': 'independent', '_GOT': 'global', '_C2OT': 'c2ot',
                  '_C2OTFIX': 'c2ot_fixed', '_CLUSTER': 'cluster'}


def discover_checkpoints(ckpt_dir: Path, epoch=None):
    """Yield (path, meta_dict) for each ablation variant's checkpoint: the latest epoch,
    or exactly `epoch` when given (variants without that epoch are skipped)."""
    by_variant = {}
    for path in sorted(ckpt_dir.glob("*.pt")):
        m = _CKPT_RE.match(path.name)
        if not m:
            continue
        ckpt_epoch = int(m["epoch"])
        if epoch is not None and ckpt_epoch != epoch:
            continue
        key = (m["prefix"], m["task"], m["ot"], m["cfg"])
        if key not in by_variant or ckpt_epoch > by_variant[key][1]:
            by_variant[key] = (path, ckpt_epoch, m.groupdict())
    for (path, epoch, meta) in by_variant.values():
        meta["epoch"] = epoch
        meta["path"] = path
        yield meta


def _config_path_for(ckpt_path: Path):
    # `..._OT_CFG_epoch_100.pt` -> `..._OT_CFG_training_config.yaml`
    stem = ckpt_path.name.split("_epoch_")[0]
    return ckpt_path.parent / f"{stem}_training_config.yaml"


def effective_cfg_scale(meta, cfg_scale):
    """Guidance scale to sample a variant at. NOCFG variants weren't trained with
    unconditional dropout, so guiding them would test an untrained null branch —
    sample them unguided (1.0) instead."""
    return cfg_scale if meta["cfg"] == "_CFG" else 1.0


def sampler_columns(num_steps, method, cfg_interval):
    """The sampler's CSV columns: solver, network evaluations per sample (before CFG's second
    pass) and the guidance interval ('full' when guidance, if any, runs at every step)."""
    interval = 'full' if cfg_interval is None else f"{cfg_interval[0]:g}-{cfg_interval[1]:g}"
    return {'method': method, 'nfe': num_steps * SOLVER_EVALS[method], 'cfg_interval': interval}


def variant_name(meta):
    """Task-agnostic variant label, e.g. 'cond_OT_CFG' or 'uncond_NOOT'. Unconditional
    labels omit CFG, which has no effect without conditioning."""
    if meta["prefix"] == "cond_":
        return f"cond{meta['ot']}{meta['cfg']}"
    return f"uncond{meta['ot']}"


def task_conditions_for(config):
    """(full, partial) conditions from a training config; ([], []) for unconditional runs."""
    actions = config.training.get('action_dist_params')
    if actions is None:
        return [], []
    return task_conditions(OmegaConf.to_container(actions, resolve=True))


def _token_sigma(config):
    """Std of the noise on the obs tokens during training (0 for clean discrete tokens)."""
    actions = config.training.get('action_dist_params')
    if actions is None:
        return 0.0
    return max(float(v) for mode in actions['sigma'] for token in mode for v in token)


def _eval_obs(cond, n, device, token_sigma, index):
    """Obs tokens for `n` samples of `cond`. With token noise in training, each sample's tokens
    get fresh noise of the same std from a generator of their own (seeded per condition), so
    the sampling noise is unchanged and every variant sees the same tokens."""
    obs = cond.obs(n, device)
    if token_sigma > 0:
        gen = torch.Generator().manual_seed(TOKEN_SEED + index)
        obs = obs + token_sigma * torch.randn(obs.shape, generator=gen).to(device)
    return obs


def _condition_targets(samples, condition, mode_poses):
    """Each sample's target mode: the condition's only valid mode, or else the nearest of its
    valid modes (a multimodal condition is satisfied by landing on any of them)."""
    if len(condition.valid_modes) == 1:
        return torch.full((samples.shape[0],), condition.valid_modes[0], dtype=torch.long)
    valid = torch.tensor(condition.valid_modes, device=mode_poses.device)
    return valid[nearest_mode_indices(samples, mode_poses[valid])].cpu()


def _condition_balance(conditions, targets_per_condition):
    """Mean imbalance KL over the multimodal conditions, or None if there are none."""
    kls = [mode_balance_kl(t, c.valid_modes)
           for c, t in zip(conditions, targets_per_condition) if len(c.valid_modes) > 1]
    return statistics.fmean(kls) if kls else None


def _distribution_metrics(samples, assigned, mode_poses, ref, ref_idx):
    out = pose_bias_spread(samples, assigned, mode_poses, ref, ref_idx)
    out['energy_distance'] = per_mode_energy_distance(samples, assigned, ref, ref_idx)
    return out


def _partial_metrics(conditions, sample_sets, mode_poses, ref, ref_idx):
    """Metrics for one-token conditioning; `sample_sets[i]` are samples for `conditions[i]`.

    validity = share of samples nearest a valid mode; balance KL = how unevenly they split
    across the valid modes (0 = even); energy distance = within-mode quality.
    """
    if not conditions:
        return dict.fromkeys(('partial_validity', 'partial_balance_kl', 'partial_energy_distance'))
    validity, balance, energy = [], [], []
    for cond, samples in zip(conditions, sample_sets):
        valid = cond.valid_modes
        nearest = nearest_mode_indices(samples, mode_poses)
        in_valid = torch.isin(nearest, torch.tensor(valid, device=nearest.device))
        validity.append(in_valid.float().mean().item())
        balance.append(mode_balance_kl(nearest, valid))
        energy.append(per_mode_energy_distance(samples[in_valid], nearest[in_valid], ref, ref_idx))
    energy = [e for e in energy if e is not None]
    return {
        'partial_validity': statistics.fmean(validity),
        'partial_balance_kl': statistics.fmean(balance),
        'partial_energy_distance': statistics.fmean(energy) if energy else None,
    }


def evaluate_pose(meta, device, num_samples=256, num_steps=100, cfg_scale=3.0, eval_seed=0,
                  method='euler', cfg_interval=None):
    conditional = meta["prefix"] == "cond_"
    cfg_scale = effective_cfg_scale(meta, cfg_scale)
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model, _ = load_pose_model(
        str(meta["path"]), device=device, model_config=model_config, conditional=conditional,
    )
    # Same start noise for every variant and sweep point, so differences come from the model.
    torch.manual_seed(eval_seed)

    goal_dist = OmegaConf.to_container(config.training.goal_dist_params, resolve=True)
    start_dist = OmegaConf.to_container(config.training.start_dist_params, resolve=True)
    start_dist_flat = {'mu': start_dist['mu'][0], 'sigma': start_dist['sigma'][0]}
    mode_poses = goal_mode_poses_from_config(goal_dist, device=device)  # [K, 7]
    ref, ref_idx = sample_goal_poses(goal_dist, REF_PER_MODE, REF_SEED, device)

    # CFG has no effect without conditioning, so leave it blank for unconditional rows.
    row = {
        'task': 'pose', 'variant': variant_name(meta), 'conditional': conditional,
        'ot': meta['ot'] == '_OT', 'pairing': SUFFIX_PAIRING[meta['ot']],
        'cfg': meta['cfg'] == '_CFG' if conditional else None,
        'cfg_scale_at_inference': cfg_scale if conditional else None,
        'num_steps': num_steps, 'num_samples': num_samples, 'epoch': meta['epoch'],
        **sampler_columns(num_steps, method, cfg_interval),
        'mode_accuracy': None, 'mode_distance': None, 'mode_coverage_kl': None,
        'class_accuracy': None, 'class_marginal_kl': None,
    }

    if conditional:
        full, partial_conditions = task_conditions_for(config)
        token_sigma = _token_sigma(config)
        per_cond = num_samples // len(full)
        all_traj, all_targets = [], []
        for i, cond in enumerate(full):
            _, traj = generate_from_distribution(
                model, start_dist_flat, batch_size=per_cond,
                obs=_eval_obs(cond, per_cond, device, token_sigma, i),
                num_steps=num_steps, return_trajectory=True,
                cfg_scale=cfg_scale, device=device, method=method, cfg_interval=cfg_interval,
            )
            all_traj.append(traj)
            all_targets.append(_condition_targets(traj[:, -1], cond, mode_poses))
        trajectory = torch.cat(all_traj, dim=0)
        samples = trajectory[:, -1]
        targets = torch.cat(all_targets, dim=0)

        row['mode_accuracy'] = pose_mode_accuracy(samples, targets, mode_poses)
        row['mode_distance'] = pose_mode_distance(samples, mode_poses, targets)
        kl, counts = pose_mode_coverage_kl(samples, mode_poses)
        row['mode_coverage_kl'] = kl
        row['per_mode_counts'] = dict(counts)
        balance = _condition_balance(full, all_targets)
        if balance is not None:  # only tasks with multimodal conditions get this column
            row['condition_balance_kl'] = balance
        assigned = targets.to(samples.device)
    else:
        _, trajectory = generate_from_distribution(
            model, start_dist_flat, batch_size=num_samples, obs=None,
            num_steps=num_steps, return_trajectory=True, device=device, method=method,
        )
        samples = trajectory[:, -1]
        row['mode_distance'] = pose_mode_distance(samples, mode_poses)
        kl, counts = pose_mode_coverage_kl(samples, mode_poses)
        row['mode_coverage_kl'] = kl
        row['per_mode_counts'] = dict(counts)
        assigned = nearest_mode_indices(samples, mode_poses)

    row.update(_distribution_metrics(samples, assigned, mode_poses, ref, ref_idx))
    row['path_straightness'], row['transport_cost'] = path_straightness(trajectory)

    if conditional:
        # Drawn after the full-condition samples, so those keep their noise.
        partial = []
        for i, cond in enumerate(partial_conditions):
            _, partial_samples = generate_from_distribution(
                model, start_dist_flat, batch_size=per_cond,
                obs=_eval_obs(cond, per_cond, device, token_sigma, len(full) + i),
                obs_mask=cond.obs_mask(per_cond, device),
                num_steps=num_steps, return_trajectory=False, cfg_scale=cfg_scale, device=device,
                method=method, cfg_interval=cfg_interval,
            )
            partial.append(partial_samples)
        row.update(_partial_metrics(partial_conditions, partial, mode_poses, ref, ref_idx))

    row['num_samples'] = samples.shape[0]  # per-mode split can round down
    return row


def _continuous_task(config):
    return ContinuousGoalTask(OmegaConf.to_container(config.training.task_spec, resolve=True))


def _continuous_reference(task, c, index, device):
    """Real goal poses for one condition, REF_PER_MODE per mode, and their mode indices."""
    gen = torch.Generator().manual_seed(REF_SEED + index)
    mu = task.mode_twists(c.reshape(1, 2).cpu())[0]                      # [K, 6]
    twists = mu[:, None] + task.sigma * torch.randn(task.num_modes, REF_PER_MODE, 6, generator=gen)
    poses = convert_twist_to_pose(twists.reshape(-1, 6), dt=1.0, return_representation='quat')
    return poses.to(device), torch.arange(task.num_modes).repeat_interleave(REF_PER_MODE).to(device)


def _continuous_metrics(task, conds, sample_sets, device):
    """Metrics of a continuous-condition task, each computed per test condition against that
    condition's own modes and real samples, then averaged over conditions. Same column names
    as the discrete evaluator; condition_balance_kl is the split between the orientations."""
    per_cond = []
    for i, (c, samples) in enumerate(zip(conds, sample_sets)):
        modes = task.mode_poses(c.reshape(1, 2))[0].to(device)           # [K, 7]
        ref, ref_idx = _continuous_reference(task, c, i, device)
        assigned = nearest_mode_indices(samples, modes)
        m = pose_bias_spread(samples, assigned, modes, ref, ref_idx)
        m['energy_distance'] = per_mode_energy_distance(samples, assigned, ref, ref_idx)
        m['condition_balance_kl'] = mode_balance_kl(assigned, list(range(task.num_modes)))
        m['mode_distance'] = pose_mode_distance(samples, modes, assigned)
        per_cond.append(m)
    return {k: statistics.fmean(m[k] for m in per_cond if m[k] is not None)
            for k in per_cond[0]}


def evaluate_pose_continuous(meta, device, num_samples=1024, num_steps=100, cfg_scale=3.0,
                             eval_seed=0, method='euler', cfg_interval=None):
    """Pose metrics for a continuous-condition task: num_samples spread evenly over the task's
    fixed test conditions."""
    cfg_scale = effective_cfg_scale(meta, cfg_scale)
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model, _ = load_pose_model(str(meta["path"]), device=device, model_config=model_config,
                               conditional=True)
    torch.manual_seed(eval_seed)
    task = _continuous_task(config)
    start = OmegaConf.to_container(config.training.start_dist_params, resolve=True)
    start_dist_flat = {'mu': start['mu'][0], 'sigma': start['sigma'][0]}
    conds = task.test_conditions(device)
    per_cond = num_samples // len(conds)

    trajectories = []
    for c in conds:
        _, traj = generate_from_distribution(
            model, start_dist_flat, batch_size=per_cond,
            obs=task.obs(c.reshape(1, 2)).repeat(per_cond, 1, 1), num_steps=num_steps,
            return_trajectory=True, cfg_scale=cfg_scale, device=device, method=method,
            cfg_interval=cfg_interval)
        trajectories.append(traj)
    row = {
        'task': 'pose', 'variant': variant_name(meta), 'conditional': True,
        'ot': meta['ot'] == '_OT', 'pairing': SUFFIX_PAIRING[meta['ot']],
        'cfg': meta['cfg'] == '_CFG', 'cfg_scale_at_inference': cfg_scale,
        'num_steps': num_steps, 'num_samples': per_cond * len(conds), 'epoch': meta['epoch'],
        **sampler_columns(num_steps, method, cfg_interval),
    }
    row.update(_continuous_metrics(task, conds, [t[:, -1] for t in trajectories], device))
    row['path_straightness'], row['transport_cost'] = path_straightness(torch.cat(trajectories))
    return row


def evaluate_pose_continuous_reference(config, epoch, device, num_samples=1024):
    """Real goal samples scored like the models: the floor of every continuous-task metric."""
    task = _continuous_task(config)
    conds = task.test_conditions()
    per_cond = num_samples // len(conds)
    sample_sets = []
    for i, c in enumerate(conds):
        gen = torch.Generator().manual_seed(DATA_SEED + i)
        twists, _ = task.sample_goals(c.reshape(1, 2).repeat(per_cond, 1), generator=gen)
        sample_sets.append(convert_twist_to_pose(twists, dt=1.0, return_representation='quat').to(device))
    row = {'task': 'pose', 'variant': 'data', 'epoch': epoch, 'num_samples': per_cond * len(conds),
           'git_commit': _commit()}
    row.update(_continuous_metrics(task, conds, sample_sets, device))
    return row


def evaluate_pose_reference(config_path, epoch, device, num_samples=256):
    """Score real goal samples with the pose metrics: what a perfect sampler gets."""
    config = OmegaConf.load(config_path)
    if config.training.get('task_type') == 'continuous':
        return evaluate_pose_continuous_reference(config, epoch, device, num_samples)
    goal_dist = OmegaConf.to_container(config.training.goal_dist_params, resolve=True)
    mode_poses = goal_mode_poses_from_config(goal_dist, device=device)
    K = mode_poses.shape[0]
    ref, ref_idx = sample_goal_poses(goal_dist, REF_PER_MODE, REF_SEED, device)
    full, partial_conditions = task_conditions_for(config)
    draw_targets = None
    if all(len(c.valid_modes) == 1 for c in full):
        # One condition per mode (or unconditional): equal draws from every mode.
        samples, targets = sample_goal_poses(goal_dist, num_samples // K, DATA_SEED, device)
    else:
        # Multimodal conditions: each condition's valid modes, picked at random. Targets are
        # assigned exactly as for the models (nearest valid mode), not from the true labels.
        draw_poses = [sample_goal_mixture(goal_dist, num_samples // len(full), DATA_SEED + 200 + i,
                                          modes=c.valid_modes, device=device)[0]
                      for i, c in enumerate(full)]
        draw_targets = [_condition_targets(d, c, mode_poses) for d, c in zip(draw_poses, full)]
        samples = torch.cat(draw_poses)
        targets = torch.cat(draw_targets).to(device)
    # Coverage of an unconditional perfect sampler: modes drawn at random, not 64 each.
    mixture, _ = sample_goal_mixture(goal_dist, num_samples, DATA_SEED + 100, device=device)

    kl, counts = pose_mode_coverage_kl(mixture, mode_poses)
    row = {
        'task': 'pose', 'variant': 'data', 'epoch': epoch, 'num_samples': samples.shape[0],
        'mode_accuracy': pose_mode_accuracy(samples, targets, mode_poses),
        'mode_distance': pose_mode_distance(samples, mode_poses, targets),
        'mode_coverage_kl': kl, 'per_mode_counts': dict(counts), 'git_commit': _commit(),
    }
    row.update(_distribution_metrics(samples, targets, mode_poses, ref, ref_idx))
    if draw_targets is not None:
        row['condition_balance_kl'] = _condition_balance(full, draw_targets)
    # One-token conditions: a perfect sampler picks each valid mode with equal probability.
    per_cond = num_samples // max(len(full), 1)
    partial = [sample_goal_mixture(goal_dist, per_cond, DATA_SEED + 1 + i,
                                   modes=c.valid_modes, device=device)[0]
               for i, c in enumerate(partial_conditions)]
    row.update(_partial_metrics(partial_conditions, partial, mode_poses, ref, ref_idx))
    return row


def evaluate_image(meta, device, classifier, num_samples=256, num_steps=100, cfg_scale=3.0,
                   eval_seed=0, method='euler', cfg_interval=None):
    conditional = meta["prefix"] == "cond_"
    cfg_scale = effective_cfg_scale(meta, cfg_scale)
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model, _ = load_image_model(
        str(meta["path"]), device=device, model_config=model_config, conditional=conditional,
    )
    torch.manual_seed(eval_seed)

    row = {
        'task': 'mnist', 'variant': variant_name(meta), 'conditional': conditional,
        'ot': meta['ot'] == '_OT', 'cfg': meta['cfg'] == '_CFG' if conditional else None,
        'cfg_scale_at_inference': cfg_scale if conditional else None,
        'num_steps': num_steps, 'num_samples': num_samples, 'epoch': meta['epoch'],
        **sampler_columns(num_steps, method, cfg_interval),
        'mode_accuracy': None, 'mode_distance': None, 'mode_coverage_kl': None,
        'class_accuracy': None, 'class_marginal_kl': None,
    }

    if conditional:
        per_class = max(1, num_samples // 10)
        all_imgs, all_labels = [], []
        for digit in range(10):
            obs = build_obs_from_digit(digit, per_class, device)
            imgs = generate_from_noise(
                model, per_class, obs=obs, num_steps=num_steps,
                return_trajectory=False, cfg_scale=cfg_scale, device=device,
                method=method, cfg_interval=cfg_interval,
            )
            all_imgs.append(imgs)
            all_labels.append(torch.full((per_class,), digit, dtype=torch.long))
        imgs = torch.cat(all_imgs, dim=0)
        labels = torch.cat(all_labels, dim=0)

        row['class_accuracy'] = mnist_class_accuracy(imgs, labels, classifier)
        kl, _ = mnist_class_marginal_kl(imgs, classifier)
        row['class_marginal_kl'] = kl
    else:
        imgs = generate_from_noise(
            model, num_samples, obs=None, num_steps=num_steps,
            return_trajectory=False, device=device, method=method,
        )
        kl, _ = mnist_class_marginal_kl(imgs, classifier)
        row['class_marginal_kl'] = kl

    row['num_samples'] = imgs.shape[0]  # per-class split can round down
    return row


@functools.lru_cache(maxsize=None)
def _acronym_task(config, device):
    return load_acronym_task(OmegaConf.to_container(config.training.task_spec, resolve=True),
                             device)['task']


def _check_level(task, level):
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level!r}. Expected one of {tuple(LEVELS)}.")
    if task.condition == 'ids' and level != 'heldout_grasps':
        raise ValueError("an id-conditioned model knows only its training objects: "
                         "evaluate it on --level heldout_grasps")


def _acronym_metrics(task, objects, sample_sets, level):
    """Pose-recreation metrics (`utils.grasp_metrics`) of each object's samples [n, 7] (pos_scale
    units) against its reference grasps, averaged over objects."""
    per_object = []
    for obj, samples in zip(objects, sample_sets):
        ref, _ = task.reference_and_floor(obj, level, samples.shape[0])
        per_object.append(recreation_metrics(task.to_metres(samples), ref, task.tcp_offset))
    return mean_over_objects(per_object)


def evaluate_pose_acronym(meta, device, num_samples=256, num_steps=100, cfg_scale=3.0,
                          eval_seed=0, method='euler', cfg_interval=None,
                          level='heldout_grasps', objects_per_batch=16):
    """Pose-recreation metrics of an ACRONYM model on one held-out level: num_samples grasps
    of each of the level's fixed evaluation objects, each object scored against its own
    held-out real grasps."""
    cfg_scale = effective_cfg_scale(meta, cfg_scale)
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    conditional = meta['prefix'] == 'cond_'
    model, _ = load_pose_model(str(meta["path"]), device=device, model_config=model_config,
                               conditional=conditional)
    task = _acronym_task(config, device)
    _check_level(task, level)
    objects = task.eval_objects(level)
    torch.manual_seed(eval_seed)

    finals, trajectories = [], []
    for i in range(0, len(objects), objects_per_batch):
        chunk = torch.tensor(objects[i:i + objects_per_batch], device=device)
        n = len(chunk) * num_samples
        x0 = convert_twist_to_pose(task.sample_start(n, device=device), dt=1.0,
                                   return_representation='quat')
        if conditional:
            traj = model.inference(x0, task.obs(chunk.repeat_interleave(num_samples)),
                                   num_steps=num_steps, return_trajectory=True,
                                   cfg_scale=cfg_scale, method=method, cfg_interval=cfg_interval)
        else:
            traj = model.inference(x0, num_steps=num_steps, return_trajectory=True, method=method)
        finals += list(traj[:, -1].split(num_samples))
        trajectories.append(traj)
    row = {
        'task': 'pose', 'variant': variant_name(meta), 'conditional': conditional,
        'ot': meta['ot'] == '_OT', 'pairing': SUFFIX_PAIRING[meta['ot']],
        'cfg': meta['cfg'] == '_CFG', 'cfg_scale_at_inference': cfg_scale,
        'num_steps': num_steps, 'num_samples': num_samples, 'num_objects': len(objects),
        'level': level, 'epoch': meta['epoch'], **sampler_columns(num_steps, method, cfg_interval),
    }
    row.update(_acronym_metrics(task, objects, finals, level))
    row['path_straightness'], row['transport_cost'] = path_straightness(torch.cat(trajectories))
    return row


def evaluate_pose_acronym_reference(config, epoch, device, num_samples=256, level='heldout_grasps'):
    """Real grasps scored like the models (each object's floor set against its reference
    set): the floor of every ACRONYM metric."""
    task = _acronym_task(config, device)
    _check_level(task, level)
    objects = task.eval_objects(level)
    floors = [task.reference_and_floor(obj, level, num_samples)[1] for obj in objects]
    # the floor sets are in metres; the metrics take pos_scale units
    floors = [torch.cat([f[:, :3] / task.pos_scale, f[:, 3:]], -1) for f in floors]
    row = {'task': 'pose', 'variant': 'data', 'epoch': epoch, 'num_samples': num_samples,
           'num_objects': len(objects), 'level': level, 'git_commit': _commit()}
    row.update(_acronym_metrics(task, objects, floors, level))
    return row


def _commit():
    return git_commit()


def evaluate(meta, device, classifier, **kwargs):
    """Run the task's evaluator. Returns None for MNIST when no classifier is loaded."""
    level = kwargs.pop('level', 'heldout_grasps')
    if meta['task'] == 'pose':
        config = OmegaConf.load(_config_path_for(meta['path']))
        task_type = config.training.get('task_type')
        if task_type == 'acronym':
            row = evaluate_pose_acronym(meta, device, level=level, **kwargs)
        elif task_type == 'continuous':
            row = evaluate_pose_continuous(meta, device, **kwargs)
        else:
            row = evaluate_pose(meta, device, **kwargs)
    elif classifier is None:
        return None
    else:
        row = evaluate_image(meta, device, classifier, **kwargs)
    row['git_commit'] = _commit()
    return row


def load_classifier_if_needed(metas, classifier_path, device):
    """Load the MNIST oracle only when an MNIST checkpoint is among `metas`."""
    if not any(m['task'] == 'image' for m in metas):
        return None
    if not Path(classifier_path).exists():
        print(f"WARNING: no classifier at {classifier_path} — MNIST metrics will be skipped. "
              f"Train one with: python -m utils.mnist_classifier")
        return None
    print(f"Loaded MNIST classifier from {classifier_path}")
    return load_classifier(classifier_path, device=device)


def write_csv(rows, output, fieldnames=None):
    if not rows:
        print("No rows produced.")
        return
    os.makedirs(os.path.dirname(str(output)) or '.', exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Evaluate the checkpoints saved at this epoch (default: latest)')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--method', type=str, default='euler', choices=tuple(SOLVER_EVALS),
                        help='ODE solver')
    parser.add_argument('--cfg_interval', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='Apply guidance only at flow times LO <= t < HI')
    parser.add_argument('--eval_seed', type=int, default=0,
                        help='Seed for the sampling noise, shared across all variants')
    parser.add_argument('--level', type=str, default='heldout_grasps', choices=tuple(LEVELS),
                        help='ACRONYM tasks: the held-out level to score (--num_samples is then '
                             'per object)')
    parser.add_argument('--output', type=str, default='experiments/results/metrics.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"No checkpoint directory at {ckpt_dir}")

    metas = list(discover_checkpoints(ckpt_dir, args.epoch))
    classifier = load_classifier_if_needed(metas, args.classifier_path, device)

    rows = []
    for meta in metas:
        tag = f"{meta['task']} {variant_name(meta)} (epoch {meta['epoch']})"
        print(f"\n=== Evaluating {tag} ===")
        try:
            row = evaluate(meta, device, classifier,
                           num_samples=args.num_samples, num_steps=args.num_steps,
                           cfg_scale=args.cfg_scale, eval_seed=args.eval_seed,
                           method=args.method, cfg_interval=args.cfg_interval, level=args.level)
        except Exception as e:
            print(f"ERROR evaluating {tag}: {e}")
            continue
        if row is None:
            print("Skipping MNIST eval — no classifier available.")
            continue
        rows.append(row)
        print({k: v for k, v in row.items() if v is not None})

    pose_metas = [m for m in metas if m['task'] == 'pose']
    if pose_metas:
        print("\n=== Scoring real goal samples (the 'data' reference row) ===")
        # A conditional run's config also carries the conditions, for the one-token metrics.
        source = next((m for m in pose_metas if m['prefix'] == 'cond_'), pose_metas[0])
        config = OmegaConf.load(_config_path_for(source['path']))
        epoch = max(m['epoch'] for m in pose_metas)
        if config.training.get('task_type') == 'acronym':
            rows.append(evaluate_pose_acronym_reference(config, epoch, device,
                                                        args.num_samples, args.level))
        else:
            rows.append(evaluate_pose_reference(_config_path_for(source['path']), epoch, device,
                                                num_samples=args.num_samples))

    write_csv(rows, args.output)


if __name__ == '__main__':
    main()
