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
    ACTION_TO_MODE,
    actions_to_mode_indices,
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
    sample_goal_poses,
)
from utils.logging_utils import git_commit
from utils.mnist_classifier import load_classifier
from pose_gen_inference import (
    build_obs_from_actions,
    generate_from_distribution,
    load_model as load_pose_model,
)
from image_gen_inference import (
    build_obs_from_digit,
    generate_from_noise,
    load_model as load_image_model,
)


# Layout of the 4 action_pairs we use to evaluate conditional pose models —
# one pair per mode, in the same index order as MODE_NAMES in eval_utils.
ACTION_PAIRS = [
    ("top", "right"),    # mode 0
    ("bottom", "right"), # mode 1
    ("top", "left"),     # mode 2
    ("bottom", "left"),  # mode 3
]

# One-token conditions: (action pair, index of the token replaced by the null token).
# The visible token alone matches two modes, so the correct output is bimodal.
PARTIAL_CONDITIONS = [
    (("top", "right"), 1),     # "top" only    -> top-right, top-left
    (("bottom", "right"), 1),  # "bottom" only -> bottom-right, bottom-left
    (("top", "right"), 0),     # "right" only  -> top-right, bottom-right
    (("top", "left"), 0),      # "left" only   -> top-left, bottom-left
]

# Real goal samples for the distribution-level metrics. They have their own seeds, so the
# model-sampling noise (and with it every older metric) is unchanged.
REF_SEED, REF_PER_MODE = 1234, 1024
DATA_SEED = 4321  # independent draws scored as if from a perfect sampler (the 'data' row)


# Filename schema:
#   {prefix}{_OT|_NOOT}{_CFG|_NOCFG}_epoch_{N}.pt
# prefix ∈ {pose_flow_matching_model, cond_pose_flow_matching_model,
#           image_flow_matching_model, cond_image_flow_matching_model}
_CKPT_RE = re.compile(
    r"^(?P<prefix>cond_)?(?P<task>pose|image)_flow_matching_model"
    r"(?P<ot>_OT|_NOOT)(?P<cfg>_CFG|_NOCFG)_epoch_(?P<epoch>\d+)\.pt$"
)


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


def variant_name(meta):
    """Task-agnostic variant label, e.g. 'cond_OT_CFG' or 'uncond_NOOT'. Unconditional
    labels omit CFG, which has no effect without conditioning."""
    if meta["prefix"] == "cond_":
        return f"cond{meta['ot']}{meta['cfg']}"
    return f"uncond{meta['ot']}"


def _partial_valid_modes(action_pair, masked):
    """Modes consistent with the visible token of a one-token condition."""
    visible = 1 - masked
    return sorted(m for pair, m in ACTION_TO_MODE.items() if pair[visible] == action_pair[visible])


def _distribution_metrics(samples, assigned, mode_poses, ref, ref_idx):
    out = pose_bias_spread(samples, assigned, mode_poses, ref, ref_idx)
    out['energy_distance'] = per_mode_energy_distance(samples, assigned, ref, ref_idx)
    return out


def _partial_metrics(sample_sets, mode_poses, ref, ref_idx):
    """Metrics for one-token conditioning; `sample_sets[i]` are samples for PARTIAL_CONDITIONS[i].

    validity = share of samples nearest one of the two valid modes; balance KL = how unevenly
    those split between the two (0 = 50/50); energy distance = within-mode quality.
    """
    validity, balance, energy = [], [], []
    for (pair, masked), samples in zip(PARTIAL_CONDITIONS, sample_sets):
        valid = _partial_valid_modes(pair, masked)
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


def evaluate_pose(meta, device, num_samples=256, num_steps=100, cfg_scale=3.0, eval_seed=0):
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
        'ot': meta['ot'] == '_OT', 'cfg': meta['cfg'] == '_CFG' if conditional else None,
        'cfg_scale_at_inference': cfg_scale if conditional else None,
        'num_steps': num_steps, 'num_samples': num_samples, 'epoch': meta['epoch'],
        'mode_accuracy': None, 'mode_distance': None, 'mode_coverage_kl': None,
        'class_accuracy': None, 'class_marginal_kl': None,
    }

    if conditional:
        per_mode = num_samples // 4
        all_traj, all_targets = [], []
        for mode_idx, action_pair in enumerate(ACTION_PAIRS):
            obs = build_obs_from_actions(list(action_pair), per_mode, device)
            _, traj = generate_from_distribution(
                model, start_dist_flat, batch_size=per_mode, obs=obs,
                num_steps=num_steps, return_trajectory=True,
                cfg_scale=cfg_scale, device=device,
            )
            all_traj.append(traj)
            all_targets.append(torch.full((per_mode,), mode_idx, dtype=torch.long))
        trajectory = torch.cat(all_traj, dim=0)
        samples = trajectory[:, -1]
        targets = torch.cat(all_targets, dim=0)

        row['mode_accuracy'] = pose_mode_accuracy(samples, targets, mode_poses)
        row['mode_distance'] = pose_mode_distance(samples, mode_poses, targets)
        kl, counts = pose_mode_coverage_kl(samples, mode_poses)
        row['mode_coverage_kl'] = kl
        row['per_mode_counts'] = dict(counts)
        assigned = targets.to(samples.device)
    else:
        _, trajectory = generate_from_distribution(
            model, start_dist_flat, batch_size=num_samples, obs=None,
            num_steps=num_steps, return_trajectory=True, device=device,
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
        for action_pair, masked in PARTIAL_CONDITIONS:
            obs = build_obs_from_actions(list(action_pair), per_mode, device)
            obs_mask = torch.zeros(per_mode, obs.shape[1], dtype=torch.bool, device=device)
            obs_mask[:, masked] = True
            _, partial_samples = generate_from_distribution(
                model, start_dist_flat, batch_size=per_mode, obs=obs, obs_mask=obs_mask,
                num_steps=num_steps, return_trajectory=False, cfg_scale=cfg_scale, device=device,
            )
            partial.append(partial_samples)
        row.update(_partial_metrics(partial, mode_poses, ref, ref_idx))

    row['num_samples'] = samples.shape[0]  # per-mode split can round down
    return row


def evaluate_pose_reference(config_path, epoch, device, num_samples=256):
    """Score real goal samples with the pose metrics: what a perfect sampler gets."""
    config = OmegaConf.load(config_path)
    goal_dist = OmegaConf.to_container(config.training.goal_dist_params, resolve=True)
    mode_poses = goal_mode_poses_from_config(goal_dist, device=device)
    K = mode_poses.shape[0]
    ref, ref_idx = sample_goal_poses(goal_dist, REF_PER_MODE, REF_SEED, device)
    samples, targets = sample_goal_poses(goal_dist, num_samples // K, DATA_SEED, device)

    kl, counts = pose_mode_coverage_kl(samples, mode_poses)
    row = {
        'task': 'pose', 'variant': 'data', 'epoch': epoch, 'num_samples': samples.shape[0],
        'mode_accuracy': pose_mode_accuracy(samples, targets, mode_poses),
        'mode_distance': pose_mode_distance(samples, mode_poses, targets),
        'mode_coverage_kl': kl, 'per_mode_counts': dict(counts), 'git_commit': _commit(),
    }
    row.update(_distribution_metrics(samples, targets, mode_poses, ref, ref_idx))
    # One-token conditions: a perfect sampler splits evenly between the two valid modes.
    partial = []
    for i, (pair, masked) in enumerate(PARTIAL_CONDITIONS):
        valid = torch.tensor(_partial_valid_modes(pair, masked), device=device)
        pool, pool_idx = sample_goal_poses(goal_dist, num_samples // K // len(valid),
                                           DATA_SEED + 1 + i, device)
        partial.append(pool[torch.isin(pool_idx, valid)])
    row.update(_partial_metrics(partial, mode_poses, ref, ref_idx))
    return row


def evaluate_image(meta, device, classifier, num_samples=256, num_steps=100, cfg_scale=3.0,
                   eval_seed=0):
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
            return_trajectory=False, device=device,
        )
        kl, _ = mnist_class_marginal_kl(imgs, classifier)
        row['class_marginal_kl'] = kl

    row['num_samples'] = imgs.shape[0]  # per-class split can round down
    return row


@functools.lru_cache(maxsize=None)
def _commit():
    return git_commit()


def evaluate(meta, device, classifier, **kwargs):
    """Run the task's evaluator. Returns None for MNIST when no classifier is loaded."""
    if meta['task'] == 'pose':
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
    parser.add_argument('--eval_seed', type=int, default=0,
                        help='Seed for the sampling noise, shared across all variants')
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
                           cfg_scale=args.cfg_scale, eval_seed=args.eval_seed)
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
        rows.append(evaluate_pose_reference(_config_path_for(pose_metas[0]['path']),
                                            max(m['epoch'] for m in pose_metas), device,
                                            num_samples=args.num_samples))

    write_csv(rows, args.output)


if __name__ == '__main__':
    main()
