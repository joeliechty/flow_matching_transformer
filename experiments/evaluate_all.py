"""Evaluate every checkpoint in `checkpoints/` and write a single metrics CSV.

For each (task, ablation) combination we generate a fresh batch of samples and compute
the metrics in `utils.eval_utils`. Pose models are evaluated in twist-space; MNIST
models are scored by the small CNN oracle at `eval_assets/mnist_cnn.pt`.

Output: experiments/results/metrics.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.eval_utils import (
    actions_to_mode_indices,
    goal_mode_poses_from_config,
    mnist_class_accuracy,
    mnist_class_marginal_kl,
    pose_mode_accuracy,
    pose_mode_coverage_kl,
)
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


# Filename schema:
#   {prefix}{_OT|_NOOT}{_CFG|_NOCFG}_epoch_{N}.pt
# prefix ∈ {pose_flow_matching_model, cond_pose_flow_matching_model,
#           image_flow_matching_model, cond_image_flow_matching_model}
_CKPT_RE = re.compile(
    r"^(?P<prefix>cond_)?(?P<task>pose|image)_flow_matching_model"
    r"(?P<ot>_OT|_NOOT)(?P<cfg>_CFG|_NOCFG)_epoch_(?P<epoch>\d+)\.pt$"
)


def discover_checkpoints(ckpt_dir: Path):
    """Yield (path, meta_dict) for the latest-epoch checkpoint of each ablation variant."""
    by_variant = {}
    for path in sorted(ckpt_dir.glob("*.pt")):
        m = _CKPT_RE.match(path.name)
        if not m:
            continue
        key = (m["prefix"], m["task"], m["ot"], m["cfg"])
        epoch = int(m["epoch"])
        if key not in by_variant or epoch > by_variant[key][1]:
            by_variant[key] = (path, epoch, m.groupdict())
    for (path, epoch, meta) in by_variant.values():
        meta["epoch"] = epoch
        meta["path"] = path
        yield meta


def _config_path_for(ckpt_path: Path):
    # `..._OT_CFG_epoch_100.pt` -> `..._OT_CFG_training_config.yaml`
    stem = ckpt_path.name.split("_epoch_")[0]
    return ckpt_path.parent / f"{stem}_training_config.yaml"


def evaluate_pose(meta, device, num_samples=256, num_steps=100, cfg_scale=3.0):
    conditional = meta["prefix"] == "cond_"
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model, _ = load_pose_model(
        str(meta["path"]), device=device, model_config=model_config, conditional=conditional,
    )

    goal_dist = OmegaConf.to_container(config.training.goal_dist_params, resolve=True)
    start_dist = OmegaConf.to_container(config.training.start_dist_params, resolve=True)
    start_dist_flat = {'mu': start_dist['mu'][0], 'sigma': start_dist['sigma'][0]}
    mode_poses = goal_mode_poses_from_config(goal_dist, device=device)  # [K, 7]

    row = {
        'task': 'pose', 'conditional': conditional,
        'ot': meta['ot'] == '_OT', 'cfg': meta['cfg'] == '_CFG',
        'cfg_scale_at_inference': cfg_scale if conditional else None,
        'num_samples': num_samples, 'epoch': meta['epoch'],
        'mode_accuracy': None, 'mode_coverage_kl': None,
        'class_accuracy': None, 'class_marginal_kl': None,
    }

    if conditional:
        per_mode = num_samples // 4
        all_samples, all_targets = [], []
        for mode_idx, action_pair in enumerate(ACTION_PAIRS):
            obs = build_obs_from_actions(list(action_pair), per_mode, device)
            _, samples = generate_from_distribution(
                model, start_dist_flat, batch_size=per_mode, obs=obs,
                num_steps=num_steps, return_trajectory=False,
                cfg_scale=cfg_scale, device=device,
            )
            all_samples.append(samples)
            all_targets.append(torch.full((per_mode,), mode_idx, dtype=torch.long))
        samples = torch.cat(all_samples, dim=0)
        targets = torch.cat(all_targets, dim=0)

        row['mode_accuracy'] = pose_mode_accuracy(samples, targets, mode_poses)
        kl, counts = pose_mode_coverage_kl(samples, mode_poses)
        row['mode_coverage_kl'] = kl
        row['per_mode_counts'] = dict(counts)
    else:
        _, samples = generate_from_distribution(
            model, start_dist_flat, batch_size=num_samples, obs=None,
            num_steps=num_steps, return_trajectory=False, device=device,
        )
        kl, counts = pose_mode_coverage_kl(samples, mode_poses)
        row['mode_coverage_kl'] = kl
        row['per_mode_counts'] = dict(counts)

    return row


def evaluate_image(meta, device, classifier, num_samples=256, num_steps=100, cfg_scale=3.0):
    conditional = meta["prefix"] == "cond_"
    config = OmegaConf.load(_config_path_for(meta["path"]))
    model_config = OmegaConf.to_container(config.model, resolve=True)
    model, _ = load_image_model(
        str(meta["path"]), device=device, model_config=model_config, conditional=conditional,
    )

    row = {
        'task': 'mnist', 'conditional': conditional,
        'ot': meta['ot'] == '_OT', 'cfg': meta['cfg'] == '_CFG',
        'cfg_scale_at_inference': cfg_scale if conditional else None,
        'num_samples': num_samples, 'epoch': meta['epoch'],
        'mode_accuracy': None, 'mode_coverage_kl': None,
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

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--output', type=str, default='experiments/results/metrics.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"No checkpoint directory at {ckpt_dir}")

    classifier = None
    if Path(args.classifier_path).exists():
        classifier = load_classifier(args.classifier_path, device=device)
        print(f"Loaded MNIST classifier from {args.classifier_path}")
    else:
        print(f"WARNING: no classifier at {args.classifier_path} — MNIST metrics will be skipped. "
              f"Train one with: python -m utils.mnist_classifier")

    rows = []
    for meta in discover_checkpoints(ckpt_dir):
        tag = (f"{'cond_' if meta['prefix'] else ''}{meta['task']}"
               f"{meta['ot']}{meta['cfg']} (epoch {meta['epoch']})")
        print(f"\n=== Evaluating {tag} ===")
        try:
            if meta['task'] == 'pose':
                row = evaluate_pose(meta, device,
                                    num_samples=args.num_samples,
                                    num_steps=args.num_steps,
                                    cfg_scale=args.cfg_scale)
            else:
                if classifier is None:
                    print("Skipping MNIST eval — no classifier available.")
                    continue
                row = evaluate_image(meta, device, classifier,
                                     num_samples=args.num_samples,
                                     num_steps=args.num_steps,
                                     cfg_scale=args.cfg_scale)
        except Exception as e:
            print(f"ERROR evaluating {tag}: {e}")
            continue
        rows.append(row)
        print({k: v for k, v in row.items() if v is not None})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not rows:
        print("No rows produced.")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()
