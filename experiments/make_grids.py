"""Generate the comparison figures for the HuggingFace model card.

Three figure families:
  1. Pose trajectory grids — 4-up plot of conditional-pose samples per ablation variant.
  2. MNIST sample grids — 4-up tile of class-conditional MNIST samples per ablation variant.
  3. CFG sweep curves — accuracy vs cfg_scale (one line per OT setting per task),
     read from experiments/results/cfg_sweep.csv.

Saves PNGs to experiments/results/.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import ACTION_PAIRS, discover_checkpoints
from pose_gen_inference import (
    build_obs_from_actions, generate_from_distribution, load_model as load_pose_model,
)
from image_gen_inference import (
    build_obs_from_digit, generate_from_noise, load_model as load_image_model,
)
from utils.visualization_utils import visualize_image_trajectory, visualize_trajectory


def _config_path_for(ckpt_path: Path):
    stem = ckpt_path.name.split("_epoch_")[0]
    return ckpt_path.parent / f"{stem}_training_config.yaml"


def _pose_variants(ckpt_dir):
    """Return the 4 conditional pose ablation checkpoints, keyed by 'OT_CFG' etc."""
    out = {}
    for meta in discover_checkpoints(ckpt_dir):
        if meta['task'] != 'pose' or meta['prefix'] != 'cond_':
            continue
        key = f"{meta['ot'].lstrip('_')}_{meta['cfg'].lstrip('_')}"
        out[key] = meta
    return out


def _image_variants(ckpt_dir):
    out = {}
    for meta in discover_checkpoints(ckpt_dir):
        if meta['task'] != 'image' or meta['prefix'] != 'cond_':
            continue
        key = f"{meta['ot'].lstrip('_')}_{meta['cfg'].lstrip('_')}"
        out[key] = meta
    return out


def make_pose_grid(ckpt_dir: Path, output: Path, device, num_samples=5,
                   num_steps=50, cfg_scale=3.0):
    """For each of the 4 conditional-pose variants, sample one trajectory per mode
    and stitch into a 2x2 figure of 3D trajectory plots."""
    variants = _pose_variants(ckpt_dir)
    ordered = [('OT_CFG', 'Full (OT + CFG)'), ('OT_NOCFG', 'No CFG'),
               ('NOOT_CFG', 'No OT'), ('NOOT_NOCFG', 'Baseline (no OT, no CFG)')]

    fig = plt.figure(figsize=(14, 12))
    for idx, (key, title) in enumerate(ordered):
        meta = variants.get(key)
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        if meta is None:
            ax.set_title(f"{title}\n(missing checkpoint)")
            continue

        config = OmegaConf.load(_config_path_for(meta['path']))
        model_config = OmegaConf.to_container(config.model, resolve=True)
        model, _ = load_pose_model(str(meta['path']), device=device,
                                   model_config=model_config, conditional=True)
        start_dist = OmegaConf.to_container(config.training.start_dist_params, resolve=True)
        start_dist_flat = {'mu': start_dist['mu'][0], 'sigma': start_dist['sigma'][0]}

        all_traj = []
        for action_pair in ACTION_PAIRS:
            obs = build_obs_from_actions(list(action_pair), num_samples, device)
            _, traj = generate_from_distribution(
                model, start_dist_flat, batch_size=num_samples, obs=obs,
                num_steps=num_steps, return_trajectory=True,
                cfg_scale=cfg_scale, device=device,
            )
            all_traj.append(traj)
        trajectory = torch.cat(all_traj, dim=0).cpu()

        # Plot positions only (trajectory[..., :3]) for a clean comparison.
        for b in range(trajectory.shape[0]):
            path = trajectory[b, :, :3].numpy()
            ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.5)
            ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], s=20)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    fig.suptitle('Conditional pose generation — ablation comparison', fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def make_mnist_grid(ckpt_dir: Path, output: Path, device, num_steps=100, cfg_scale=3.0):
    """For each of the 4 conditional-MNIST variants, sample digits 0-9 (one each)
    and stitch into a 2x2 figure of 1x10 sample strips."""
    variants = _image_variants(ckpt_dir)
    ordered = [('OT_CFG', 'Full (OT + CFG)'), ('OT_NOCFG', 'No CFG'),
               ('NOOT_CFG', 'No OT'), ('NOOT_NOCFG', 'Baseline (no OT, no CFG)')]

    fig, axes = plt.subplots(4, 10, figsize=(14, 6))
    for row, (key, title) in enumerate(ordered):
        meta = variants.get(key)
        if meta is None:
            for col in range(10):
                axes[row, col].axis('off')
            axes[row, 0].set_ylabel(f"{title}\n(missing)", rotation=0, ha='right', va='center')
            continue
        config = OmegaConf.load(_config_path_for(meta['path']))
        model_config = OmegaConf.to_container(config.model, resolve=True)
        model, _ = load_image_model(str(meta['path']), device=device,
                                    model_config=model_config, conditional=True)
        for digit in range(10):
            obs = build_obs_from_digit(digit, 1, device)
            img = generate_from_noise(model, 1, obs=obs, num_steps=num_steps,
                                      return_trajectory=False,
                                      cfg_scale=cfg_scale, device=device)
            arr = img[0, 0].cpu().numpy()
            axes[row, digit].imshow(arr, cmap='gray', vmin=-1, vmax=1)
            axes[row, digit].axis('off')
            if row == 0:
                axes[row, digit].set_title(str(digit))
        axes[row, 0].set_ylabel(title, rotation=0, ha='right', va='center', labelpad=40)

    fig.suptitle('Class-conditional MNIST — ablation comparison', fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def make_cfg_sweep_plot(csv_path: Path, output: Path):
    if not csv_path.exists():
        print(f"No CFG sweep CSV at {csv_path} — skipping plot.")
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    fig, (ax_pose, ax_mnist) = plt.subplots(1, 2, figsize=(12, 4))

    def _plot_task(ax, task_name, metric_key, title):
        series = {}
        for r in rows:
            if r.get('task') != task_name:
                continue
            variant = r.get('variant', '?')
            try:
                x = float(r['cfg_scale_at_inference'])
                y = float(r[metric_key])
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(variant, []).append((x, y))
        for variant, pts in sorted(series.items()):
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker='o', label=variant)
        ax.set_title(title)
        ax.set_xlabel('cfg_scale')
        ax.set_ylabel(metric_key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_task(ax_pose, 'pose', 'mode_accuracy', 'SE(3) pose: mode accuracy vs CFG')
    _plot_task(ax_mnist, 'mnist', 'class_accuracy', 'MNIST: class accuracy vs CFG')

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--results_dir', type=str, default='experiments/results/')
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--num_steps_pose', type=int, default=50)
    parser.add_argument('--num_steps_image', type=int, default=100)
    parser.add_argument('--skip_pose', action='store_true')
    parser.add_argument('--skip_image', action='store_true')
    parser.add_argument('--skip_cfg_sweep', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)

    if not args.skip_pose:
        make_pose_grid(ckpt_dir, results_dir / 'pose_grid.png', device,
                       num_steps=args.num_steps_pose, cfg_scale=args.cfg_scale)
    if not args.skip_image:
        make_mnist_grid(ckpt_dir, results_dir / 'mnist_grid.png', device,
                        num_steps=args.num_steps_image, cfg_scale=args.cfg_scale)
    if not args.skip_cfg_sweep:
        make_cfg_sweep_plot(results_dir / 'cfg_sweep.csv',
                            results_dir / 'cfg_sweep.png')


if __name__ == '__main__':
    main()
