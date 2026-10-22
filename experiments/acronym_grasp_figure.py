"""Generated against held-out real grasps for a few evaluation objects of an ACRONYM model:
gripper wireframes over the object's point cloud (when the task has one), one panel per object.

  python experiments/acronym_grasp_figure.py --checkpoint <run>_epoch_100.pt --level heldout_objects
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import _acronym_task, _check_level, _config_path_for
from pose_gen_inference import load_model
from utils.acronym import LEVELS
from utils.tf_utils import convert_twist_to_pose
from utils.visualization_utils import plot_grasps

REAL, SAMPLE, CLOUD = '#8a8a85', '#2a78d6', '#c8c8c4'  # reference grey, series blue, recessive points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--level', type=str, default='heldout_grasps', choices=tuple(LEVELS))
    parser.add_argument('--num_objects', type=int, default=4)
    parser.add_argument('--num_grasps', type=int, default=24)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--method', type=str, default='midpoint')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = Path(args.checkpoint)
    config = OmegaConf.load(_config_path_for(ckpt))
    task = _acronym_task(config, device)
    _check_level(task, args.level)
    model, _ = load_model(str(ckpt), device=device,
                          model_config=OmegaConf.to_container(config.model, resolve=True),
                          conditional=config.training.conditional)
    torch.manual_seed(0)

    objects = task.eval_objects(args.level)[:args.num_objects]
    fig = plt.figure(figsize=(4 * len(objects), 4.4))
    for i, obj in enumerate(objects):
        n = args.num_grasps
        x0 = convert_twist_to_pose(task.sample_start(n, device=device), dt=1.0, return_representation='quat')
        if config.training.conditional:
            obs = task.obs(torch.full((n,), obj, device=device))
            samples = model.inference(x0, obs, num_steps=args.num_steps, cfg_scale=1.0, method=args.method)
        else:
            samples = model.inference(x0, num_steps=args.num_steps, method=args.method)
        samples = task.to_metres(samples.reshape(n, 7))
        ref, _ = task.reference_and_floor(obj, args.level, n)
        ax = fig.add_subplot(1, len(objects), i + 1, projection='3d')
        if task.points is not None:
            p = task.points[obj, 0].cpu() * task.pos_scale
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, color=CLOUD, depthshade=False)
        pts = torch.cat([plot_grasps(ax, ref[:n], task.tcp_offset, REAL, label='held-out real'),
                         plot_grasps(ax, samples, task.tcp_offset, SAMPLE, label='generated')])
        centre, half = pts.mean(0), (pts.max(0).values - pts.min(0).values).max() / 2
        for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            set_lim(c - half, c + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"{task.objects[obj]['category']} {task.objects[obj]['shapenet_id'][:6]}",
                     fontsize=10, color='#333333')
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.suptitle(f"{ckpt.stem}: {args.level.replace('_', ' ')}, {args.num_steps} {args.method} steps",
                 fontsize=11, color='#333333')
    out = args.output or str(ckpt.with_name(f'{ckpt.stem}_grasps_{args.level}.png'))
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
