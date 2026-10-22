"""Sanity check of the ACRONYM pose-recreation metrics on real data, before any training (gate
G1): score the data floor (real grasps against held-out real grasps) and three references
against it, per held-out level.

  data        a disjoint real set: the floor every model is compared with
  prior       samples of the task's prior (what an untrained flow returns)
  collapsed   one real grasp repeated: should fail recreation_err
  dispersed   real grasps plus noise (2 cm, 0.2 rad): should fail fidelity_err

  python experiments/acronym_metric_check.py --task_config configs/pose_tasks/acronym_10cat_ids.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_all import write_csv
from utils.acronym import LEVELS, load_acronym_task
from utils.grasp_metrics import mean_over_objects, recreation_metrics
from utils.tf_utils import _quaternion_multiply, convert_twist_to_pose


def references(task, obj, level, n, gen):
    """{name: (samples in metres, reference in metres)} for one object."""
    ref, floor = task.reference_and_floor(obj, level, n)
    prior = convert_twist_to_pose(task.sample_start(n, generator=gen, device='cpu'), dt=1.0,
                                  return_representation='quat').to(ref.device)
    noise = torch.cat([0.02 * torch.randn(len(floor), 3, generator=gen),
                       convert_twist_to_pose(torch.cat([torch.zeros(len(floor), 3),
                                                        0.2 * torch.randn(len(floor), 3, generator=gen)], -1),
                                             dt=1.0, return_representation='quat')[:, 3:]], -1).to(ref.device)
    dispersed = torch.cat([floor[:, :3] + noise[:, :3],
                           _quaternion_multiply(floor[:, 3:], noise[:, 3:])], -1)
    return {'data': floor, 'prior': task.to_metres(prior), 'collapsed': floor[:1].repeat(n, 1),
            'dispersed': dispersed}, ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--levels', type=str, nargs='+', default=list(LEVELS))
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spec = OmegaConf.to_container(OmegaConf.load(args.task_config), resolve=True)
    task = load_acronym_task(spec, device)['task']

    rows = []
    for level in args.levels:
        per_object = {}
        for obj in task.eval_objects(level):
            gen = torch.Generator().manual_seed(obj)
            sets, ref = references(task, obj, level, args.num_samples, gen)
            for name, samples in sets.items():
                per_object.setdefault(name, []).append(recreation_metrics(samples, ref, task.tcp_offset))
        for name, metrics in per_object.items():
            row = {'level': level, 'set': name, 'num_objects': len(metrics), **mean_over_objects(metrics)}
            rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    output = args.output or f"experiments/results/sota/pose_{spec['name']}/metric_check.csv"
    write_csv(rows, output)


if __name__ == '__main__':
    main()
