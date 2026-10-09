"""One-time preparation of the ACRONYM grasp data [Eppner et al. 2021] for `utils.acronym`.

  python -I prepare_acronym.py index
      Scan every grasp file in data/acronym/raw/grasps into data/acronym/index.csv: category,
      ShapeNet id, scale, successful grasp count and size of each object.
  python -I prepare_acronym.py build --dataset configs/datasets/acronym_10cat.yaml [--mesh_root DIR]
      Pick the dataset's objects, split them and write its cache (the file's `cache` path).

Grasps are kept only if the object stayed in the gripper (`object_in_gripper`). Each is stored
as a pose (x, y, z, qw, qx, qy, qz) in metres in the object frame: the mesh frame shifted to the
object's centre of mass (`object/com`, within a few mm of the mesh centroid), with the gripper
frame moved from the Franka hand's base to the centre of the finger pads, TCP_OFFSET along the
approach axis z. Point clouds need the ShapeNetSem meshes (--mesh_root).
"""
import argparse
import csv
import os

import h5py
import numpy as np
import torch
import yaml

from utils.acronym import HELDOUT_CATEGORY, HELDOUT_OBJECT, LEVELS, TRAIN, HELDOUT_GRASP
from utils.tf_utils import _rot_mat_to_quat

REPO = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(REPO, 'data', 'acronym', 'raw', 'grasps')
INDEX = os.path.join(REPO, 'data', 'acronym', 'index.csv')

# Franka hand: the finger pads span z = 0.066 to 0.112 m from the hand's base frame
TCP_OFFSET = 0.089


def parse_name(filename):
    """'<category>_<shapenet id>_<scale>.h5' -> (category, shapenet id, scale)."""
    category, shapenet_id, scale = os.path.basename(filename)[:-3].split('_')
    return category, shapenet_id, float(scale)


def read_grasps(path):
    """Successful grasps of one file as poses [n, 7] in the object frame, plus its h5 fields."""
    with h5py.File(path, 'r') as f:
        T = f['grasps/transforms'][()]
        success = f['grasps/qualities/flex/object_in_gripper'][()] > 0
        com = f['object/com'][()]
        mesh_file = f['object/file'][()].decode()
        scale = float(f['object/scale'][()])
    T = T[success]
    R = T[:, :3, :3]
    pos = T[:, :3, 3] + R[:, :, 2] * TCP_OFFSET - com
    quat = _rot_mat_to_quat(torch.from_numpy(R).double()).numpy()
    poses = np.concatenate([pos, quat], axis=1).astype(np.float32)
    return poses, {'com': com, 'mesh_file': mesh_file, 'scale': scale,
                   'num_grasps': len(success)}


def build_index():
    files = sorted(f for f in os.listdir(RAW) if f.endswith('.h5'))
    rows = []
    for i, name in enumerate(files):
        category, shapenet_id, _ = parse_name(name)
        poses, info = read_grasps(os.path.join(RAW, name))
        spread = poses[:, :3].std(0).mean() if len(poses) > 1 else 0.0
        rows.append({'file': name, 'category': category, 'shapenet_id': shapenet_id,
                     'scale': info['scale'], 'num_grasps': info['num_grasps'],
                     'num_success': len(poses), 'grasp_spread_m': f'{spread:.4f}',
                     'mesh_file': info['mesh_file']})
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(files)}')
    with open(INDEX, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} objects -> {INDEX}')


def _eligible(index, categories, min_success):
    """Per category, its objects with enough successful grasps, one file per ShapeNet id."""
    out = {}
    for c in categories:
        seen, keep = set(), []
        for row in index:
            if (row['category'] == c and int(row['num_success']) >= min_success
                    and row['shapenet_id'] not in seen):
                seen.add(row['shapenet_id'])
                keep.append(row)
        out[c] = keep
    return out


def build_cache(dataset, mesh_root=None):
    with open(dataset) as f:
        data = yaml.safe_load(f)
    with open(INDEX) as f:
        index = list(csv.DictReader(f))
    rng = np.random.default_rng(data['split_seed'])

    # Objects: training categories split into training and held-out objects; held-out
    # categories contribute only held-out objects
    chosen = []  # (row, category id, level)
    if 'files' in data:  # an explicit list of training objects (smoke tests)
        rows = [r for r in index if r['file'] in data['files']]
        data.setdefault('heldout_categories', [])
        data['train_categories'] = sorted({r['category'] for r in rows})
        chosen = [(r, r['category'], TRAIN) for r in rows]
    train_cats, test_cats = data['train_categories'], data['heldout_categories']
    eligible = _eligible(index, train_cats + test_cats, data['min_success']) if not chosen else {}
    for c in (train_cats + test_cats) if not chosen else []:
        rows = [eligible[c][i] for i in rng.permutation(len(eligible[c]))]
        if c in train_cats:
            rows = rows[:data['max_objects_per_category']]
            n_test = max(1, round(data['heldout_object_fraction'] * len(rows)))
            levels = [HELDOUT_OBJECT] * n_test + [TRAIN] * (len(rows) - n_test)
        else:
            rows = rows[:data['max_objects_per_heldout_category']]
            levels = [HELDOUT_CATEGORY] * len(rows)
        chosen += [(row, c, level) for row, level in zip(rows, levels)]
        print(f'  {c:16s} {len(eligible[c]):4d} eligible, {len(rows):3d} used')

    categories = train_cats + test_cats
    poses, object_index, split, objects = [], [], [], []
    for i, (row, c, level) in enumerate(chosen):
        p, info = read_grasps(os.path.join(RAW, row['file']))
        s = np.full(len(p), level, dtype=np.uint8)
        if level == TRAIN:  # hold out some of a training object's grasps
            n_test = round(data['heldout_grasp_fraction'] * len(p))
            s[rng.permutation(len(p))[:n_test]] = HELDOUT_GRASP
        poses.append(p)
        object_index.append(np.full(len(p), i))
        split.append(s)
        objects.append({'file': row['file'], 'category': c, 'category_id': categories.index(c),
                        'shapenet_id': row['shapenet_id'], 'scale': info['scale'],
                        'com': info['com'].tolist(), 'mesh_file': info['mesh_file'],
                        'level': int(level)})

    poses = torch.from_numpy(np.concatenate(poses))
    split = torch.from_numpy(np.concatenate(split))
    cache = {
        'poses': poses,
        'object_index': torch.from_numpy(np.concatenate(object_index)),
        'split': split,
        'objects': objects,
        'name': data['name'],
        'categories': categories,
        'train_categories': train_cats,
        # one length unit: the std of the training grasps' positions, over all axes
        'pos_scale': float(poses[split == TRAIN, :3].std()),
        'tcp_offset': TCP_OFFSET,
        'points': sample_point_clouds(objects, mesh_root, data) if mesh_root else None,
    }
    path = os.path.join(REPO, data['cache'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    counts = {name: int((split == code).sum()) for name, code in
              [('train', TRAIN), *LEVELS.items()]}
    print(f'{len(objects)} objects, {counts}, pos_scale {cache["pos_scale"]:.4f} m -> {path}')


def sample_point_clouds(objects, mesh_root, data):
    """[num_objects, R, P, 3] surface samples of each watertight mesh, in the object frame."""
    import trimesh
    R, P = data['point_samplings'], data['points_per_cloud']
    clouds = np.zeros((len(objects), R, P, 3), dtype=np.float32)
    for i, obj in enumerate(objects):
        mesh = trimesh.load(os.path.join(mesh_root, obj['mesh_file']), force='mesh')
        mesh.apply_scale(obj['scale'])
        for r in range(R):
            pts, _ = trimesh.sample.sample_surface(mesh, P, seed=r)
            clouds[i, r] = pts - np.asarray(obj['com'])
    return torch.from_numpy(clouds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('stage', choices=('index', 'build'))
    parser.add_argument('--dataset', type=str, help='build: the dataset file (configs/datasets/)')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='build: folder holding meshes/<category>/<id>.obj (watertight); '
                             'without it the cache has no point clouds')
    args = parser.parse_args()
    if args.stage == 'index':
        build_index()
    else:
        build_cache(args.dataset, args.mesh_root)
