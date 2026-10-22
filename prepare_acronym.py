"""One-time preparation of the ACRONYM grasp data [Eppner et al. 2021] for `utils.acronym`.

  python -I prepare_acronym.py index
      Scan every grasp file in data/acronym/raw/grasps into data/acronym/index.csv: category,
      ShapeNet id, scale, successful grasp count and size of each object.
  python -I prepare_acronym.py build --dataset configs/datasets/acronym_10cat.yaml [--mesh_root DIR]
      Pick the dataset's objects, split them and write its cache (the file's `cache` path).
  python -I prepare_acronym.py watertight --dataset configs/datasets/acronym_10cat.yaml \
          --obj_dir <ShapeNetSem>/models-OBJ/models --manifold_dir <Manifold>/build
      Make watertight meshes of the cache's objects with Manifold [Huang et al. 2018], into
      data/acronym/meshes/<category>/<id>.obj; then rebuild with --mesh_root data/acronym for
      point clouds. ACRONYM ran `manifold -s` and then `simplify -m -r 0.02`; here `-s` (sharp
      features) leaves garbage vertices in ~28% of these meshes and `simplify` aborts or hangs on
      them, while plain `manifold` is closer to ACRONYM's own meshes on the two example objects
      (0.8 and 1.3 mm mean chamfer, against 1.3 and 2.3 mm with -s, simplified or not).

Grasps are kept only if the object stayed in the gripper (`object_in_gripper`). Each is stored
as a pose (x, y, z, qw, qx, qy, qz) in metres in the object frame: the mesh frame shifted to the
object's centre of mass (`object/com`, within a few mm of the mesh centroid), with the gripper
frame moved from the Franka hand's base to the centre of the finger pads, TCP_OFFSET along the
approach axis z. Point clouds need the ShapeNetSem meshes (--mesh_root).
"""
import argparse
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

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
    torch.save(cache, path + '.tmp')
    os.replace(path + '.tmp', path)  # readers never see a half-written cache
    counts = {name: int((split == code).sum()) for name, code in
              [('train', TRAIN), *LEVELS.items()]}
    print(f'{len(objects)} objects, {counts}, pos_scale {cache["pos_scale"]:.4f} m -> {path}')


def sample_point_clouds(objects, mesh_root, data):
    """[num_objects, R, P, 3] surface samples of each watertight mesh, in the object frame."""
    import trimesh
    R, P = data['point_samplings'], data['points_per_cloud']
    clouds = np.zeros((len(objects), R, P, 3), dtype=np.float32)
    offsets = []
    for i, obj in enumerate(objects):
        mesh = trimesh.load(os.path.join(mesh_root, obj['mesh_file']), force='mesh')
        mesh.apply_scale(obj['scale'])
        # alignment check: the mesh's centroid against ACRONYM's centre of mass
        offsets.append(np.linalg.norm(mesh.centroid - np.asarray(obj['com'])))
        obj['centroid_offset_mm'] = float(1000 * offsets[-1])
        for r in range(R):
            pts, _ = trimesh.sample.sample_surface(mesh, P, seed=r)
            clouds[i, r] = pts - np.asarray(obj['com'])
    offsets = 1000 * np.array(offsets)
    print(f'mesh centroid vs ACRONYM centre of mass: median {np.median(offsets):.1f} mm, '
          f'95th percentile {np.percentile(offsets, 95):.1f} mm, max {offsets.max():.1f} mm')
    return torch.from_numpy(clouds)


def _within(mesh_file, source_file, margin=0.1):
    """True if every vertex of mesh_file is finite and inside source_file's bounding box grown by
    `margin` of its size."""
    import trimesh
    lo, hi = trimesh.load(source_file, force='mesh').bounds
    v = trimesh.load(mesh_file, force='mesh').vertices
    pad = margin * (hi - lo).max()
    return bool(np.isfinite(v).all() and (v >= lo - pad).all() and (v <= hi + pad).all())


def make_watertight(dataset, obj_dir, manifold_dir, workers=8, timeout=600):
    """Watertight copies of the cache's meshes under data/acronym/<mesh_file>."""
    with open(dataset) as f:
        data = yaml.safe_load(f)
    objects = torch.load(os.path.join(REPO, data['cache']), weights_only=False)['objects']
    root = os.path.join(REPO, 'data', 'acronym')

    def one(obj):
        out = os.path.join(root, obj['mesh_file'])
        if os.path.exists(out):
            return None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            watertight = os.path.join(tmp, 'watertight.obj')
            source = os.path.join(obj_dir, obj['shapenet_id'] + '.obj')
            cmd = [os.path.join(manifold_dir, 'manifold'), source, watertight]
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return f"{obj['file']}: manifold timed out"
            if done.returncode != 0 or not os.path.exists(watertight):
                return f"{obj['file']}: manifold failed: {done.stderr[-200:]}"
            if not _within(watertight, source):
                return f"{obj['file']}: manifold output has vertices far outside the model"
            os.replace(watertight, out)
        return None

    with ThreadPoolExecutor(workers) as pool:
        errors = [e for e in pool.map(one, objects) if e]
    print(f'{len(objects) - len(errors)} watertight meshes under {root}/meshes')
    for e in errors:
        print('  ' + e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('stage', choices=('index', 'build', 'watertight'))
    parser.add_argument('--dataset', type=str, help='build: the dataset file (configs/datasets/)')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='build: folder holding meshes/<category>/<id>.obj (watertight); '
                             'without it the cache has no point clouds')
    parser.add_argument('--obj_dir', type=str, help="watertight: ShapeNetSem's models-OBJ/models folder")
    parser.add_argument('--manifold_dir', type=str, help='watertight: the folder holding the built '
                                                          'manifold and simplify executables')
    args = parser.parse_args()
    if args.stage == 'index':
        build_index()
    elif args.stage == 'watertight':
        make_watertight(args.dataset, args.obj_dir, args.manifold_dir)
    else:
        build_cache(args.dataset, args.mesh_root)
