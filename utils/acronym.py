"""ACRONYM grasp poses [Eppner et al. 2021] as a pose-generation task (`type: acronym`).

The data is a cache built once by `prepare_acronym.py` from a dataset file (configs/datasets/):
the successful grasps of a set of objects as poses in each object's frame, split into training
grasps and three held-out levels, plus surface point clouds of the objects when the meshes
were available. A task file (configs/pose_tasks/acronym_*.yaml) picks the dataset and how the
model is conditioned on the object:
  condition: points   the object's point cloud, tokenised into patches (continuous conditioning)
  condition: ids      [category, object] id tokens (discrete conditioning)
  start:     the prior; translation N(0, trans_sigma²) per axis, and a rotation that is either
             uniform on SO(3) or exp of N(0, rot_sigma²) per axis (`ROT_PRIORS`)
  batch:     objects × grasps_per_object samples per network batch; the object is the
             condition id, so per-condition OT pairs within each object
  eval:      objects per held-out level and generated samples per object
Positions and point clouds are in units of the dataset's pos_scale, the std of the training
grasps' positions, so poses are O(1) as in the synthetic tasks. Training goals get a random half
turn about the approach axis, which makes the data exactly symmetric under the jaw swap.
"""
import os

import torch
import yaml

from utils.tf_utils import _quaternion_multiply, compute_twist_between_poses

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Split code of each grasp in the cache
TRAIN, HELDOUT_GRASP, HELDOUT_OBJECT, HELDOUT_CATEGORY = 0, 1, 2, 3
# Held-out levels: grasps of training objects, objects of training categories, other categories
LEVELS = {'heldout_grasps': HELDOUT_GRASP, 'heldout_objects': HELDOUT_OBJECT,
          'heldout_categories': HELDOUT_CATEGORY}
CONDITIONS = ('points', 'ids')
ROT_PRIORS = ('uniform', 'gaussian')
NUM_PATCHES = 64          # point-cloud condition tokens
EVAL_SEED = 2024          # fixed evaluation objects and reference sets

# Quaternion (w, x, y, z) of a half turn about the approach axis z: the jaw swap
JAW_FLIP = (0.0, 0.0, 0.0, 1.0)


def random_rotations(n, generator=None, device='cpu'):
    """n quaternions uniform on SO(3) (normalised 4-D Gaussians), with qw >= 0."""
    q = torch.randn(n, 4, generator=generator).to(device)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


def load_dataset(dataset):
    """The dataset file (a path relative to the repo, or absolute) as a dict."""
    with open(os.path.join(REPO, dataset)) as f:
        return yaml.safe_load(f)


class AcronymGraspTask:
    """Grasp poses of ACRONYM objects, conditioned on the object. See the module docstring."""

    def __init__(self, spec, device='cpu'):
        self.spec = spec
        self.condition = spec['condition']
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {self.condition!r}. Expected one of {CONDITIONS}.")
        start = spec.get('start', {})
        self.trans_sigma = float(start.get('trans_sigma', 1.0))
        self.rotation = start.get('rotation', 'uniform')
        if self.rotation not in ROT_PRIORS:
            raise ValueError(f"Unknown rotation prior: {self.rotation!r}. Expected one of {ROT_PRIORS}.")
        self.rot_sigma = float(start.get('rot_sigma', 1.0))
        self.objects_per_batch = int(spec['batch']['objects'])
        self.grasps_per_object = int(spec['batch']['grasps_per_object'])
        ev = spec.get('eval', {})
        self.eval_objects_per_level = int(ev.get('objects_per_level', 64))
        self.samples_per_object = int(ev.get('samples_per_object', 256))
        self.max_reference = int(ev.get('max_reference', 512))
        self.device = device

        data = load_dataset(spec['dataset'])
        cache = torch.load(os.path.join(REPO, data['cache']), weights_only=False)
        self.pos_scale = float(cache['pos_scale'])
        self.tcp_offset = float(cache['tcp_offset'])
        poses = cache['poses'].clone()
        poses[:, :3] /= self.pos_scale
        self.poses = poses.to(device)
        self.object_index = cache['object_index'].to(device)
        self.split = cache['split'].to(device)
        self.objects = cache['objects']
        self.categories = cache['categories']
        self.category_ids = torch.tensor([o['category_id'] for o in self.objects], device=device)
        points = cache.get('points')
        self.points = points.to(device) / self.pos_scale if points is not None else None
        if self.condition == 'points' and self.points is None:
            raise ValueError(f"{data['cache']} has no point clouds: build it with "
                             "prepare_acronym.py build --mesh_root <meshes>")

        # Training grasps grouped by object, for drawing k grasps of each sampled object
        train = (self.split == TRAIN).nonzero().squeeze(1)
        train = train[self.object_index[train].argsort(stable=True)]
        objs, counts = torch.unique_consecutive(self.object_index[train], return_counts=True)
        self.train_grasps, self.train_objects, self.train_counts = train, objs, counts
        self.train_offsets = counts.cumsum(0) - counts

    # -- conditioning -------------------------------------------------------
    @property
    def obs_encoder(self):
        return 'point_patch' if self.condition == 'points' else 'embedding'

    @property
    def num_obs_tokens(self):
        return NUM_PATCHES if self.condition == 'points' else 2

    @property
    def obs_dim(self):
        return 3 if self.condition == 'points' else 1

    @property
    def obs_vocab(self):
        """Ids: the categories, then the objects."""
        return len(self.categories) + len(self.objects) if self.condition == 'ids' else None

    @property
    def batch_size(self):
        return self.objects_per_batch * self.grasps_per_object

    def obs(self, objects, sampling=0):
        """Condition of each object in objects [n]: its point cloud [n, P, 3] (surface sampling
        `sampling`, an int or [n]) or its ids [n, 2] = (category, num_categories + object)."""
        if self.condition == 'ids':
            return torch.stack([self.category_ids[objects], len(self.categories) + objects], -1)
        return self.points[objects, sampling]

    # -- sampling -------------------------------------------------------------
    def sample_start(self, n, generator=None, device=None):
        """n prior samples as twists [n, 6] (positions in pos_scale units)."""
        device = device or self.device
        trans = self.trans_sigma * torch.randn(n, 3, generator=generator).to(device)
        if self.rotation == 'gaussian':
            rot = self.rot_sigma * torch.randn(n, 3, generator=generator).to(device)
            return torch.cat([trans, rot], dim=-1)
        pose = torch.cat([trans, random_rotations(n, generator, device)], dim=-1)
        return compute_twist_between_poses(pose, None)

    def batch_sampler(self, seq_len=1, grasps_per_object=None):
        """`Pairer` sampler n -> (start [n, 1, 6], goal [n, 1, 6], obs, object ids [n]): n / k
        training objects (distinct while there are enough), k random training grasps of each,
        and one point-cloud sampling per object."""
        if seq_len != 1:
            raise ValueError("ACRONYM grasp tasks support seq_len = 1 only")
        k = grasps_per_object or self.grasps_per_object
        flip = torch.tensor(JAW_FLIP, device=self.device)

        def sampler(n):
            if n % k:
                raise ValueError(f"batch size {n} must be a multiple of {k} grasps per object")
            m, num = n // k, len(self.train_objects)
            pick = (torch.randperm(num, device=self.device)[:m] if m <= num
                    else torch.randint(num, (m,), device=self.device))
            j = (torch.rand(m, k, device=self.device) * self.train_counts[pick, None]).long()
            goal = self.poses[self.train_grasps[(self.train_offsets[pick, None] + j).flatten()]]
            turned = torch.rand(n, 1, device=self.device) < 0.5
            quat = torch.where(turned, _quaternion_multiply(goal[:, 3:], flip.expand(n, 4)),
                               goal[:, 3:])
            goal = compute_twist_between_poses(torch.cat([goal[:, :3], quat], -1), None)
            objects = self.train_objects[pick].repeat_interleave(k)
            sampling = (torch.randint(self.points.shape[1], (m,), device=self.device)
                        .repeat_interleave(k) if self.condition == 'points' else 0)
            start = self.sample_start(n)
            return start.unsqueeze(1), goal.unsqueeze(1), self.obs(objects, sampling), objects
        return sampler

    # -- evaluation -----------------------------------------------------------
    def eval_objects(self, level):
        """The fixed evaluation objects of a held-out level (`LEVELS`), at most
        eval_objects_per_level of them."""
        code = LEVELS[level]
        have = self.object_index[self.split == code].unique()
        gen = torch.Generator().manual_seed(EVAL_SEED)
        order = torch.randperm(len(have), generator=gen)
        return have.cpu()[order][:self.eval_objects_per_level].tolist()

    def reference_and_floor(self, obj, level, n_floor):
        """Real poses of one evaluation object, in metres: the reference set the samples are
        scored against (at most max_reference held-out grasps), and a disjoint real set of up
        to n_floor grasps scored the same way (the data floor). For held-out grasps the floor
        comes from the object's training grasps; otherwise its grasps are split in two."""
        gen = torch.Generator().manual_seed(EVAL_SEED + obj)
        mine = self.object_index == obj
        if level == 'heldout_grasps':
            ref = (mine & (self.split == HELDOUT_GRASP)).nonzero().squeeze(1)
            other = (mine & (self.split == TRAIN)).nonzero().squeeze(1)
            ref = ref[torch.randperm(len(ref), generator=gen).to(ref.device)]
        else:
            idx = mine.nonzero().squeeze(1)
            idx = idx[torch.randperm(len(idx), generator=gen).to(idx.device)]
            ref, other = idx[:len(idx) // 2], idx[len(idx) // 2:]
        other = other[torch.randperm(len(other), generator=gen).to(other.device)]
        return (self.to_metres(self.poses[ref[:self.max_reference]]),
                self.to_metres(self.poses[other[:n_floor]]))

    def to_metres(self, poses):
        """Poses [..., 7] from pos_scale units to metres."""
        return torch.cat([poses[..., :3] * self.pos_scale, poses[..., 3:]], dim=-1)


def load_acronym_task(spec, device='cpu'):
    """`load_pose_task`'s dict for an ACRONYM task file."""
    task = AcronymGraspTask(spec, device)
    trans, rot = task.trans_sigma, task.rot_sigma
    return {
        'name': spec['name'],
        'type': 'acronym',
        'spec': spec,
        'task': task,
        'mode_names': None,
        'tokens': None,
        'obs_dim': task.obs_dim,
        # the prior, for the record (the task samples it, see `sample_start`)
        'start_dist_params': {'mu': [[0.0] * 6], 'sigma': [[trans] * 3 + [rot] * 3],
                              'rotation': task.rotation},
        'goal_dist_params': None,
        'action_dist_params': None,
    }
