import torch
import os
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from utils.euclid_utils import compute_velocity_between_states
from scipy.optimize import linear_sum_assignment
import numpy as np
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel


# Per-manifold shape conventions used by the trainer:
#   se3:        start/goal in twist (D=6); interpolated states are quaternion poses (D=7);
#               target vector field is twist (D=6).
#   euclidean:  start/goal/interpolated/target all share the same dim D (e.g. 16 patch features).
#
# To keep the rest of the loop generic, the manifold-dispatch helpers below return both the
# interpolated state and target velocity in their natural dims, and we pass those dims to the
# minibatch driver.

def sequence_ot_pairing(start_seq, goal_seq, manifold='se3'):
    """Apply OT pairing independently per sequence frame.

    For SE(3) pose flows this is correct: each frame is its own geodesic problem.
    For image (Euclidean) flows, prefer `flat_ot_pairing` so spatial coherence of
    the noise image is preserved across patches (otherwise each patch is paired
    against a different target image and the field becomes a tangle).
    """
    chunk_size = start_seq.shape[1]
    paired_start = torch.zeros_like(start_seq)

    for i in range(chunk_size):
        paired_start[:, i, :] = optimal_transport_pairing(
            start_seq[:, i, :],
            goal_seq[:, i, :],
            manifold=manifold,
        )
    return paired_start


def flat_ot_pairing(start_seq, goal_seq, manifold='euclidean'):
    """OT pairing on flattened sequences: [B, S, D] -> single permutation across the whole [B, S*D] vector.

    Used for image flows so all patches of a single noise sample are reordered together.
    """
    if manifold != 'euclidean':
        raise ValueError("flat_ot_pairing currently only supports manifold='euclidean'.")
    B, S, D = start_seq.shape
    start_flat = start_seq.reshape(B, S * D)
    goal_flat = goal_seq.reshape(B, S * D)
    paired_flat = optimal_transport_pairing(start_flat, goal_flat, manifold='euclidean')
    return paired_flat.reshape(B, S, D)


def optimal_transport_pairing(start, goal, manifold='se3'):
    """
    Compute optimal transport pairing between start and goal samples.

    Args:
        start: Tensor of shape [batch_size, D]. For 'se3', D=6 (twists).
        goal: Tensor of shape [batch_size, D]. For 'se3', D=6 (twists).
        manifold: 'se3' uses geodesic twist distance; 'euclidean' uses L2.

    Returns:
        paired_start: Tensor of shape [batch_size, D], reordered for OT.
    """
    dists = _pairwise_cost(start, goal, manifold)
    row_ind, col_ind = linear_sum_assignment(dists.detach().cpu().numpy())
    sorted_indices = np.argsort(col_ind)
    return start[row_ind[sorted_indices]]


def _pairwise_cost(start, goal, manifold='se3'):
    """[B, B] cost of pairing start i with goal j: the geodesic twist norm ('se3') or L2
    ('euclidean'), unsquared."""
    if manifold == 'se3':
        # Convert twists to quaternion poses for geodesic distance
        start_pose = convert_twist_to_pose(start, dt=1.0, return_representation='quat')
        goal_pose = convert_twist_to_pose(goal, dt=1.0, return_representation='quat')
        batch_size = start_pose.shape[0]

        start_repeated = start_pose.unsqueeze(1).repeat(1, batch_size, 1)
        goal_repeated = goal_pose.unsqueeze(0).repeat(batch_size, 1, 1)
        start_flat = start_repeated.reshape(batch_size * batch_size, 7)
        goal_flat = goal_repeated.reshape(batch_size * batch_size, 7)

        twist_pairwise = compute_twist_between_poses(start_flat, goal_flat, dt=1.0)
        dists = torch.norm(twist_pairwise, p=2, dim=-1).reshape(batch_size, batch_size)
    elif manifold == 'euclidean':
        # Plain pairwise L2 in the state space
        dists = torch.cdist(start, goal, p=2)
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}.")
    return dists


# Backward-compat alias for any external callers
def geodesic_optimal_transport_pairing(start_poses, goal_poses):
    return optimal_transport_pairing(start_poses, goal_poses, manifold='se3')


def _mode_condition_ids(action_mu):
    """Condition index of each goal mode. Modes named by identical token lists share one: the
    model can't tell them apart from its input, so OT pairs across them, not within each."""
    keys, ids = [], []
    for tokens in action_mu:
        key = tuple(tuple(float(x) for x in token) for token in tokens)
        if key not in keys:
            keys.append(key)
        ids.append(keys.index(key))
    return ids


def _pair_within_conditions(start, goal, cond_ids, pair_fn):
    """OT-pair `start` to `goal` separately within each condition.

    Pairing across the whole batch would hand each condition only the noise samples
    nearest its own goals, so a conditional model would never see the rest of the noise
    distribution for that condition, while sampling draws from all of it.
    """
    paired = torch.empty_like(start)
    for c in torch.unique(cond_ids):
        idx = (cond_ids == c).nonzero(as_tuple=True)[0]
        paired[idx] = pair_fn(start[idx], goal[idx])
    return paired


# -- condition-aware pairing ------------------------------------------------------------
#
# A conditional model is sampled from the full noise distribution under every condition, so
# training should show each condition all of it. Pairing within each condition does that
# for discrete conditions, but when every condition differs (continuous or noisy ones) each
# group is a single sample and it falls back to random pairing. The pairings (readme
# section 6):
#   independent  random pairing (I-CFM)
#   ot           OT within each discrete condition id; global OT when there are none
#   global       OT over the whole batch, ignoring conditions (skews each condition's noise)
#   c2ot         cost + w·d(c_i, c_j), w set per batch so a share r_tar of all pairs are
#                admissible [C²OT, Cheng & Schwing 2025]
#   c2ot_fixed   cost + w·d with one large w fixed at the start [COT-FM, Kerrigan et al. 2024]
#   cluster      cost + γ·|c̄_i − c̄_j|² on K-means centroids of the conditions
#                [COT Policy, Sochopoulos et al. 2025]
# Noise i starts out carrying the condition c_i of the data row it was drawn with, which is a
# random partner. d is the squared distance between condition vectors. Only the noise is
# reordered: the network sees each data row with its own condition.
PAIRINGS = ('independent', 'ot', 'global', 'c2ot', 'c2ot_fixed', 'cluster')


def _batch_cost(start, goal, manifold='se3', ot_mode='per_frame'):
    """[B, B] cost of pairing noise sequence i with data sequence j [B, S, D]: per-frame
    costs summed ('per_frame'), or the L2 cost of the flattened sequences ('flat')."""
    if ot_mode == 'flat':
        return _pairwise_cost(start.flatten(1), goal.flatten(1), manifold='euclidean')
    return sum(_pairwise_cost(start[:, s], goal[:, s], manifold) for s in range(start.shape[1]))


def _sq_dists(a, b):
    """[N, M] squared Euclidean distances between the rows of a [N, C] and b [M, C]."""
    return (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)


def _assignment(cost):
    """Hungarian assignment as an index into the noise: noise perm[j] goes with data row j."""
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(row[np.argsort(col)], device=cost.device)


def admissible_ratio(base, cond_dist, w):
    """C²OT's r(w): the share of (noise i, data j) pairs whose cost, condition penalty
    included, is no more than that of noise i's own random partner (j = i)."""
    return ((base + w * cond_dist) <= base.diagonal().unsqueeze(1)).float().mean().item()


def c2ot_weight(base, cond_dist, r_tar, w_init=None, iters=30, max_doublings=60):
    """Condition weight w with r(w) ≈ r_tar: exponential search for an upper bound, then
    bisection (r falls as w grows). When no w gets there, because pairs with identical
    conditions alone exceed r_tar (discrete conditions), the largest w tried is returned,
    which pairs within conditions."""
    if cond_dist.max() == 0 or admissible_ratio(base, cond_dist, 0.0) <= r_tar:
        return 0.0
    lo, hi = 0.0, w_init or (base.mean() / cond_dist.mean()).item()
    for _ in range(max_doublings):
        if admissible_ratio(base, cond_dist, hi) <= r_tar:
            break
        lo, hi = hi, hi * 2
    else:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if admissible_ratio(base, cond_dist, mid) > r_tar:
            lo = mid
        else:
            hi = mid
    return hi


def nearest_centroid(cond, centroids):
    """Index of each condition's nearest centroid [B]."""
    return torch.cdist(cond, centroids).argmin(dim=1)


def kmeans(x, k, iters=30):
    """Lloyd's K-means with k-means++ seeding on x's device (global RNG). x [N, C] -> [k, C]."""
    centroids = x[torch.randint(x.shape[0], (1,), device=x.device)]
    d2 = _sq_dists(x, centroids).squeeze(1)
    for _ in range(1, k):
        new = x[torch.multinomial(d2 / d2.sum(), 1)]
        centroids = torch.cat([centroids, new])
        d2 = torch.minimum(d2, _sq_dists(x, new).squeeze(1))
    for _ in range(iters):
        assign = nearest_centroid(x, centroids)
        counts = torch.bincount(assign, minlength=k).unsqueeze(1)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        centroids = torch.where(counts > 0, sums / counts.clamp_min(1), centroids)
    return centroids


def condition_aware_pairing(start, goal, method, cond=None, cond_ids=None, manifold='se3',
                            ot_mode='per_frame', r_tar=0.01, w=None, centroids=None,
                            gamma_scale=10.0):
    """Pair noise `start` [B, S, D] with data `goal` by one of `PAIRINGS`.

    cond:      [B, C] condition vector of each data row (the flattened obs tokens).
    cond_ids:  [B] discrete condition ids, for 'ot'.
    w:         the condition weight for 'c2ot_fixed', or the search's warm start for 'c2ot'.
    centroids: [K, C] K-means centroids of the conditions, for 'cluster'.

    Returns the reordered noise (row j now goes with goal[j]) and a dict of what was chosen
    ('w' and the realised admissible ratio 'r' for c2ot, 'gamma' for cluster).
    """
    if method == 'independent':
        return start, {}
    if method == 'ot':
        if ot_mode == 'flat':
            pair_fn = lambda s, g: flat_ot_pairing(s, g, manifold=manifold)
        else:
            pair_fn = lambda s, g: sequence_ot_pairing(s, g, manifold=manifold)
        if cond_ids is None:
            return pair_fn(start, goal), {}
        return _pair_within_conditions(start, goal, cond_ids, pair_fn), {}

    base = _batch_cost(start, goal, manifold, ot_mode)
    info = {}
    if method == 'global':
        cost = base
    elif method in ('c2ot', 'c2ot_fixed'):
        d = _sq_dists(cond, cond)
        if method == 'c2ot':
            w = c2ot_weight(base, d, r_tar, w_init=w)
        cost = base + w * d
        info = {'w': w, 'r': admissible_ratio(base, d, w)}
    elif method == 'cluster':
        # COT Policy gives the noise a random permutation of the batch's centroids. Noise and
        # data are drawn independently here, so noise i's own partner already is one.
        cbar = centroids[nearest_centroid(cond, centroids)]
        d = _sq_dists(cbar, cbar)
        gamma = (gamma_scale * base.mean() / d.mean().clamp_min(1e-12)).item()
        cost = base + gamma * d
        info = {'gamma': gamma}
    else:
        raise ValueError(f"Unknown pairing: {method!r}. Expected one of {PAIRINGS}.")
    return start[_assignment(cost)], info


class Pairer:
    """Serves paired network batches for any of `PAIRINGS`.

    One OT batch of `ot_batch_mult` network batches is drawn from `sampler(n)` -> (start,
    goal, obs, cond_ids), paired at once, shuffled and handed out one network batch at a time.
    A bigger OT batch gives each sample more near-condition partners to pair with [C²OT].
    'c2ot_fixed' fixes w once, at `cond_scale` times the ratio of mean sample cost to mean
    condition distance; 'cluster' scales γ the same way per batch, after fitting K-means
    centroids (K defaults to the OT batch size) once. The papers' scale is 10.
    """

    def __init__(self, method, sampler, batch_size, ot_batch_mult=1, manifold='se3',
                 ot_mode='per_frame', r_tar=0.01, num_clusters=None, cond_scale=10.0):
        if method not in PAIRINGS:
            raise ValueError(f"Unknown pairing: {method!r}. Expected one of {PAIRINGS}.")
        self.method, self.sampler, self.batch_size = method, sampler, batch_size
        self.ot_batch = batch_size * ot_batch_mult
        self.manifold, self.ot_mode, self.r_tar = manifold, ot_mode, r_tar
        self.cond_scale = cond_scale
        self.w, self.centroids, self.info, self.queue = None, None, {}, []
        if method == 'c2ot_fixed':
            self.w = self._fixed_weight(scale=cond_scale)
        elif method == 'cluster':
            self.centroids = self._fit_centroids(num_clusters or self.ot_batch)

    def _fixed_weight(self, n_batches=8, scale=10.0):
        base_sum = cond_sum = 0.0
        for _ in range(n_batches):
            start, goal, obs, _ = self.sampler(self.ot_batch)
            cond = obs.flatten(1)
            base_sum += _batch_cost(start, goal, self.manifold, self.ot_mode).mean().item()
            cond_sum += _sq_dists(cond, cond).mean().item()
        return scale * base_sum / cond_sum

    def _fit_centroids(self, k, n=100_000):
        conds, have = [], 0
        while have < n:
            obs = self.sampler(self.ot_batch)[2]
            conds.append(obs.flatten(1))
            have += obs.shape[0]
        return kmeans(torch.cat(conds)[:n], k)

    def next_batch(self):
        """(start, goal, obs) for one network step; obs is None for unconditional models."""
        if not self.queue:
            start, goal, obs, cond_ids = self.sampler(self.ot_batch)
            cond = obs.flatten(1) if obs is not None else None
            start, self.info = condition_aware_pairing(
                start, goal, self.method, cond=cond, cond_ids=cond_ids, manifold=self.manifold,
                ot_mode=self.ot_mode, r_tar=self.r_tar, w=self.w, centroids=self.centroids,
                gamma_scale=self.cond_scale)
            if self.method == 'c2ot':
                self.w = self.info['w']  # warm start for the next OT batch
            order = torch.randperm(self.ot_batch, device=start.device)
            self.queue = [(start[idx], goal[idx], obs[idx] if obs is not None else None)
                          for idx in order.split(self.batch_size)]
        return self.queue.pop(0)

    def describe(self):
        """The last OT batch's chosen weights, for the training log."""
        return ''.join(f", {k}: {v:.3g}" for k, v in self.info.items())


def generate_interpolated_states(start, goal, n_steps=10, manifold='se3', t=None):
    """
    Build an interpolation between start and goal samples, plus a per-step target velocity.

    For manifold='se3': start/goal are twists [B, 6], interpolated states are quaternion
    poses [B, n_steps, 7], target velocity is twist [B, n_steps, 6].
    For manifold='euclidean': start/goal/interpolated/target all share dim D.

    t: optional times [B, n_steps] to interpolate at, one row per pair (default: the same
       n_steps evenly spaced times in [0, 1] for every pair).

    Returns:
        interp:   [B, n_steps, state_dim]
        t:        [B, n_steps]
        v_target: [B, n_steps, vel_dim]
    """
    batch_size = start.shape[0]
    device = start.device
    dtype = start.dtype

    if t is None:
        t = torch.linspace(0, 1, n_steps, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    n_steps = t.shape[1]

    if manifold == 'se3':
        start_pose = convert_twist_to_pose(start, dt=1.0, return_representation='quat')  # [B, 7]
        goal_pose = convert_twist_to_pose(goal, dt=1.0, return_representation='quat')    # [B, 7]
        twist_s_to_g = compute_twist_between_poses(start_pose, goal_pose, dt=1.0)        # [B, 6]

        start_flat = start_pose.repeat_interleave(n_steps, dim=0)                          # [B*n_steps, 7]
        twist_flat = twist_s_to_g.repeat_interleave(n_steps, dim=0)                        # [B*n_steps, 6]
        t_flat = t.reshape(-1, 1)                                                          # [B*n_steps, 1]

        interp_flat = add_twist_to_pose(start_flat, twist_flat, t_flat)                    # [B*n_steps, 7]
        interp = interp_flat.reshape(batch_size, n_steps, 7)

        v_target = twist_s_to_g.unsqueeze(1).repeat(1, n_steps, 1)                         # [B, n_steps, 6]
    elif manifold == 'euclidean':
        # Linear interp: x_t = start + t*(goal - start);  v_target = goal - start (constant in t)
        diff = compute_velocity_between_states(start, goal, dt=1.0)                        # [B, D]
        # Broadcast t against batch: [B, n_steps, D] = start[:,None,:] + t[:,:,None]*diff[:,None,:]
        interp = start.unsqueeze(1) + t.unsqueeze(-1) * diff.unsqueeze(1)                  # [B, n_steps, D]
        v_target = diff.unsqueeze(1).repeat(1, n_steps, 1)                                 # [B, n_steps, D]
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}.")

    return interp, t, v_target


# Backward-compat alias preserving the original signature/name
def generate_interpolated_poses(start_poses, goal_poses, n_steps=10):
    return generate_interpolated_states(start_poses, goal_poses, n_steps=n_steps, manifold='se3')


def _build_cond_mask(batch_size, num_tokens, use_cfg, device):
    """[batch, M] bool mask; True = replace that obs token with the null token.

    Each token is dropped independently with p=0.1; with CFG, whole samples are also
    dropped (unconditional) with p=0.1.
    """
    indep_mask = torch.rand(batch_size, num_tokens, device=device) < 0.1
    if use_cfg:
        uncond_mask = torch.rand(batch_size, device=device) < 0.1
        return indep_mask | uncond_mask.unsqueeze(-1)
    # Without CFG, per-token dropout must never blank a whole sample: with a single obs
    # token (MNIST) it otherwise *is* unconditional dropout, and NOCFG would still learn
    # the null branch.
    return indep_mask & ~indep_mask.all(dim=-1, keepdim=True)


# Training-time densities of the flow time t in [0, 1] (t = 0 is noise, t = 1 data), an
# ablation axis:
#   uniform       Rectified Flow / I-CFM
#   logit_normal  sigmoid of N(0, 1), weighting the middle of the path [SD3, Esser et al. 2024]
#   beta          π0's density, weighting the noisy end: t = 0.999·(1 − u), u ~ Beta(1.5, 1)
#                 [Black et al. 2024]
T_DISTS = ('uniform', 'logit_normal', 'beta')


def sample_times(n, dist='uniform', device='cpu', dtype=torch.float32):
    """`n` training times drawn from one of `T_DISTS`."""
    if dist == 'uniform':
        return torch.rand(n, device=device, dtype=dtype)
    if dist == 'logit_normal':
        return torch.sigmoid(torch.randn(n, device=device, dtype=dtype))
    if dist == 'beta':
        u = torch.distributions.Beta(torch.tensor(1.5, device=device),
                                     torch.tensor(1.0, device=device)).sample((n,))
        return (0.999 * (1 - u)).to(dtype)
    raise ValueError(f"Unknown time distribution: {dist!r}. Expected one of {T_DISTS}.")


class EMA:
    """Exponential moving average of a model's parameters, the weights to sample from [as in
    DiT, SiT, EDM and SD3]. The decay warms up as min(decay, (1 + n) / (10 + n)) over the first
    updates, so the average isn't dominated by the random initialisation."""

    def __init__(self, model, decay):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.detach(), 1 - decay)

    def state_dict(self, model):
        """`model`'s state_dict with every parameter replaced by its average."""
        return {**model.state_dict(), **self.shadow}


def pose_normalizer_stats(sampler, n=50_000, seed=0, chunk=1024):
    """Position and velocity statistics for an SE(3) model's `set_normalizer`: the per-axis mean
    and std of the positions on the training paths (uniform t) and of the twist targets, over
    `n` randomly paired (start, goal) draws from `sampler(m) -> (start, goal, obs, cond_ids)`.

    Drawn under `seed` inside a forked RNG, so the training run's random stream is unchanged.
    """
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        starts, goals, have = [], [], 0
        while have < n:
            start, goal, _, _ = sampler(chunk)
            starts.append(start.reshape(-1, 6))
            goals.append(goal.reshape(-1, 6))
            have += starts[-1].shape[0]
        start, goal = torch.cat(starts)[:n], torch.cat(goals)[:n]
        t = torch.rand(start.shape[0], 1, device=start.device, dtype=start.dtype)
        interp, _, v = generate_interpolated_states(start, goal, manifold='se3', t=t)
    pos, vel = interp[:, 0, :3], v[:, 0]
    return {'pos_mean': pos.mean(0).cpu(), 'pos_std': pos.std(0).clamp_min(1e-3).cpu(),
            'vel_mean': vel.mean(0).cpu(), 'vel_std': vel.std(0).clamp_min(1e-3).cpu()}


def _run_flow_matching_step(model, optimizer, start, goal, obs, n_steps,
                            state_dim, vel_dim, manifold, use_ot, use_cfg, device,
                            ot_mode='per_frame', scheduler=None, cond_ids=None,
                            t_dist='uniform', ema=None, grad_clip=None):
    """
    Shared core: interpolate -> forward+loss -> backprop.

    start/goal: [B, S, D_start] where D_start matches the manifold's natural input
                (6 for SE(3) twists, vel_dim==state_dim for Euclidean).
    obs:        [B, M, obs_dim] or None.
    cond_ids:   [B] condition index per sample (conditional models), or None. With OT,
                pairing then stays within each condition.

    Flow times are drawn from `t_dist` (`T_DISTS`). SE(3) pairs each get n_steps times, so the
    batch fans out to [B*n_steps, S, D] and every geodesic is reused n_steps times; Euclidean
    samples get one time each.
    ot_mode:
        'per_frame' — independent OT permutation per sequence index (SE(3) trainer).
        'flat'      — single permutation across the flattened [B, S*D] sample
                      so spatial coherence is preserved (image trainer).
    ema:        an `EMA` to update after the optimiser step, or None.
    grad_clip:  max gradient norm, or None.
    """
    B, S, _ = start.shape

    if use_ot:
        if ot_mode == 'flat':
            pair_fn = lambda s, g: flat_ot_pairing(s, g, manifold=manifold)
        elif ot_mode == 'per_frame':
            pair_fn = lambda s, g: sequence_ot_pairing(s, g, manifold=manifold)
        else:
            raise ValueError(f"Unknown ot_mode: {ot_mode!r}")
        if cond_ids is None:
            start = pair_fn(start, goal)
        else:
            start = _pair_within_conditions(start, goal, cond_ids, pair_fn)

    if manifold == 'euclidean':
        t_input = sample_times(B, t_dist, device=device, dtype=start.dtype)  # [B]
        diff = goal - start                                              # [B, S, D]
        x_t = start + t_input.view(B, 1, 1) * diff                        # [B, S, D]
        v_target = diff                                                   # [B, S, D]

        if obs is not None:
            cond_mask = _build_cond_mask(B, obs.shape[1], use_cfg, device)
        else:
            cond_mask = None
    elif manifold == 'se3':
        flat_start = start.reshape(B * S, start.shape[-1])
        flat_goal = goal.reshape(B * S, goal.shape[-1])
        # One set of times per sample, shared by its sequence positions
        t = sample_times(B * n_steps, t_dist, device=device, dtype=start.dtype).view(B, n_steps)

        interp_flat, t_flat, v_flat = generate_interpolated_states(
            flat_start, flat_goal, manifold=manifold, t=t.repeat_interleave(S, dim=0)
        )
        # interp_flat: [B*S, n_steps, state_dim]
        # v_flat:      [B*S, n_steps, vel_dim]

        x_t = interp_flat.view(B, S, n_steps, state_dim).transpose(1, 2)   # [B, n_steps, S, state_dim]
        v_target = v_flat.view(B, S, n_steps, vel_dim).transpose(1, 2)     # [B, n_steps, S, vel_dim]

        x_t = x_t.reshape(B * n_steps, S, state_dim)
        v_target = v_target.reshape(B * n_steps, S, vel_dim)
        t_input = t.reshape(B * n_steps)

        if obs is not None:
            obs = obs.repeat_interleave(n_steps, dim=0)
            cond_mask = _build_cond_mask(obs.shape[0], obs.shape[1], use_cfg, device)
        else:
            cond_mask = None
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}.")

    if isinstance(model, ConditionalFlowMatchingTransformerModel):
        loss = model.cfm_loss(x_t, t_input, v_target, obs, cond_mask=cond_mask, reduction='mean')
    else:
        loss = model.cfm_loss(x_t, t_input, v_target, reduction='mean')

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)

    return loss.item()


def train_one_minibatch(model, optimizer, batch_size, n_steps, start_dist_params, goal_dist_params,
                        action_dist_params, seq_len=1, use_ot=True, use_cfg=True, device='cpu',
                        scheduler=None, t_dist='uniform', ema=None, grad_clip=None):
    """Pose-flow minibatch: samples start/goal twists from mode distributions (manifold='se3')."""
    model.train()

    start_poses, goal_poses, obs, cond_ids = sample_pose_batch(
        batch_size, start_dist_params, goal_dist_params, action_dist_params,
        seq_len=seq_len, device=device)

    return _run_flow_matching_step(
        model, optimizer,
        start=start_poses, goal=goal_poses, obs=obs,
        n_steps=n_steps,
        state_dim=7, vel_dim=6,
        manifold='se3', use_ot=use_ot, use_cfg=use_cfg, device=device,
        ot_mode='per_frame', scheduler=scheduler, cond_ids=cond_ids,
        t_dist=t_dist, ema=ema, grad_clip=grad_clip,
    )


def train_one_paired_minibatch(model, optimizer, pairer, n_steps, use_cfg=True, device='cpu',
                               scheduler=None, manifold='se3', t_dist='uniform', ema=None,
                               grad_clip=None):
    """One step on the next network batch from `pairer` (already paired). SE(3) pose flows use
    n_steps times per pair, as in `train_one_minibatch`; Euclidean ones one time per sample."""
    model.train()
    start, goal, obs = pairer.next_batch()
    dim = goal.shape[-1]
    state_dim, vel_dim = (7, 6) if manifold == 'se3' else (dim, dim)
    return _run_flow_matching_step(
        model, optimizer, start=start, goal=goal, obs=obs, n_steps=n_steps,
        state_dim=state_dim, vel_dim=vel_dim, manifold=manifold, use_ot=False, use_cfg=use_cfg,
        device=device, ot_mode=pairer.ot_mode, scheduler=scheduler,
        t_dist=t_dist, ema=ema, grad_clip=grad_clip,
    )


def sample_pose_batch(batch_size, start_dist_params, goal_dist_params, action_dist_params,
                      seq_len=1, device='cpu'):
    """A pose batch from the mode distributions: start and goal twists [B, seq_len, 6], laid
    out mode by mode, with obs tokens [B, M, obs_dim] and condition ids [B] for conditional
    tasks (None otherwise)."""
    if batch_size % len(goal_dist_params['mu']) != 0:
        raise ValueError("Batch size must be divisible by the number of goal distribution modes.")

    if action_dist_params is not None:
        if batch_size % len(action_dist_params['mu']) != 0:
            raise ValueError("Batch size must be divisible by the number of action distribution modes.")
        if len(goal_dist_params['mu']) != len(action_dist_params['mu']):
            raise ValueError("Number of goal and action distribution modes must match for conditional models.")

    # Sample start poses: seq_len independent draws → [batch_size, seq_len, 6]
    start_poses = torch.stack([
        sample_random_twist(
            batch_size=batch_size,
            mu=start_dist_params['mu'],
            sigma=start_dist_params['sigma'],
            device=device
        )
        for _ in range(seq_len)
    ], dim=1)

    # Sample goal poses per mode → [batch_size, seq_len, 6]
    goal_seqs = []
    for _ in range(seq_len):
        mode_samples = []
        for i in range(len(goal_dist_params['mu'])):
            mode_samples.append(sample_random_twist(
                batch_size=batch_size // len(goal_dist_params['mu']),
                mu=goal_dist_params['mu'][i],
                sigma=goal_dist_params['sigma'][i],
                device=device
            ))
        goal_seqs.append(torch.cat(mode_samples, dim=0))
    goal_poses = torch.stack(goal_seqs, dim=1)

    # Sample observations from action distributions (conditional only)
    if action_dist_params is not None:
        obs_list = []
        num_modes = len(action_dist_params['mu'])
        bs_per_mode = batch_size // num_modes

        for i in range(num_modes):
            mu_tensor = torch.tensor(action_dist_params['mu'][i], dtype=torch.float32, device=device)
            sigma_tensor = torch.tensor(action_dist_params['sigma'][i], dtype=torch.float32, device=device)
            eps = torch.randn(bs_per_mode, mu_tensor.shape[0], mu_tensor.shape[1], device=device)
            obs_i = mu_tensor.unsqueeze(0) + eps * sigma_tensor.unsqueeze(0)
            obs_list.append(obs_i)
        obs = torch.cat(obs_list, dim=0)
        # Goals and obs are both laid out mode by mode, so row i belongs to mode i // bs_per_mode;
        # modes named by the same tokens share a condition.
        cond_ids = torch.tensor(_mode_condition_ids(action_dist_params['mu']),
                                device=device).repeat_interleave(bs_per_mode)
    else:
        obs = None
        cond_ids = None

    return start_poses, goal_poses, obs, cond_ids


def train_one_minibatch_image(model, optimizer, dataloader_iter, n_steps,
                              num_classes=10, use_ot=True, use_cfg=True, device='cpu',
                              patch_encode=None, ot_mode='flat', scheduler=None,
                              t_dist='uniform', ema=None, grad_clip=None):
    """
    Image-flow minibatch: pulls a batch from `dataloader_iter`, builds Gaussian noise as
    the start, and runs the shared training step on the Euclidean manifold.

    Args:
        dataloader_iter: an iterator over (images, labels) yielding image tensors of shape
            [batch, 1, H, W] (e.g. MNIST). Should be wrapped so callers can call next() and
            reset on StopIteration externally.
        patch_encode: callable mapping [B, 1, H, W] -> [B, S, D]. Required.

    Returns:
        loss: float
        new_iter: iterator (possibly re-created if exhausted)
    """
    model.train()
    if patch_encode is None:
        raise ValueError("patch_encode callable is required for train_one_minibatch_image.")

    try:
        images, labels = next(dataloader_iter)
    except StopIteration:
        return None, None  # caller should rebuild iterator

    images = images.to(device)
    labels = labels.to(device)

    goal = patch_encode(images)              # [B, S, D]
    start = torch.randn_like(goal)           # [B, S, D]

    B, _, D = goal.shape

    is_conditional = isinstance(model, ConditionalFlowMatchingTransformerModel)
    if is_conditional:
        # One-hot class label as a single obs token: [B, 1, num_classes]
        obs = torch.zeros(B, num_classes, device=device)
        obs.scatter_(1, labels.view(-1, 1), 1.0)
        obs = obs.unsqueeze(1)
    else:
        obs = None

    loss = _run_flow_matching_step(
        model, optimizer,
        start=start, goal=goal, obs=obs,
        n_steps=n_steps,
        state_dim=D, vel_dim=D,
        manifold='euclidean', use_ot=use_ot, use_cfg=use_cfg, device=device,
        ot_mode=ot_mode, scheduler=scheduler, cond_ids=labels if is_conditional else None,
        t_dist=t_dist, ema=ema, grad_clip=grad_clip,
    )
    return loss, dataloader_iter


def train_one_epoch(model, optimizer, num_batches, batch_size, n_steps, start_dist_params,
                    goal_dist_params, action_dist_params, seq_len=1, use_ot=True, use_cfg=True,
                    device='cpu', manifold='se3', dataloader=None, num_classes=10,
                    patch_encode=None, ot_mode='flat', scheduler=None, pairer=None,
                    t_dist='uniform', ema=None, grad_clip=None):
    total_loss = 0.0
    completed = 0
    step_opts = {'scheduler': scheduler, 't_dist': t_dist, 'ema': ema, 'grad_clip': grad_clip}

    if manifold == 'euclidean':
        if dataloader is None:
            raise ValueError("dataloader is required when manifold='euclidean'.")
        data_iter = iter(dataloader)
        for batch_idx in range(num_batches):
            loss, data_iter = train_one_minibatch_image(
                model, optimizer, data_iter, n_steps,
                num_classes=num_classes, use_ot=use_ot, use_cfg=use_cfg,
                device=device, patch_encode=patch_encode,
                ot_mode=ot_mode, **step_opts,
            )
            if loss is None:
                # Iterator exhausted — restart and retry this batch index
                data_iter = iter(dataloader)
                loss, data_iter = train_one_minibatch_image(
                    model, optimizer, data_iter, n_steps,
                    num_classes=num_classes, use_ot=use_ot, use_cfg=use_cfg,
                    device=device, patch_encode=patch_encode,
                    ot_mode=ot_mode, **step_opts,
                )
            total_loss += loss
            completed += 1
            if (batch_idx + 1) % 10 == 0:
                lr_str = ''
                if scheduler is not None:
                    lr_str = f", LR: {optimizer.param_groups[0]['lr']:.2e}"
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}{lr_str}")
    else:
        for batch_idx in range(num_batches):
            if pairer is None:
                loss = train_one_minibatch(
                    model, optimizer, batch_size, n_steps,
                    start_dist_params, goal_dist_params, action_dist_params,
                    seq_len=seq_len, use_ot=use_ot, use_cfg=use_cfg, device=device,
                    **step_opts,
                )
            else:
                loss = train_one_paired_minibatch(
                    model, optimizer, pairer, n_steps, use_cfg=use_cfg, device=device,
                    **step_opts,
                )
            total_loss += loss
            completed += 1
            if (batch_idx + 1) % 10 == 0:
                extra = pairer.describe() if pairer is not None else ''
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}{extra}")

    return total_loss / max(completed, 1)


def train(model, optimizer, num_epochs, num_batches_per_epoch, batch_size, n_steps,
          start_dist_params=None, goal_dist_params=None, action_dist_params=None,
          seq_len=1, use_ot=True, use_cfg=True, device='cpu', save_path=None,
          manifold='se3', dataloader=None, num_classes=10, patch_encode=None,
          ot_mode='flat', scheduler=None, pairer=None, t_dist='uniform', ema=None,
          grad_clip=None):
    """
    Full training loop. SE(3) (default) trains from distribution params; 'euclidean' trains
    from a torch DataLoader yielding (images, labels). An SE(3) run given a `Pairer` draws
    its (already paired) batches from it instead of from the distribution params.

    t_dist: training-time density (`T_DISTS`); ema: an `EMA` of the weights, saved with every
    checkpoint as 'ema_state_dict'; grad_clip: max gradient norm.
    """
    loss_history = []

    print(f"Starting training for {num_epochs} epochs (manifold={manifold})...")
    print(f"Batches per epoch: {num_batches_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Interpolation steps: {n_steps}")
    print(f"Device: {device}")
    print()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        avg_loss = train_one_epoch(
            model, optimizer, num_batches_per_epoch, batch_size, n_steps,
            start_dist_params, goal_dist_params, action_dist_params,
            seq_len=seq_len, use_ot=use_ot, use_cfg=use_cfg, device=device,
            manifold=manifold, dataloader=dataloader, num_classes=num_classes,
            patch_encode=patch_encode,
            ot_mode=ot_mode, scheduler=scheduler, pairer=pairer,
            t_dist=t_dist, ema=ema, grad_clip=grad_clip,
        )

        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.6f}\n")

        if save_path and (epoch + 1) % 10 == 0:
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Created checkpoint directory: {checkpoint_dir}")
            checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
            extra = {'ema_state_dict': ema.state_dict(model)} if ema is not None else {}
            model.save_checkpoint(
                filepath=checkpoint_path,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                **extra,
            )
            print(f"Checkpoint saved to {checkpoint_path}\n")

    print("Training completed!")
    return loss_history
