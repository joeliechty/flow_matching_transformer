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
    """Apply OT pairing independently per sequence frame."""
    chunk_size = start_seq.shape[1]
    paired_start = torch.zeros_like(start_seq)

    for i in range(chunk_size):
        paired_start[:, i, :] = optimal_transport_pairing(
            start_seq[:, i, :],
            goal_seq[:, i, :],
            manifold=manifold,
        )
    return paired_start


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

    row_ind, col_ind = linear_sum_assignment(dists.detach().cpu().numpy())
    sorted_indices = np.argsort(col_ind)
    return start[row_ind[sorted_indices]]


# Backward-compat alias for any external callers
def geodesic_optimal_transport_pairing(start_poses, goal_poses):
    return optimal_transport_pairing(start_poses, goal_poses, manifold='se3')


def generate_interpolated_states(start, goal, n_steps=10, manifold='se3'):
    """
    Build an interpolation between start and goal samples, plus a per-step target velocity.

    For manifold='se3': start/goal are twists [B, 6], interpolated states are quaternion
    poses [B, n_steps, 7], target velocity is twist [B, n_steps, 6].
    For manifold='euclidean': start/goal/interpolated/target all share dim D.

    Returns:
        interp:   [B, n_steps, state_dim]
        t:        [B, n_steps]
        v_target: [B, n_steps, vel_dim]
    """
    batch_size = start.shape[0]
    device = start.device
    dtype = start.dtype

    t = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)

    if manifold == 'se3':
        start_pose = convert_twist_to_pose(start, dt=1.0, return_representation='quat')  # [B, 7]
        goal_pose = convert_twist_to_pose(goal, dt=1.0, return_representation='quat')    # [B, 7]
        twist_s_to_g = compute_twist_between_poses(start_pose, goal_pose, dt=1.0)        # [B, 6]

        start_flat = start_pose.repeat_interleave(n_steps, dim=0)                          # [B*n_steps, 7]
        twist_flat = twist_s_to_g.repeat_interleave(n_steps, dim=0)                        # [B*n_steps, 6]
        t_flat = t.repeat(batch_size).unsqueeze(-1)                                        # [B*n_steps, 1]

        interp_flat = add_twist_to_pose(start_flat, twist_flat, t_flat)                    # [B*n_steps, 7]
        interp = interp_flat.reshape(batch_size, n_steps, 7)

        v_target = twist_s_to_g.unsqueeze(1).repeat(1, n_steps, 1)                         # [B, n_steps, 6]
    elif manifold == 'euclidean':
        # Linear interp: x_t = start + t*(goal - start);  v_target = goal - start (constant in t)
        diff = compute_velocity_between_states(start, goal, dt=1.0)                        # [B, D]
        # Broadcast t against batch: [B, n_steps, D] = start[:,None,:] + t[None,:,None]*diff[:,None,:]
        t_b = t.view(1, n_steps, 1)
        interp = start.unsqueeze(1) + t_b * diff.unsqueeze(1)                              # [B, n_steps, D]
        v_target = diff.unsqueeze(1).repeat(1, n_steps, 1)                                 # [B, n_steps, D]
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}.")

    t_out = t.unsqueeze(0).repeat(batch_size, 1)                                            # [B, n_steps]
    return interp, t_out, v_target


# Backward-compat alias preserving the original signature/name
def generate_interpolated_poses(start_poses, goal_poses, n_steps=10):
    return generate_interpolated_states(start_poses, goal_poses, n_steps=n_steps, manifold='se3')


def _run_flow_matching_step(model, optimizer, start, goal, obs, n_steps,
                            state_dim, vel_dim, manifold, use_ot, use_cfg, device):
    """
    Shared core: interpolate -> reshape -> forward+loss -> backprop.

    start/goal: [B, S, D_start] where D_start matches the manifold's natural input
                (6 for SE(3) twists, vel_dim==state_dim for Euclidean).
    obs:        [B, M, obs_dim] or None.
    """
    B, S, _ = start.shape

    if use_ot:
        start = sequence_ot_pairing(start, goal, manifold=manifold)

    flat_start = start.reshape(B * S, start.shape[-1])
    flat_goal = goal.reshape(B * S, goal.shape[-1])

    interp_flat, t_flat, v_flat = generate_interpolated_states(
        flat_start, flat_goal, n_steps=n_steps, manifold=manifold
    )
    # interp_flat: [B*S, n_steps, state_dim]
    # v_flat:      [B*S, n_steps, vel_dim]

    x_t = interp_flat.view(B, S, n_steps, state_dim).transpose(1, 2)   # [B, n_steps, S, state_dim]
    v_target = v_flat.view(B, S, n_steps, vel_dim).transpose(1, 2)     # [B, n_steps, S, vel_dim]
    t = t_flat.view(B, S, n_steps).transpose(1, 2)                     # [B, n_steps, S]

    x_t = x_t.reshape(B * n_steps, S, state_dim)
    v_target = v_target.reshape(B * n_steps, S, vel_dim)
    t_input = t[:, :, 0].reshape(B * n_steps)

    if obs is not None:
        obs = obs.repeat_interleave(n_steps, dim=0)
        B_steps, M, _ = obs.shape
        indep_mask = torch.rand(B_steps, M, device=device) < 0.1
        if use_cfg:
            uncond_mask = torch.rand(B_steps, device=device) < 0.1
            cond_mask = indep_mask | uncond_mask.unsqueeze(-1)
        else:
            cond_mask = indep_mask
    else:
        cond_mask = None

    if isinstance(model, ConditionalFlowMatchingTransformerModel):
        loss = model.cfm_loss(x_t, t_input, v_target, obs, cond_mask=cond_mask, reduction='mean')
    else:
        loss = model.cfm_loss(x_t, t_input, v_target, reduction='mean')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_one_minibatch(model, optimizer, batch_size, n_steps, start_dist_params, goal_dist_params,
                        action_dist_params, seq_len=1, use_ot=True, use_cfg=True, device='cpu'):
    """Pose-flow minibatch: samples start/goal twists from mode distributions (manifold='se3')."""
    model.train()

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
    else:
        obs = None

    return _run_flow_matching_step(
        model, optimizer,
        start=start_poses, goal=goal_poses, obs=obs,
        n_steps=n_steps,
        state_dim=7, vel_dim=6,
        manifold='se3', use_ot=use_ot, use_cfg=use_cfg, device=device,
    )


def train_one_minibatch_image(model, optimizer, dataloader_iter, n_steps,
                              num_classes=10, use_ot=True, use_cfg=True, device='cpu',
                              patch_encode=None):
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
    )
    return loss, dataloader_iter


def train_one_epoch(model, optimizer, num_batches, batch_size, n_steps, start_dist_params,
                    goal_dist_params, action_dist_params, seq_len=1, use_ot=True, use_cfg=True,
                    device='cpu', manifold='se3', dataloader=None, num_classes=10,
                    patch_encode=None):
    total_loss = 0.0
    completed = 0

    if manifold == 'euclidean':
        if dataloader is None:
            raise ValueError("dataloader is required when manifold='euclidean'.")
        data_iter = iter(dataloader)
        for batch_idx in range(num_batches):
            loss, data_iter = train_one_minibatch_image(
                model, optimizer, data_iter, n_steps,
                num_classes=num_classes, use_ot=use_ot, use_cfg=use_cfg,
                device=device, patch_encode=patch_encode,
            )
            if loss is None:
                # Iterator exhausted — restart and retry this batch index
                data_iter = iter(dataloader)
                loss, data_iter = train_one_minibatch_image(
                    model, optimizer, data_iter, n_steps,
                    num_classes=num_classes, use_ot=use_ot, use_cfg=use_cfg,
                    device=device, patch_encode=patch_encode,
                )
            total_loss += loss
            completed += 1
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}")
    else:
        for batch_idx in range(num_batches):
            loss = train_one_minibatch(
                model, optimizer, batch_size, n_steps,
                start_dist_params, goal_dist_params, action_dist_params,
                seq_len=seq_len, use_ot=use_ot, use_cfg=use_cfg, device=device,
            )
            total_loss += loss
            completed += 1
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}")

    return total_loss / max(completed, 1)


def train(model, optimizer, num_epochs, num_batches_per_epoch, batch_size, n_steps,
          start_dist_params=None, goal_dist_params=None, action_dist_params=None,
          seq_len=1, use_ot=True, use_cfg=True, device='cpu', save_path=None,
          manifold='se3', dataloader=None, num_classes=10, patch_encode=None):
    """
    Full training loop. SE(3) (default) trains from distribution params; 'euclidean' trains
    from a torch DataLoader yielding (images, labels).
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
        )

        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.6f}\n")

        if save_path and (epoch + 1) % 10 == 0:
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Created checkpoint directory: {checkpoint_dir}")
            checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
            model.save_checkpoint(
                filepath=checkpoint_path,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss
            )
            print(f"Checkpoint saved to {checkpoint_path}\n")

    print("Training completed!")
    return loss_history
