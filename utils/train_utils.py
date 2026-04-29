import torch
import os
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel

def sequence_ot_pairing(start_poses_seq, goal_poses_seq):
    # start_poses_seq shape: [batch_size, chunk_size, 6]
    chunk_size = start_poses_seq.shape[1]
    paired_start = torch.zeros_like(start_poses_seq)
    
    for i in range(chunk_size):
        # Independently pair Frame i's noise to Frame i's target across the batch
        paired_start[:, i, :] = geodesic_optimal_transport_pairing(
            start_poses_seq[:, i, :], 
            goal_poses_seq[:, i, :]
        )
    return paired_start

def geodesic_optimal_transport_pairing(start_poses, goal_poses):
    """
    Compute optimal transport pairing between start and goal poses on SE(3) manifold.
    
    Args:
        start_poses: Tensor of shape [batch_size, 6] representing start poses as twists.
        goal_poses: Tensor of shape [batch_size, 6] representing goal poses as twists.

    Returns:
        paired_goal_poses: Tensor of shape [batch_size, 6], reordered goal poses for optimal transport.
    """
    # Convert twists to poses (use quat for better batch handling)
    start_pose = convert_twist_to_pose(start_poses, dt=1.0, return_representation='quat')  # [batch_size, 7]
    goal_pose = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='quat')    # [batch_size, 7]
    
    batch_size = start_pose.shape[0]
    
    # Compute pairwise geodesic distances by repeating tensors to have matching batch dimensions
    # Repeat start for each goal: [batch_size, batch_size, 7]
    start_repeated = start_pose.unsqueeze(1).repeat(1, batch_size, 1)  # [batch_size, batch_size, 7]
    # Repeat goal for each start: [batch_size, batch_size, 7]
    goal_repeated = goal_pose.unsqueeze(0).repeat(batch_size, 1, 1)    # [batch_size, batch_size, 7]
    
    # Flatten to compute all pairwise twists
    start_flat = start_repeated.reshape(batch_size * batch_size, 7)  # [batch_size^2, 7]
    goal_flat = goal_repeated.reshape(batch_size * batch_size, 7)    # [batch_size^2, 7]
    
    # Compute all pairwise twists at once
    twist_pairwise_flat = compute_twist_between_poses(start_flat, goal_flat, dt=1.0)  # [batch_size^2, 6]
    
    # Compute distances (L2 norm of each twist)
    dists_flat = torch.norm(twist_pairwise_flat, p=2, dim=-1)  # [batch_size^2]
    dists = dists_flat.reshape(batch_size, batch_size)  # [batch_size, batch_size]
    
    # Solve linear sum assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(dists.cpu().numpy())
    
    # We want to reorder start_poses so that new_start[i] pairs with goal[i].
    # This means we need to sort the assignment by col_ind.
    sorted_indices = np.argsort(col_ind)
    
    # Apply the sorted indices to the start poses
    paired_start_poses = start_poses[row_ind[sorted_indices]]
    
    return paired_start_poses

def generate_interpolated_poses(start_poses, goal_poses, n_steps=10):
    """
    Generate interpolated poses between start and goal poses using twist representation.
    Properly handles interpolation on the SE(3) manifold.
    
    Args:
        start_poses: Tensor of shape [batch_size, 6] representing start poses as twists.
        goal_poses: Tensor of shape [batch_size, 6] representing goal poses as twists.
        n_steps: Number of interpolation steps.

    Returns:
        interpolated_poses: Tensor of shape [batch_size, n_steps, 9], representing interpolated poses in ortho6d format.
        t: Tensor of shape [n_steps], interpolation coefficients from 0 to 1.
        twist_start_to_goal: Tensor of shape [batch_size, 6], the twist from start to goal poses.
    """
    batch_size = start_poses.shape[0]
    device = start_poses.device
    dtype = start_poses.dtype
    
    # Convert twists to poses (using ortho6d representation for continuous manifold)
    start_pose = convert_twist_to_pose(start_poses, dt=1.0, return_representation='quat')  # [batch_size, 7]
    goal_pose = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='quat')    # [batch_size, 7]
    
    # Compute the twist that takes us from start to goal in dt=1.0
    twist_start_to_goal = compute_twist_between_poses(start_pose, goal_pose, dt=1.0)  # [batch_size, 6]
    
    # Create interpolation coefficients from 0 to 1
    t = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)  # [n_steps]
    
    # Use repeat_interleave to efficiently create flattened batch tensors
    # Repeat each batch element n_steps times: [elem0, elem0, ..., elem1, elem1, ...]
    start_flat = start_pose.repeat_interleave(n_steps, dim=0)  # [batch_size * n_steps, 7]
    twist_flat = twist_start_to_goal.repeat_interleave(n_steps, dim=0)  # [batch_size * n_steps, 6]
    
    # Repeat entire t sequence for each batch: [t0, t1, ..., tn, t0, t1, ..., tn, ...]
    t_flat = t.repeat(batch_size).unsqueeze(-1)  # [batch_size * n_steps, 1]
    
    # Apply fraction of twist to start pose for each interpolation step
    # This properly integrates on the SE(3) manifold
    interpolated_flat = add_twist_to_pose(start_flat, twist_flat, t_flat)  # [batch_size * n_steps, 7]
    
    # Reshape back to [batch_size, n_steps, 7]
    interpolated_poses = interpolated_flat.reshape(batch_size, n_steps, 7)

    # repeat t for batch size
    t = t.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, n_steps]

    # repeat twist_start_to_goal for n_steps
    twist_start_to_goal = twist_start_to_goal.unsqueeze(1).repeat(1, n_steps, 1)  # [batch_size, n_steps, 6]

    return interpolated_poses, t, twist_start_to_goal

def train_one_minibatch(model, optimizer, batch_size, n_steps, start_dist_params, goal_dist_params, action_dist_params, seq_len=1, use_ot=True, device='cpu'):
    """
    Train for one minibatch.
    
    Args:
        model: FlowMatchingTransformerModel instance
        optimizer: torch optimizer
        batch_size: number of samples in minibatch
        n_steps: number of interpolation steps per trajectory
        start_dist_params: dict with 'mu' and 'sigma' for start pose distribution
        goal_dist_params: dict with 'mu' and 'sigma' for goal pose distribution
        device: device to run on
        
    Returns:
        loss: scalar loss value
    """
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

    # Sample goal poses: seq_len independent draws per mode → [batch_size, seq_len, 6]
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
    goal_poses = torch.stack(goal_seqs, dim=1)  # [batch_size, seq_len, 6]

    # Sample actions from action distributions (only for conditional models)
    if action_dist_params is not None:
        obs_list = []
        num_modes = len(action_dist_params['mu'])
        bs_per_mode = batch_size // num_modes

        for i in range(num_modes):
            # mu_tensor shape: [M, 6]
            mu_tensor = torch.tensor(action_dist_params['mu'][i], dtype=torch.float32, device=device)
            sigma_tensor = torch.tensor(action_dist_params['sigma'][i], dtype=torch.float32, device=device)

            # Sample noise for [bs_per_mode, M, 6]
            eps = torch.randn(bs_per_mode, mu_tensor.shape[0], mu_tensor.shape[1], device=device)
            obs_i = mu_tensor.unsqueeze(0) + eps * sigma_tensor.unsqueeze(0)
            obs_list.append(obs_i)
        
        obs = torch.cat(obs_list, dim=0) # [batch_size, M, 6]

    else:
        obs = None

    # Get sequence length (S). Assume start/goal are [batch_size, S, 6]
    B, S, _ = start_poses.shape

    # optimal transport pairing
    if use_ot:
        start_poses = sequence_ot_pairing(start_poses, goal_poses)  # [batch_size, 6]

    flat_start = start_poses.reshape(B * S, 6)
    flat_goal = goal_poses.reshape(B * S, 6)

    # Compute interpolated poses, time steps, and target vector fields (twists)
    interp_flat, t_flat, twist_flat = generate_interpolated_poses(
        flat_start, flat_goal, n_steps=n_steps
    )
    # interp_flat shape: [B * S, n_steps, 7]
    # twist_flat shape: [B * S, n_steps, 6]

    # Reshape and permute to [Batch * n_steps, SeqLen, Dim]
    x_t = interp_flat.view(B, S, n_steps, 7).transpose(1, 2)       # [B, n_steps, S, 7]
    v_target = twist_flat.view(B, S, n_steps, 6).transpose(1, 2)   # [B, n_steps, S, 6]
    t = t_flat.view(B, S, n_steps).transpose(1, 2)                 # [B, n_steps, S]

    # Flatten the Flow Time (n_steps) into the Batch dimension
    x_t = x_t.reshape(B * n_steps, S, 7)         # The Transformer input
    v_target = v_target.reshape(B * n_steps, S, 6) # The Twist targets

    # Time 't' is identical across the sequence (S), so we just take index 0
    t_input = t[:, :, 0].reshape(B * n_steps)    # [B * n_steps]

    # For conditional models, repeat observations and flatten, add sequence dimension
    if obs is not None:
        # Expand [batch_size, M, 6] to [batch_size * n_steps, M, 6]
        obs = obs.repeat_interleave(n_steps, dim=0)

        # Independent conditioning mask
        B_steps, M, _ = obs.shape

        # Independent dropout (10% chance to drop each specific observation token)
        indep_mask = torch.rand(B_steps, M, device=device) < 0.1  # [B_steps, M]

        # Unconditional dropout (10% chance to drop ALL observation tokens in a batch)
        uncond_mask = torch.rand(B_steps, device=device) < 0.1

        # combine: Token is masked if individually dropped OR batch is fully dropped
        cond_mask = indep_mask | uncond_mask.unsqueeze(-1)  # [B_steps, M]
    else:
        cond_mask = None

    # Compute loss (conditional vs non-conditional)
    if isinstance(model, ConditionalFlowMatchingTransformerModel):
        loss = model.cfm_loss(x_t, t_input, v_target, obs, cond_mask=cond_mask, reduction='mean')
    else:
        loss = model.cfm_loss(x_t, t_input, v_target, reduction='mean')
    
    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_one_epoch(model, optimizer, num_batches, batch_size, n_steps, start_dist_params, goal_dist_params, action_dist_params, seq_len=1, use_ot=True, device='cpu'):
    """
    Train for one epoch.
    
    Args:
        model: FlowMatchingTransformerModel instance
        optimizer: torch optimizer
        num_batches: number of minibatches per epoch
        batch_size: number of samples in minibatch
        n_steps: number of interpolation steps per trajectory
        start_dist_params: dict with 'mu' and 'sigma' for start pose distribution
        goal_dist_params: dict with 'mu' and 'sigma' for goal pose distribution
        action_dist_params: dict with 'mu' and 'sigma' for action distribution
        device: device to run on
        
    Returns:
        avg_loss: average loss over the epoch
    """
    total_loss = 0.0
    
    for batch_idx in range(num_batches):
        loss = train_one_minibatch(
            model, optimizer, batch_size, n_steps,
            start_dist_params, goal_dist_params, action_dist_params, seq_len=seq_len, use_ot=use_ot, device=device
        )
        total_loss += loss
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(model, optimizer, num_epochs, num_batches_per_epoch, batch_size, n_steps,
          start_dist_params, goal_dist_params, action_dist_params, seq_len=1, use_ot=True, device='cpu', save_path=None):
    """
    Full training loop.
    
    Args:
        model: FlowMatchingTransformerModel instance
        optimizer: torch optimizer
        num_epochs: number of epochs to train
        num_batches_per_epoch: number of minibatches per epoch
        batch_size: number of samples in minibatch
        n_steps: number of interpolation steps per trajectory
        start_dist_params: dict with 'mu' and 'sigma' for start pose distribution
        goal_dist_params: dict with 'mu' and 'sigma' for goal pose distribution
        action_dist_params: dict with 'mu' and 'sigma' for action distribution
        device: device to run on
        save_path: path to save model checkpoints (optional)
        
    Returns:
        loss_history: list of average losses per epoch
    """
    loss_history = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Batches per epoch: {num_batches_per_epoch}")
    print(f"Batch size: {batch_size}")
    print(f"Interpolation steps: {n_steps}")
    print(f"Device: {device}")
    print()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        avg_loss = train_one_epoch(
            model, optimizer, num_batches_per_epoch, batch_size, n_steps,
            start_dist_params, goal_dist_params, action_dist_params, seq_len=seq_len, use_ot=use_ot, device=device
        )
        
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.6f}\n")
        
        # Save checkpoint
        if save_path and (epoch + 1) % 10 == 0:
            # Create directory if it doesn't exist
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
