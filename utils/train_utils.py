import torch
import os
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch


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
