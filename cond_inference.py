import torch
import numpy as np
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, _quat_to_rot_mat


def load_model(checkpoint_path, device='cpu', model_config=None):
    """
    Load a trained flow matching model from checkpoint.
    
    Args:
        checkpoint_path: path to the checkpoint file
        device: device to load model on
        model_config: optional model config dict (required for old checkpoints without saved config)
        
    Returns:
        model: loaded FlowMatchingTransformerModel
        checkpoint: full checkpoint dict
    """
    model, checkpoint = ConditionalFlowMatchingTransformerModel.load_checkpoint(
        checkpoint_path, device=device, model_config=model_config
    )
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6f}")
    
    return model, checkpoint


def generate_from_start_poses(model, start_poses, obs, num_steps=100, return_trajectory=False, device='cpu'):
    """
    Generate goal poses from start poses using the trained model.
    
    Args:
        model: trained FlowMatchingTransformerModel
        start_poses: start poses as twists [batch, 6] or quaternions [batch, 7]
        obs: observation conditioning tensor [batch, obs_dim]
        num_steps: number of ODE integration steps
        return_trajectory: if True, return full trajectory
        device: device to run on
        
    Returns:
        goal_poses: generated goal poses [batch, 7]
        or trajectory: full trajectory [batch, num_steps+1, 7] if return_trajectory=True
    """
    model.eval()
    
    with torch.no_grad():
        # Convert start poses to quaternion format if needed
        if start_poses.shape[-1] == 6:
            # Input is twist, convert to quaternion pose
            x0 = convert_twist_to_pose(start_poses, dt=1.0, return_representation='quat')
        elif start_poses.shape[-1] == 7:
            # Already in quaternion format
            x0 = start_poses
        else:
            raise ValueError(f"Expected poses with dimension 6 or 7, got {start_poses.shape[-1]}")
        
        x0 = x0.to(device)
        
        # Generate goal poses
        result = model.inference(x0, obs, num_steps=num_steps, return_trajectory=return_trajectory)
        
    return result


def generate_from_distribution(model, distribution_params, obs_params, batch_size, num_steps=100, 
                               return_trajectory=False, device='cpu'):
    """
    Sample start poses from a distribution and generate goal poses.
    
    Args:
        model: trained FlowMatchingTransformerModel
        distribution_params: dict with 'mu' and 'sigma' for twist distribution
        batch_size: number of samples to generate
        num_steps: number of ODE integration steps
        return_trajectory: if True, return full trajectory
        device: device to run on
        
    Returns:
        start_poses: sampled start poses [batch, 7]
        goal_poses: generated goal poses [batch, 7]
        or trajectory: full trajectory [batch, num_steps+1, 7] if return_trajectory=True
    """
    # Sample start poses from distribution
    start_twists = sample_random_twist(
        batch_size=batch_size,
        mu=distribution_params['mu'],
        sigma=distribution_params['sigma'],
        device=device
    )
    
    obs = sample_random_twist(
        batch_size=batch_size,
        mu=obs_params['mu'],
        sigma=obs_params['sigma'],
        device=device
    )

    obs = obs.unsqueeze(1)  # Add sequence dimension

    # Convert to quaternion format
    start_poses = convert_twist_to_pose(start_twists, dt=1.0, return_representation='quat')
    
    # Generate goal poses
    result = generate_from_start_poses(
        model, start_poses, obs, num_steps=num_steps, 
        return_trajectory=return_trajectory, device=device
    )
    
    if return_trajectory:
        return start_poses, result
    else:
        return start_poses, result


def visualize_trajectory(trajectory, save_path=None, mean_goal_pose=None):
    """
    Visualize the trajectory of poses (positions and orientations).
    
    Args:
        trajectory: tensor of shape [batch, num_steps, 7]
        save_path: optional path to save figure
        mean_goal_pose: optional mean goal pose tensor of shape [7] to display as reference
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    trajectory_np = trajectory.cpu().numpy()
    batch_size, num_steps, _ = trajectory_np.shape
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    for i in range(min(batch_size, 10)):  # Plot up to 10 trajectories
        positions = trajectory_np[i, :, :3]  # Extract x, y, z positions
        quaternions = trajectory_np[i, :, 3:7]  # Extract quaternions
        
        # Plot position trajectory (no markers)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                alpha=0.6, linewidth=1.5, label=f'Trajectory {i+1}')
        
        # Plot orientation axes at intervals
        arrow_step = max(1, num_steps // 10)  # Show ~10 frames per trajectory
        arrow_length = 0.3
        arrow_linewidth = 1.5
        
        for j in range(0, num_steps, arrow_step):
            pos = positions[j]
            quat = quaternions[j]
            
            # Convert quaternion to rotation matrix using tf_utils
            quat_tensor = torch.from_numpy(quat).float()
            R = _quat_to_rot_mat(quat_tensor, w_first=True).numpy()
            
            # Extract the three axes from the rotation matrix (columns)
            x_axis = R[:, 0]  # X-axis in red
            y_axis = R[:, 1]  # Y-axis in green
            z_axis = R[:, 2]  # Z-axis in blue
            
            # Use bigger, bolder axes for start and end
            if j == 0 or j == num_steps - 1:
                length = arrow_length * 2.0  # Bigger for start/end
                linewidth = arrow_linewidth * 2.5  # Bolder for start/end
                alpha = 1.0  # Fully opaque
            else:
                length = arrow_length
                linewidth = arrow_linewidth
                alpha = 0.7
            
            # Draw three arrows showing the coordinate frame axes
            ax.quiver(pos[0], pos[1], pos[2],
                     x_axis[0], x_axis[1], x_axis[2],
                     length=length, linewidth=linewidth, alpha=alpha,
                     arrow_length_ratio=0.3, color='red')
            ax.quiver(pos[0], pos[1], pos[2],
                     y_axis[0], y_axis[1], y_axis[2],
                     length=length, linewidth=linewidth, alpha=alpha,
                     arrow_length_ratio=0.3, color='lime')
            ax.quiver(pos[0], pos[1], pos[2],
                     z_axis[0], z_axis[1], z_axis[2],
                     length=length, linewidth=linewidth, alpha=alpha,
                     arrow_length_ratio=0.3, color='blue')
    
    # Plot mean goal pose if provided
    if mean_goal_pose is not None:
        # convert from either torch tensor or list to numpy
        if torch.is_tensor(mean_goal_pose):
            mean_pose_np = mean_goal_pose.cpu().numpy()
        elif isinstance(mean_goal_pose, list):
            mean_pose_np = np.array(mean_goal_pose)
        else:
            mean_pose_np = mean_goal_pose
        
        # Check if it's a twist (6 elements) or pose (7 elements)
        if mean_pose_np.shape[-1] == 6:
            # Convert twist to pose
            mean_twist_tensor = torch.from_numpy(mean_pose_np).float()
            mean_pose_tensor = convert_twist_to_pose(mean_twist_tensor, dt=1.0, return_representation='quat')
            mean_pose_np = mean_pose_tensor.cpu().numpy()
        
        mean_pos = mean_pose_np[:3]
        mean_quat = mean_pose_np[3:7]
        
        # Convert quaternion to rotation matrix using tf_utils
        mean_quat_tensor = torch.from_numpy(mean_quat).float()
        R_mean = _quat_to_rot_mat(mean_quat_tensor, w_first=True).numpy()
        
        # Extract axes
        x_axis_mean = R_mean[:, 0]
        y_axis_mean = R_mean[:, 1]
        z_axis_mean = R_mean[:, 2]
        
        # Plot mean goal pose with distinctive styling (extra large and thick, darker colors)
        mean_length = arrow_length * 3.0  # Even larger than start/end
        mean_linewidth = arrow_linewidth * 4.0  # Extra thick
        
        ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                 x_axis_mean[0], x_axis_mean[1], x_axis_mean[2],
                 length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                 arrow_length_ratio=0.3, color='darkred', label='Mean Goal')
        ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                 y_axis_mean[0], y_axis_mean[1], y_axis_mean[2],
                 length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                 arrow_length_ratio=0.3, color='darkgreen')
        ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                 z_axis_mean[0], z_axis_mean[1], z_axis_mean[2],
                 length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                 arrow_length_ratio=0.3, color='darkblue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = 'Flow Matching Trajectories\n(RGB Arrows=XYZ Axes, Larger=Start/Goal'
    if mean_goal_pose is not None:
        title += ', Largest/Dark=Mean Goal'
    title += ')'
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_poses(poses, name="Poses"):
    """Pretty print poses."""
    print(f"\n{name}:")
    print(f"  Shape: {poses.shape}")
    if poses.dim() == 2:
        for i, pose in enumerate(poses[:5]):  # Print first 5
            pos = pose[:3]
            quat = pose[3:7] if pose.shape[0] >= 7 else None
            print(f"  [{i}] pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", end="")
            if quat is not None:
                print(f" quat: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})")
            else:
                print()
        if poses.shape[0] > 5:
            print(f"  ... ({poses.shape[0] - 5} more)")


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Example 1: Load model and generate from specific start poses
    print("=" * 60)
    print("EXAMPLE 1: Load model and generate from start poses")
    print("=" * 60)
    
    checkpoint_path = "checkpoints/cond_flow_matching_model_OT_epoch_10.pt"
    
    # Model config (needed for old checkpoints that don't have model_config saved)
    # This should match the configuration used during training
    model_config = {
        'input_dim': 7,
        'output_dim': 6,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'phase_dim': 128,  # Must match training config
        'max_seq_len': 1   # Must match training config
    }
    
    try:
        model, checkpoint = load_model(checkpoint_path, device=device, model_config=model_config)
        
        # Define start poses
        start_twists = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
        ], device=device)
        
        start_poses = convert_twist_to_pose(start_twists, dt=1.0, return_representation='quat')
        print_poses(start_poses, "Start Poses")

        # all "go down" observations
        obs = torch.tensor([
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        ], device=device)
        obs = obs.unsqueeze(1)  # Add sequence dimension
        
        # Generate goal poses
        goal_poses = generate_from_start_poses(
            model, start_poses, obs, num_steps=100, return_trajectory=False, device=device
        )
        print_poses(goal_poses, "Generated Goal Poses")
        
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_flow_model.py\n")
    
    # Example 2: Generate from distribution with trajectory
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generate from distribution with trajectory")
    print("=" * 60)
    
    try:
        # Define start distribution (same as training)
        start_dist_params = {
            'mu': [0, 0, 0, 0, 0, 0],
            'sigma': [1, 1, 1, 1, 1, 1]
        }

        goal_dist_params = {
            'mu': [5, 5, 5, 0, 0, 1.5708],  # 90 deg around z
            'sigma': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }

        obs_dist_params = {
            'mu': [0, 0, -1, 0, 0, 0],  # "go up"
            'sigma': [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
        }
        
        n_samples = 15
        # Generate trajectories
        start_poses, trajectory = generate_from_distribution(
            model, start_dist_params, obs_dist_params, batch_size=n_samples, num_steps=50,
            return_trajectory=True, device=device
        )
        
        print_poses(start_poses, "Sampled Start Poses")
        print(f"\nTrajectory shape: {trajectory.shape}")
        print(f"  [batch_size={trajectory.shape[0]}, num_steps={trajectory.shape[1]}, pose_dim={trajectory.shape[2]}]")
        
        # Print final poses
        goal_poses = trajectory[:, -1, :]
        print_poses(goal_poses, "Final Goal Poses")
        
        # Visualize trajectories
        print("\nGenerating trajectory visualization...")
        # visualize_trajectory(trajectory, save_path="trajectory_visualization.png")
        visualize_trajectory(trajectory, mean_goal_pose=goal_dist_params['mu'])

    except NameError:
        print("Model not loaded. Skipping this example.\n")
    
    # Example 3: Batch inference
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch inference from multiple start poses")
    print("=" * 60)
    
    try:
        batch_size = 100
        start_dist_params = {
            'mu': [0, 0, 0, 0, 0, 0],
            'sigma': [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
        }
        
        start_poses, goal_poses = generate_from_distribution(
            model, start_dist_params, batch_size=batch_size, 
            num_steps=100, return_trajectory=False, device=device
        )
        
        print_poses(start_poses, f"Batch of {batch_size} Start Poses")
        print_poses(goal_poses, f"Batch of {batch_size} Goal Poses")
        
        # Compute statistics
        start_mean = start_poses[:, :3].mean(dim=0)
        goal_mean = goal_poses[:, :3].mean(dim=0)
        
        print(f"\nPosition Statistics:")
        print(f"  Start mean: ({start_mean[0]:.3f}, {start_mean[1]:.3f}, {start_mean[2]:.3f})")
        print(f"  Start std:  ({start_poses[:, :3].std(dim=0)[0]:.3f}, {start_poses[:, :3].std(dim=0)[1]:.3f}, {start_poses[:, :3].std(dim=0)[2]:.3f})")
        print(f"  Goal mean:  ({goal_mean[0]:.3f}, {goal_mean[1]:.3f}, {goal_mean[2]:.3f})")
        print(f"  Goal std:   ({goal_poses[:, :3].std(dim=0)[0]:.3f}, {goal_poses[:, :3].std(dim=0)[1]:.3f}, {goal_poses[:, :3].std(dim=0)[2]:.3f})")

        
    except NameError:
        print("Model not loaded. Skipping this example.\n")
    
    print("\n" + "=" * 60)
    print("Inference examples completed!")
    print("=" * 60)
