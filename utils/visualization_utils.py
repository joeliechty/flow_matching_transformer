import torch
import numpy as np
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, _quat_to_rot_mat


def visualize_trajectory(trajectory, save_path=None, mean_goal_poses=None):
    """
    Visualize the trajectory of poses (positions and orientations).

    Args:
        trajectory: tensor of shape [batch, num_steps, 7]
        save_path: optional path to save figure
        mean_goal_poses: optional list of mean goal poses (each 6D twist or 7D quat) to display as reference
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
    
    # Plot mean goal poses if provided
    if mean_goal_poses is not None:
        # Normalize to a list of poses
        if not isinstance(mean_goal_poses, (list, tuple)) or (
            len(mean_goal_poses) > 0 and not isinstance(mean_goal_poses[0], (list, tuple, np.ndarray))
        ):
            mean_goal_poses = [mean_goal_poses]

        mode_colors = [
            ('darkred',    'darkgreen',      'darkblue'),
            ('saddlebrown','darkolivegreen',  'navy'),
            ('maroon',     'teal',            'indigo'),
            ('sienna',     'darkslategray',   'midnightblue'),
        ]

        mean_length = arrow_length * 3.0
        mean_linewidth = arrow_linewidth * 4.0

        for mode_idx, mean_goal_pose in enumerate(mean_goal_poses):
            if torch.is_tensor(mean_goal_pose):
                mean_pose_np = mean_goal_pose.cpu().numpy()
            elif isinstance(mean_goal_pose, list):
                mean_pose_np = np.array(mean_goal_pose)
            else:
                mean_pose_np = mean_goal_pose

            if mean_pose_np.shape[-1] == 6:
                mean_twist_tensor = torch.from_numpy(mean_pose_np).float()
                mean_pose_tensor = convert_twist_to_pose(mean_twist_tensor, dt=1.0, return_representation='quat')
                mean_pose_np = mean_pose_tensor.cpu().numpy()

            # collapse any leading batch/seq dim so indexing below is always on a 1-D array
            if mean_pose_np.ndim == 2:
                mean_pose_np = mean_pose_np.mean(axis=0)

            mean_pos = mean_pose_np[:3]
            mean_quat = mean_pose_np[3:7]

            mean_quat_tensor = torch.from_numpy(mean_quat).float()
            R_mean = _quat_to_rot_mat(mean_quat_tensor, w_first=True).numpy()

            x_axis_mean = R_mean[:, 0]
            y_axis_mean = R_mean[:, 1]
            z_axis_mean = R_mean[:, 2]

            colors = mode_colors[mode_idx % len(mode_colors)]
            ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                     x_axis_mean[0], x_axis_mean[1], x_axis_mean[2],
                     length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                     arrow_length_ratio=0.3, color=colors[0], label=f'Goal Mode {mode_idx}')
            ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                     y_axis_mean[0], y_axis_mean[1], y_axis_mean[2],
                     length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                     arrow_length_ratio=0.3, color=colors[1])
            ax.quiver(mean_pos[0], mean_pos[1], mean_pos[2],
                     z_axis_mean[0], z_axis_mean[1], z_axis_mean[2],
                     length=mean_length, linewidth=mean_linewidth, alpha=1.0,
                     arrow_length_ratio=0.3, color=colors[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = 'Flow Matching Trajectories\n(RGB Arrows=XYZ Axes, Larger=Start/Goal'
    if mean_goal_poses is not None:
        title += ', Largest/Dark=Goal Modes'
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


def visualize_image_trajectory(trajectory, num_samples=8, num_timesteps=10, save_path=None,
                               grid_hw=(28, 28), title=None):
    """
    Plot a tiled grid showing the denoising trajectory of image samples.

    Rows = different samples; Columns = evenly spaced timesteps from pure noise (left)
    to the final denoised image (right).

    Args:
        trajectory: tensor of shape [batch, T, 1, H, W] OR [batch, T, H*W] (any flat-image form).
        num_samples: number of rows (samples) to display.
        num_timesteps: number of columns (timesteps) to display, evenly spaced over T.
        save_path: optional path to save figure.
        grid_hw: (H, W) used to reshape flat trajectories.
        title: optional figure suptitle.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    traj = trajectory.detach().cpu()
    if traj.dim() == 3:
        # [B, T, H*W] -> [B, T, H, W]
        B, T, _ = traj.shape
        H, W = grid_hw
        traj = traj.view(B, T, H, W)
    elif traj.dim() == 5:
        # [B, T, 1, H, W] -> [B, T, H, W]
        traj = traj.squeeze(2)

    B, T, H, W = traj.shape
    num_samples = min(num_samples, B)
    num_timesteps = min(num_timesteps, T)

    # Pick evenly spaced timesteps, inclusive of first and last
    if num_timesteps == 1:
        t_idx = [T - 1]
    else:
        t_idx = np.linspace(0, T - 1, num_timesteps).round().astype(int).tolist()

    fig, axes = plt.subplots(
        num_samples, num_timesteps,
        figsize=(num_timesteps * 1.1, num_samples * 1.1),
        squeeze=False,
    )

    for r in range(num_samples):
        for c, ti in enumerate(t_idx):
            ax = axes[r][c]
            ax.imshow(traj[r, ti].numpy(), cmap='gray', vmin=traj[r].min(), vmax=traj[r].max())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={ti}", fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Image trajectory saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
