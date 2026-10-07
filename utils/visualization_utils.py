import torch
import numpy as np
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, _quat_to_rot_mat

FRAME_COLORS = ('red', 'lime', 'blue')  # x, y, z axes of a pose frame
MODE_FRAME_COLORS = [
    ('darkred',    'darkgreen',      'darkblue'),
    ('saddlebrown','darkolivegreen',  'navy'),
    ('maroon',     'teal',            'indigo'),
    ('sienna',     'darkslategray',   'midnightblue'),
]
ARROW_LENGTH, ARROW_LINEWIDTH = 0.3, 1.5


def draw_frames(ax, poses, length=ARROW_LENGTH, linewidth=ARROW_LINEWIDTH, alpha=1.0,
                colors=FRAME_COLORS, label=None):
    """Draw the x/y/z axes of each pose (quaternion poses [..., 7]) as arrows."""
    poses = torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 7).cpu()
    R = _quat_to_rot_mat(poses[:, 3:7], w_first=True).numpy()  # [N, 3, 3], columns = axes
    p = poses[:, :3].numpy()
    for axis, color in enumerate(colors):
        d = R[:, :, axis]
        ax.quiver(p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2],
                  length=length, linewidth=linewidth, alpha=alpha,
                  arrow_length_ratio=0.3, color=color, label=label if axis == 0 else None)


def _as_pose_list(mean_goal_poses):
    """Normalize goal-mode means (6-D twists or 7-D quaternion poses, nested or not) to a
    list of 7-D numpy poses. A 2-D tensor is a batch of modes, one per row."""
    if torch.is_tensor(mean_goal_poses) and mean_goal_poses.dim() == 2:
        mean_goal_poses = list(mean_goal_poses)
    elif not isinstance(mean_goal_poses, (list, tuple)) or (
        len(mean_goal_poses) > 0 and not isinstance(mean_goal_poses[0], (list, tuple, np.ndarray))
    ):
        mean_goal_poses = [mean_goal_poses]
    out = []
    for mean_goal_pose in mean_goal_poses:
        if torch.is_tensor(mean_goal_pose):
            mean_pose_np = mean_goal_pose.cpu().numpy()
        elif isinstance(mean_goal_pose, list):
            mean_pose_np = np.array(mean_goal_pose)
        else:
            mean_pose_np = mean_goal_pose
        if mean_pose_np.shape[-1] == 6:
            mean_twist_tensor = torch.from_numpy(mean_pose_np).float()
            mean_pose_np = convert_twist_to_pose(mean_twist_tensor, dt=1.0,
                                                 return_representation='quat').cpu().numpy()
        # collapse any leading batch/seq dim so each entry is a 1-D pose
        if mean_pose_np.ndim == 2:
            mean_pose_np = mean_pose_np.mean(axis=0)
        out.append(mean_pose_np)
    return out


def draw_goal_modes(ax, mean_goal_poses, length=ARROW_LENGTH * 3.0, linewidth=ARROW_LINEWIDTH * 4.0,
                    label=True):
    """Draw each goal mode's mean pose as a large, dark frame."""
    for mode_idx, pose in enumerate(_as_pose_list(mean_goal_poses)):
        draw_frames(ax, torch.from_numpy(pose).float(), length=length, linewidth=linewidth,
                    colors=MODE_FRAME_COLORS[mode_idx % len(MODE_FRAME_COLORS)],
                    label=f'Goal Mode {mode_idx}' if label else None)


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
    arrow_step = max(1, num_steps // 10)  # Show ~10 frames per trajectory
    for i in range(min(batch_size, 10)):  # Plot up to 10 trajectories
        positions = trajectory_np[i, :, :3]  # Extract x, y, z positions

        # Plot position trajectory (no markers)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                alpha=0.6, linewidth=1.5, label=f'Trajectory {i+1}')

        # Orientation frames at intervals; bigger and bolder at the start and end
        for j in range(0, num_steps, arrow_step):
            ends = j == 0 or j == num_steps - 1
            draw_frames(ax, trajectory[i, j],
                        length=ARROW_LENGTH * (2.0 if ends else 1.0),
                        linewidth=ARROW_LINEWIDTH * (2.5 if ends else 1.0),
                        alpha=1.0 if ends else 0.7)

    # Plot mean goal poses if provided
    if mean_goal_poses is not None:
        draw_goal_modes(ax, mean_goal_poses)

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


# -- start -> goal mapping comparisons ------------------------------------------
# Categorical colours for conditions, in fixed order (validated palette, slots 1-4); marker
# shapes repeat the encoding so conditions never rest on hue alone.
CONDITION_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
CONDITION_MARKERS = ('o', 's', '^', 'D')
MUTED = '#8a8a85'


def plot_condition_mappings(ax, trajectory, condition_idx, mode_poses=None, title=None,
                            condition_names=None):
    """Start -> goal sampling paths [N, T+1, 7] (positions), one colour + marker per condition:
    small marker at the start, larger at the goal; goal-mode frames for reference."""
    traj = trajectory.detach().cpu().numpy()
    cond = np.asarray(condition_idx)
    for c in np.unique(cond):
        color = CONDITION_COLORS[c % len(CONDITION_COLORS)]
        marker = CONDITION_MARKERS[c % len(CONDITION_MARKERS)]
        paths = traj[cond == c]
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=0.45, linewidth=1.0)
        ax.scatter(paths[:, 0, 0], paths[:, 0, 1], paths[:, 0, 2], color=color, marker=marker,
                   s=12, alpha=0.9, depthshade=False)
        name = condition_names[c] if condition_names else f'condition {c}'
        ax.scatter(paths[:, -1, 0], paths[:, -1, 1], paths[:, -1, 2], color=color, marker=marker,
                   s=26, edgecolors='white', linewidths=0.4, depthshade=False, label=name)
    if mode_poses is not None:
        draw_goal_modes(ax, mode_poses, length=0.9, linewidth=3.0, label=False)
    if title:
        ax.set_title(title, fontsize=10)


def plot_goal_cloud(ax, poses, center, radius, frame_length=0.12, title=None):
    """Zoomed view of goal samples [N, 7] around one mode centre [7]: positions with small
    orientation frames, plus the mode's own frame. Same `radius` across panels compares spread.
    Samples farther than `radius` from the centre are left out (3-D axes don't clip); returns
    how many were left out."""
    poses = torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 7).cpu()
    center = torch.as_tensor(center, dtype=torch.float32).reshape(7).cpu()
    inside = (poses[:, :3] - center[:3]).abs().max(dim=-1).values <= radius
    poses = poses[inside]
    draw_frames(ax, poses, length=frame_length, linewidth=0.8, alpha=0.75)
    p = poses[:, :3].numpy()
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], color='black', s=4, depthshade=False)
    draw_frames(ax, center, length=frame_length * 3, linewidth=3.0,
                colors=MODE_FRAME_COLORS[0])
    c = center[:3].numpy()
    for lim, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
        lim(mid - radius, mid + radius)
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title, fontsize=10)
    return int((~inside).sum())


def plot_pairings(ax, start_poses, goal_poses, condition_idx, title=None, condition_names=None):
    """Training-time pairing of one minibatch: each start position coloured by the condition of
    the goal it is paired with, with a faint line to that goal."""
    s = torch.as_tensor(start_poses).reshape(-1, 7)[:, :3].cpu().numpy()
    g = torch.as_tensor(goal_poses).reshape(-1, 7)[:, :3].cpu().numpy()
    cond = np.asarray(condition_idx)
    for c in np.unique(cond):
        color = CONDITION_COLORS[c % len(CONDITION_COLORS)]
        sel = cond == c
        for a, b in zip(s[sel], g[sel]):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, alpha=0.18, linewidth=0.8)
        name = condition_names[c] if condition_names else f'condition {c}'
        ax.scatter(s[sel, 0], s[sel, 1], s[sel, 2], color=color,
                   marker=CONDITION_MARKERS[c % len(CONDITION_MARKERS)], s=16,
                   depthshade=False, label=name)
    if title:
        ax.set_title(title, fontsize=10)

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
