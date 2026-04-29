import torch
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, _quat_to_rot_mat
from utils.visualization_utils import visualize_trajectory, print_poses
from omegaconf import OmegaConf

def load_model(checkpoint_path, device='cpu', model_config=None, conditional=False):
    """
    Load a trained flow matching model from checkpoint.

    Args:
        checkpoint_path: path to the checkpoint file
        device: device to load model on
        model_config: optional model config dict (required for old checkpoints without saved config)
        conditional: whether to load conditional or unconditional model

    Returns:
        model: loaded FlowMatchingTransformerModel
        checkpoint: full checkpoint dict
    """
    if conditional:
        ModelClass = ConditionalFlowMatchingTransformerModel
    else:
        ModelClass = FlowMatchingTransformerModel
        # Remove obs_dim if present (not needed for unconditional model)
        if model_config is not None and 'obs_dim' in model_config:
            model_config = model_config.copy()
            del model_config['obs_dim']

    model, checkpoint = ModelClass.load_checkpoint(
        checkpoint_path, device=device, model_config=model_config
    )
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6f}")

    return model, checkpoint

def generate_from_start_poses(model, start_poses, obs=None, num_steps=100, return_trajectory=False, device='cpu'):
    """
    Generate goal poses from start poses using the trained model.

    Args:
        model: trained FlowMatchingTransformerModel (conditional or non-conditional)
        start_poses: start poses as twists [batch, 6] or quaternions [batch, 7]
        obs: observation conditioning tensor [batch, obs_dim] (required for conditional models)
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

        # Check if model is conditional
        is_conditional = isinstance(model, ConditionalFlowMatchingTransformerModel)

        if is_conditional and obs is not None:
            obs = obs.to(device)

        # Generate goal poses
        try:
            if is_conditional:
                result = model.inference(x0, obs, num_steps=num_steps, return_trajectory=return_trajectory)
            else:
                result = model.inference(x0, num_steps=num_steps, return_trajectory=return_trajectory)
        except Exception as e:
            print("Error during model inference:", e)
            raise e

    return result

def generate_from_distribution(model, distribution_params, batch_size, obs=None, num_steps=100,
                               return_trajectory=False, device='cpu'):
    """
    Sample start poses from a distribution and generate goal poses.

    Args:
        model: trained FlowMatchingTransformerModel (conditional or non-conditional)
        distribution_params: dict with 'mu' and 'sigma' for twist distribution
        batch_size: number of samples to generate
        obs: tensor of observations
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Inference with trained flow matching model")
    parser.add_argument('--actions', '-A', type=str, nargs='*', default=None, help="List of conditions to combine (e.g., top left, bottom right, top). Omit for unconditional (null-token) inference.")
    parser.add_argument('--conditional', '-C', action='store_true', help="Whether to use conditional model (requires obs parameters)")
    parser.add_argument('--checkpoint_epoch', '-CE', type=int, default=10, help="Model epoch to load")
    parser.add_argument('--checkpoint_path', '-CP', type=str, default=None, help="Path to checkpoint directory (default: checkpoints/)")
    parser.add_argument('--num_samples', '-N', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--num_steps', '-STEPS', type=int, default=100, help="Number of ODE integration steps")
    parser.add_argument('--return_trajectory', '-RT', action='store_true', help="Whether to return full trajectory")
    parser.add_argument('--no_ot', '-NOOT', action='store_true', help="Load a model trained without optimal transport pairing")
    return parser.parse_args()

def get_config_and_checkpoint_paths(args):
    """
    Determine config and checkpoint paths based on args.

    Returns:
        config_path: path to training config yaml
        checkpoint_path: path to model checkpoint
    """
    base_path = args.checkpoint_path if args.checkpoint_path else "checkpoints/"
    ot_suffix = '_NOOT' if args.no_ot else '_OT'
    model_name = 'cond_pose_flow_matching_model' if args.conditional else 'pose_flow_matching_model'

    config_path = f"{base_path}{model_name}{ot_suffix}_training_config.yaml"
    checkpoint_path = f"{base_path}{model_name}{ot_suffix}_epoch_{args.checkpoint_epoch}.pt"

    return config_path, checkpoint_path

def get_goal_modes_for_actions(actions, goal_dist):
    """Returns the subset of goal mode means whose positions satisfy all action conditions.

    Actions filter on the goal twist (y=index1, z=index2):
      top/bottom → z > 0 / z < 0
      right/left → y > 0 / y < 0
    When multiple actions are given, only modes satisfying ALL conditions are returned.
    """
    conditions = {
        'top':    lambda mu: mu[2] > 0,
        'bottom': lambda mu: mu[2] < 0,
        'right':  lambda mu: mu[1] > 0,
        'left':   lambda mu: mu[1] < 0,
    }
    return [mu for mu in goal_dist['mu']
            if all(conditions[a](mu) for a in actions if a in conditions)]

def build_obs_from_actions(actions, batch_size, device):
    """Dynamically builds [batch, M, 6] tensor based on requested conditions."""
    action_map = {
        'top':    [0.0,  0.0, 1.0, 0.0, 0.0, 0.0],
        'bottom': [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        'right':  [0.0,  1.0, 0.0, 0.0, 0.0, 0.0],
        'left':   [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    obs_tokens = []
    for a in actions:
        if a in action_map:
            obs_tokens.append(action_map[a])
        else:
            raise ValueError(f"Unknown action: {a}")
            
    # [M, 6] -> [1, M, 6] -> [batch, M, 6]
    obs_tensor = torch.tensor(obs_tokens, dtype=torch.float32, device=device)
    return obs_tensor.unsqueeze(0).repeat(batch_size, 1, 1)


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    args = parse_args()

    # Get paths based on conditional flag
    config_path, checkpoint_path = get_config_and_checkpoint_paths(args)

    # Load the config
    try:
        config = OmegaConf.load(config_path)
        print(f"Loaded training config from {config_path}")
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Please ensure the model was trained with the unified training script that saves configs.")
        exit(1)

    # Convert config to dict and extract model config
    model_config = OmegaConf.to_container(config.model, resolve=True)

    # Example 1: Load model and generate from specific start poses
    print("=" * 60)
    print("EXAMPLE 1: Load model and generate from start poses")
    print("=" * 60)

    try:
        model, checkpoint = load_model(checkpoint_path, device=device,
                                       model_config=model_config,
                                       conditional=args.conditional)

        # Define start poses
        start_twists = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
        ], device=device)

        start_poses = convert_twist_to_pose(start_twists, dt=1.0, return_representation='quat')
        print_poses(start_poses, "Start Poses")

        # Create observations for conditional model
        if args.conditional and args.actions:
            obs = build_obs_from_actions(args.actions, start_poses.shape[0], device)
            print(f"Using actions '{args.actions}' with {obs.shape[1]} tokens.")
        elif args.conditional:
            obs = None
            print("No actions provided — using null tokens (unconditional).")
        else:
            obs = None

        # Generate goal poses
        goal_poses = generate_from_start_poses(
            model, start_poses, obs, num_steps=args.num_steps, return_trajectory=False, device=device
        )
        print_poses(goal_poses, "Generated Goal Poses")

    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train_flow_model.py\n")
        exit(1)

    # Example 2: Generate from distribution with trajectory
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generate from distribution with trajectory")
    print("=" * 60)

    # Get distribution params from config
    start_dist_params = OmegaConf.to_container(config.training.start_dist_params, resolve=True)

    # Read all goal distribution modes from config
    goal_dist = OmegaConf.to_container(config.training.goal_dist_params, resolve=True)
    goal_dist_params = {'mu': goal_dist['mu'][0], 'sigma': goal_dist['sigma'][0]}

    if args.conditional and args.actions:
        obs = build_obs_from_actions(args.actions, args.num_samples, device)
        goal_all_means = get_goal_modes_for_actions(args.actions, goal_dist)
        print(f"Using actions '{args.actions}', matched {len(goal_all_means)} goal mode(s):")
    elif args.conditional:
        obs = None
        goal_all_means = goal_dist['mu']
        print(f"No actions provided — using null tokens (unconditional). All goal modes ({len(goal_all_means)}):")
    else:
        obs = None
        goal_all_means = goal_dist['mu']
        print(f"Goal distribution ({len(goal_all_means)} modes):")

    for i, mu in enumerate(goal_all_means):
        print(f"  Mode {i}: {mu}")

    # Generate trajectories
    start_poses, trajectory = generate_from_distribution(
        model, start_dist_params, args.num_samples, obs=obs, num_steps=50,
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
    visualize_trajectory(trajectory, mean_goal_poses=goal_all_means)

    # Example 3: Batch inference
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch inference from multiple start poses")
    print("=" * 60)

    batch_size = 100

    obs_batch = build_obs_from_actions(args.actions, batch_size, device) if (args.conditional and args.actions) else None
    start_poses, goal_poses = generate_from_distribution(
        model, start_dist_params, batch_size=batch_size, obs=obs_batch,
        num_steps=args.num_steps, return_trajectory=False, device=device
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

    # Show expected goal position from config
    print(f"\nExpected goal position: {goal_dist_params['mu'][:3]}")

    print("\n" + "=" * 60)
    print("Inference examples completed!")
    print("=" * 60)
