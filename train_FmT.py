import torch
import os
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.train_utils import geodesic_optimal_transport_pairing, generate_interpolated_poses
from omegaconf import OmegaConf


def train_one_minibatch(model, optimizer, batch_size, n_steps, start_dist_params, goal_dist_params, action_dist_params, device='cpu'):
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
    
    # Sample start poses from start distribution
    start_poses = sample_random_twist(
        batch_size=batch_size,
        mu=start_dist_params['mu'],
        sigma=start_dist_params['sigma'],
        device=device
    )  # [batch_size, 6]
    
    # Sample goal poses from goal distributions, shape: [batch_size, 6]
    for i in range(len(goal_dist_params['mu'])):
        if i == 0:
            goal_poses = sample_random_twist(
                batch_size=batch_size // len(goal_dist_params['mu']),
                mu=goal_dist_params['mu'][i],
                sigma=goal_dist_params['sigma'][i],
                device=device
            )
        else:
            goal_poses = torch.cat((
                goal_poses,
                sample_random_twist(
                    batch_size=batch_size // len(goal_dist_params['mu']),
                    mu=goal_dist_params['mu'][i],
                    sigma=goal_dist_params['sigma'][i],
                    device=device
                )
            ), dim=0)

    # Sample actions from action distributions (only for conditional models)
    if action_dist_params is not None:
        for i in range(len(action_dist_params['mu'])):
            if i == 0:
                obs = sample_random_twist(
                    batch_size=batch_size // len(action_dist_params['mu']),
                    mu=action_dist_params['mu'][i],
                    sigma=action_dist_params['sigma'][i],
                    device=device
                )
            else:
                obs = torch.cat((
                    obs,
                    sample_random_twist(
                        batch_size=batch_size // len(action_dist_params['mu']),
                        mu=action_dist_params['mu'][i],
                        sigma=action_dist_params['sigma'][i],
                        device=device
                    )
                ), dim=0)
    else:
        obs = None

    # optimal transport pairing
    start_poses = geodesic_optimal_transport_pairing(start_poses, goal_poses)  # [batch_size, 6]

    # Compute interpolated poses, time steps, and target vector fields (twists)
    interpolated_poses, t, twist_target = generate_interpolated_poses(
        start_poses, goal_poses, n_steps=n_steps
    )  # [batch_size, n_steps, 7], [batch_size, n_steps], [batch_size, n_steps, 6]

    # Flatten batch and steps dimensions
    batch_steps = batch_size * n_steps
    x_t = interpolated_poses.reshape(batch_steps, 7)  # [batch_size * n_steps, 7]
    t_flat = t.reshape(batch_steps)  # [batch_size * n_steps]
    v_target = twist_target.reshape(batch_steps, 6)  # [batch_size * n_steps, 6]

    # Add sequence dimension (treating each pose as a single sequence element)
    x_t = x_t.unsqueeze(1)  # [batch_size * n_steps, 1, 7]
    v_target = v_target.unsqueeze(1)  # [batch_size * n_steps, 1, 6]

    # For conditional models, repeat observations and flatten, add sequence dimension
    if obs is not None:
        obs = obs.unsqueeze(1).repeat_interleave(n_steps, dim=0)  # [batch_size * n_steps, 6]

    # Compute loss (conditional vs non-conditional)
    if isinstance(model, ConditionalFlowMatchingTransformerModel):
        loss = model.cfm_loss(x_t, t_flat, v_target, obs, reduction='mean')
    else:
        loss = model.cfm_loss(x_t, t_flat, v_target, reduction='mean')
    
    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_one_epoch(model, optimizer, num_batches, batch_size, n_steps, start_dist_params, goal_dist_params, action_dist_params, device='cpu'):
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
            start_dist_params, goal_dist_params, action_dist_params, device
        )
        total_loss += loss
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(model, optimizer, num_epochs, num_batches_per_epoch, batch_size, n_steps, 
          start_dist_params, goal_dist_params, action_dist_params, device='cpu', save_path=None):
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
            start_dist_params, goal_dist_params, action_dist_params, device
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching Transformer Model")
    parser.add_argument('--num_epochs', '-E', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_batches_per_epoch', '-BE', type=int, default=100, help='Number of minibatches per epoch')
    parser.add_argument('--batch_size', '-B', type=int, default=None, help='Number of samples in each minibatch')
    parser.add_argument('--conditional', '-C', action='store_true', help='Whether to train conditional model (with observations)')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of interpolation steps per trajectory')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='Path to save model checkpoints')
    args = parser.parse_args()
    return args

def generate_training_and_model_config(args, start_dist_params=None, goal_dist_params=None, action_dist_params=None):
    # Define distribution parameters (twist representation)
    if start_dist_params is None:
        start_dist_params = {
            'mu': [0, 0, 0, 0, 0, 0],
            'sigma': [1, 1, 1, 1, 1, 1]
        }
    
    # goal pos at (5,5,5) with 90 deg rotation around z axis (twist representation)
    if goal_dist_params is None:
        goal_dist_params = {
            'mu': [[5, 5, 5, 0, 0, 1.5708],[5, 5, -5, 0, 0, -1.5708]],
            'sigma': [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
        }

    if args.conditional:
        if action_dist_params is None:
            # the "go up" twist and the "go down" twist
            action_dist_params = {
                'mu': [[0, 0, 1, 0, 0, 0],[0, 0, -1, 0, 0, 0]],
                'sigma': [[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],[0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]
            }

    if args.batch_size is None:
        batch_size = len(goal_dist_params['mu'])*32
    else:
        batch_size = args.batch_size

    if args.conditional:
        save_path = os.path.join(args.save_path, 'cond_flow_matching_model_OT')
        obs_dim = 6  # action representation (vx, vy, vz, wx, wy, wz)
    else:
        save_path = os.path.join(args.save_path, 'flow_matching_model_OT')
        obs_dim = None

    # make a config object
    training_config = {
        'conditional': args.conditional,
        'num_epochs': args.num_epochs,
        'num_batches_per_epoch': args.num_batches_per_epoch,
        'batch_size': batch_size,
        'n_interp_steps': args.n_steps,
        'save_path': save_path,
        'start_dist_params': start_dist_params,
        'goal_dist_params': goal_dist_params,
        'action_dist_params': action_dist_params if args.conditional else None,
        'lr': 1e-4,
        'weight_decay': 1e-5
    }    
    model_config = {
        'input_dim': 7,  # quaternion pose representation (x, y, z, qw, qx, qy, qz)
        'output_dim': 6,  # twist representation (vx, vy, vz, wx, wy, wz)
        'obs_dim': obs_dim,  # action representation (vx, vy, vz, wx, wy, wz)
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'phase_dim': 128,
        'max_seq_len': 1
    }
    config_dict = {
        'training': training_config,
        'model': model_config
    }
    config = OmegaConf.create(config_dict)

    # save the config to the same directory as the checkpoints
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Created directory for saving checkpoints and config: {args.save_path}")
    config_save_path = save_path + '_training_config.yaml'
    OmegaConf.save(config, config_save_path)
    print(f"Training and model configuration saved to {config_save_path}\n")

    return config

def test_data_generation(device):
    # start pose distribution at the origin
    start_poses = sample_random_twist(batch_size=3, mu=[0,0,0,0,0,0], sigma=[1,1,1,1,1,1], device=device)
    print("start_poses (as twists):\n", start_poses)
    start_poses_ortho6d = convert_twist_to_pose(start_poses, dt=1.0, return_representation='ortho6d')
    print("start_poses (as ortho6d poses):\n", start_poses_ortho6d)
    start_poses_T = convert_twist_to_pose(start_poses, dt=1.0, return_representation='T_mat')
    print("start_poses (as T_mat poses):\n", start_poses_T)

    # goal pose distribution at different position with with no rotation
    goal_poses = sample_random_twist(batch_size=3, mu=[5,5,5,0,0,0], sigma=[0.1,0.1,0.1,0.1,0.1,0.1], device=device)
    print("goal_poses (as twists):\n", goal_poses)
    goal_poses_ortho6d = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='ortho6d')
    print("goal_poses (as ortho6d poses):\n", goal_poses_ortho6d)
    goal_poses_T = convert_twist_to_pose(goal_poses, dt=1.0, return_representation='T_mat')
    print("goal_poses (as T_mat poses):\n", goal_poses_T)

    n_steps = 5
    interpolated_poses, t, twist_target = generate_interpolated_poses(start_poses, goal_poses, n_steps=n_steps)
    print("interpolated_poses shape:", interpolated_poses.shape)
    print("t shape:", t.shape)
    print("twist_target shape:", twist_target.shape)
    print()

if __name__ == "__main__":

    args = parse_args()

    # Set device (cuda > mps > cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test data generation
    print("=" * 60)
    print("TESTING DATA GENERATION")
    print("=" * 60)
    
    test_data_generation(device)
    
    # Training setup
    print("=" * 60)
    print("TRAINING FLOW MATCHING TRANSFORMER")
    print("=" * 60)
    
    config = generate_training_and_model_config(args)

    print(f"Input Dimension: {config.model.input_dim}")
    print(f"Number of Epochs: {config.training.num_epochs}")

    if args.conditional:
        model = ConditionalFlowMatchingTransformerModel(
            input_dim=config.model.input_dim,  # quaternion pose representation (x, y, z, qw, qx, qy, qz)
            output_dim=config.model.output_dim,  # twist representation (vx, vy, vz, wx, wy, wz)
            obs_dim=config.model.obs_dim,  # action representation (vx, vy, vz, wx, wy, wz)
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            phase_dim=config.model.phase_dim,
            max_seq_len=config.model.max_seq_len
        ).to(device)
    else:
        model = FlowMatchingTransformerModel(
            input_dim=config.model.input_dim,  # quaternion pose representation (x, y, z, qw, qx, qy, qz)
            output_dim=config.model.output_dim,  # twist representation (vx, vy, vz, wx, wy, wz)
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            phase_dim=config.model.phase_dim,
            max_seq_len=config.model.max_seq_len
        ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    
    # Train the model
    loss_history = train(
        model=model,
        optimizer=optimizer,
        num_epochs=config.training.num_epochs,
        num_batches_per_epoch=config.training.num_batches_per_epoch,
        batch_size=config.training.batch_size,
        n_steps=config.training.n_interp_steps,
        start_dist_params=config.training.start_dist_params,
        goal_dist_params=config.training.goal_dist_params,
        action_dist_params=config.training.action_dist_params,
        device=device,
        save_path=config.training.save_path
    )
    
    # Plot loss history
    print("\nLoss history:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: {loss:.6f}")
    
    # # Test sampling
    # print("\n" + "=" * 60)
    # print("TESTING SAMPLING")
    # print("=" * 60)
    
    # model.eval()
    # with torch.no_grad():
    #     # Sample from start distribution
    #     test_start_poses = sample_random_twist(
    #         batch_size=6,
    #         mu=config.training.start_dist_params.mu,
    #         sigma=config.training.start_dist_params.sigma,
    #         device=device
    #     )

    #     # go up obs
    #     down = torch.tensor([
    #         [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    #     ], device=device)
    #     # go down obs
    #     up = down*-1.0
    #     obs = torch.cat((up.repeat(3,1), down.repeat(3,1)), dim=0)  # [6, 6]
    #     obs = obs.unsqueeze(1)  # Add sequence dimension
        
    #     # Convert to pose representation
    #     x0 = convert_twist_to_pose(test_start_poses, dt=1.0, return_representation='quat')
    #     x0 = x0.unsqueeze(1)  # Add sequence dimension [5, 1, 7]
        
    #     print("Initial poses (from start distribution):")
    #     print(x0.squeeze(1))
        
    #     # Generate samples by flowing to goal distribution
    #     x1 = model.sample(x0, num_steps=50, method='euler')
        
    #     print("\nGenerated poses (should be near goal distribution):")
    #     print(x1.squeeze(1)[:, :3])  # Print positions
        
    #     print("\nExpected goal positions around:", config.training.goal_dist_params.mu[:3])


    