import torch
import os
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from models.flow_matching_transformer import FlowMatchingTransformerModel
from scipy.optimize import linear_sum_assignment
import numpy as np
from utils.train_utils import geodesic_optimal_transport_pairing, generate_interpolated_poses




def train_one_minibatch(model, optimizer, batch_size, n_steps, start_dist_params, goal_dist_params, device='cpu'):
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
    
    # Sample start poses from start distribution
    start_poses = sample_random_twist(
        batch_size=batch_size,
        mu=start_dist_params['mu'],
        sigma=start_dist_params['sigma'],
        device=device
    )  # [batch_size, 6]
    
    # Sample goal poses from goal distribution
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
    # goal_poses = sample_random_twist(
    #     batch_size=batch_size,
    #     mu=goal_dist_params['mu'],
    #     sigma=goal_dist_params['sigma'],
    #     device=device
    # )  # [batch_size, 6]

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
    
    # Compute loss
    loss = model.cfm_loss(x_t, t_flat, v_target, reduction='mean')
    
    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_one_epoch(model, optimizer, num_batches, batch_size, n_steps, start_dist_params, goal_dist_params, device='cpu'):
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
        device: device to run on
        
    Returns:
        avg_loss: average loss over the epoch
    """
    total_loss = 0.0
    
    for batch_idx in range(num_batches):
        loss = train_one_minibatch(
            model, optimizer, batch_size, n_steps,
            start_dist_params, goal_dist_params, device
        )
        total_loss += loss
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train(model, optimizer, num_epochs, num_batches_per_epoch, batch_size, n_steps, 
          start_dist_params, goal_dist_params, device='cpu', save_path=None):
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
            start_dist_params, goal_dist_params, device
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

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test data generation
    print("=" * 60)
    print("TESTING DATA GENERATION")
    print("=" * 60)
    
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
    
    # Training setup
    print("=" * 60)
    print("TRAINING FLOW MATCHING TRANSFORMER")
    print("=" * 60)
    
    # Define distribution parameters (twist representation)
    start_dist_params = {
        'mu': [0, 0, 0, 0, 0, 0],
        'sigma': [1, 1, 1, 1, 1, 1]
    }
    
    # goal pos at (5,5,5) with 90 deg rotation around z axis (twist representation)
    goal_dist_params = {
        'mu': [[5, 5, 5, 0, 0, 1.5708],[5, 5, -5, 0, 0, -1.5708]],
        'sigma': [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    }

    # Training hyperparameters
    num_epochs = 10
    num_batches_per_epoch = 100
    batch_size = len(goal_dist_params['mu'])*32
    n_interpolation_steps = 10
    
    # Model configuration
    model = FlowMatchingTransformerModel(
        input_dim=7,  # quaternion pose representation (x, y, z, qw, qx, qy, qz)
        output_dim=6,  # twist representation (vx, vy, vz, wx, wy, wz)
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        phase_dim=128,
        max_seq_len=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Train the model
    loss_history = train(
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        batch_size=batch_size,
        n_steps=n_interpolation_steps,
        start_dist_params=start_dist_params,
        goal_dist_params=goal_dist_params,
        device=device,
        save_path='checkpoints/flow_matching_model_OT'
    )
    
    # Plot loss history
    print("\nLoss history:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: {loss:.6f}")
    
    # Test sampling
    print("\n" + "=" * 60)
    print("TESTING SAMPLING")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # Sample from start distribution
        test_start_poses = sample_random_twist(
            batch_size=5,
            mu=start_dist_params['mu'],
            sigma=start_dist_params['sigma'],
            device=device
        )
        
        # Convert to pose representation
        x0 = convert_twist_to_pose(test_start_poses, dt=1.0, return_representation='quat')
        x0 = x0.unsqueeze(1)  # Add sequence dimension [5, 1, 7]
        
        print("Initial poses (from start distribution):")
        print(x0.squeeze(1))
        
        # Generate samples by flowing to goal distribution
        x1 = model.sample(x0, num_steps=50, method='euler')
        
        print("\nGenerated poses (should be near goal distribution):")
        print(x1.squeeze(1)[:, :3])  # Print positions
        
        print("\nExpected goal positions around:", goal_dist_params['mu'][:3])


    