import torch
import os
import random
import numpy as np
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.train_utils import generate_interpolated_poses, train
from omegaconf import OmegaConf
from utils.logging_utils import _Tee

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching Transformer Model")
    parser.add_argument('--num_epochs', '-E', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_batches_per_epoch', '-BE', type=int, default=100, help='Number of minibatches per epoch')
    parser.add_argument('--batch_size', '-B', type=int, default=None, help='Number of samples in each minibatch')
    parser.add_argument('--conditional', '-C', action='store_true', help='Whether to train conditional model (with observations)')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of interpolation steps per trajectory')
    parser.add_argument('--save_path', type=str, default='checkpoints/', help='Path to save model checkpoints')
    parser.add_argument('--seq_len', '-S', type=int, default=1, help='Sequence length per trajectory')
    parser.add_argument('--no_ot', '-NOOT', action='store_true', help='Disable optimal transport pairing during training')
    parser.add_argument('--no_cfg', '-NOCFG', action='store_true', help='Disable classifier-free guidance (no unconditional dropout during training)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for torch/numpy/random (for reproducible ablations)')
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_training_and_model_config(args, start_dist_params=None, goal_dist_params=None, action_dist_params=None):
    # Define distribution parameters (twist representation)
    if start_dist_params is None:
        start_dist_params = {
            'mu': [[0, 0, 0, 0, 0, 0]],
            'sigma': [[1, 1, 1, 1, 1, 1]]
        }
    
    # goal pos at (5,5,5) with 90 deg rotation around z axis (twist representation)
    if goal_dist_params is None:
        goal_dist_params = {
            'mu': [
                [[5, 5, 5, 0, 0, 1.5708]],    # Top-right with 90 deg rotation
                [[5, 5, -5, 0, 0, -1.5708]],  # Bottom-right with -90 deg rotation
                [[5, -5, 5, 0, 0, 3.14159]],  # Top-left with 180 deg rotation
                [[5, -5, -5, 0, 0, 3.14159]]  # Bottom-left with 180 deg rotation
                   ],
            'sigma': [
                [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
                      ]
        }

    if args.conditional:
        if action_dist_params is None:
            # the "go up" twist and the "go down" twist
            t_v, b_v = [0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]
            r_v, l_v = [0, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0]

            action_dist_params = {
                'mu': [
                    [t_v, r_v], # TR matches goal index 0 (Top-right)
                    [b_v, r_v], # BR matches goal index 1 (Bottom-right)
                    [t_v, l_v], # TL matches goal index 2 (Top-left)
                    [b_v, l_v]  # BL matches goal index 3 (Bottom-left)
                ],
                'sigma': [[[0.0]*6, [0.0]*6]] * 4
            }

    if args.batch_size is None:
        batch_size = len(goal_dist_params['mu'])*32
    else:
        batch_size = args.batch_size

    ot_suffix = '_NOOT' if args.no_ot else '_OT'
    cfg_suffix = '_NOCFG' if args.no_cfg else '_CFG'
    if args.conditional:
        save_path = os.path.join(args.save_path, f'cond_pose_flow_matching_model{ot_suffix}{cfg_suffix}')
        obs_dim = 6  # action representation (vx, vy, vz, wx, wy, wz)
    else:
        save_path = os.path.join(args.save_path, f'pose_flow_matching_model{ot_suffix}{cfg_suffix}')
        obs_dim = None

    # make a config object
    training_config = {
        'conditional': args.conditional,
        'use_ot': not args.no_ot,
        'use_cfg': not args.no_cfg,
        'seed': args.seed,
        'num_epochs': args.num_epochs,
        'num_batches_per_epoch': args.num_batches_per_epoch,
        'batch_size': batch_size,
        'n_interp_steps': args.n_steps,
        'seq_len': args.seq_len,
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
        'max_seq_len': args.seq_len
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
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    _ot_suffix = '_NOOT' if args.no_ot else '_OT'
    _cfg_suffix = '_NOCFG' if args.no_cfg else '_CFG'
    _model_name = 'cond_pose_flow_matching_model' if args.conditional else 'pose_flow_matching_model'
    _log_root = os.path.join(args.save_path, f'{_model_name}{_ot_suffix}{_cfg_suffix}')
    _tee = _Tee(_log_root + '_log.txt')

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
        seq_len=config.training.seq_len,
        use_ot=config.training.use_ot,
        use_cfg=config.training.use_cfg,
        device=device,
        save_path=config.training.save_path
    )
    
    # Plot loss history
    print("\nLoss history:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: {loss:.6f}")

    _tee.close()





