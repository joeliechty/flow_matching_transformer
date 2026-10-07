import torch
import os
import random
import numpy as np
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, compute_twist_between_poses, add_twist_to_pose
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.train_utils import PAIRINGS, Pairer, generate_interpolated_poses, sample_pose_batch, train
from omegaconf import OmegaConf
from utils.logging_utils import _Tee, git_commit
from utils.pose_task import ContinuousGoalTask, load_pose_task

DEFAULT_TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'configs', 'pose_tasks', 'four_corners.yaml')

# Checkpoint-name suffix of each noise-data pairing (see `utils.train_utils.PAIRINGS`).
PAIRING_SUFFIX = {'independent': '_NOOT', 'ot': '_OT', 'global': '_GOT', 'c2ot': '_C2OT',
                  'c2ot_fixed': '_C2OTFIX', 'cluster': '_CLUSTER'}

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
    parser.add_argument('--task_config', type=str, default=DEFAULT_TASK,
                        help='Pose task file: start/goal distributions and conditioning tokens')
    parser.add_argument('--pairing', type=str, default=None, choices=PAIRINGS,
                        help="Noise-data pairing (default: 'ot', or 'independent' with --no_ot). "
                             "'global', 'c2ot', 'c2ot_fixed' and 'cluster' are for continuous conditions")
    parser.add_argument('--r_tar', type=float, default=0.01,
                        help="c2ot: target share of admissible pairs that sets the condition weight")
    parser.add_argument('--ot_batch_mult', type=int, default=1,
                        help='Pair over this many network batches at once (the OT batch)')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='cluster: K-means clusters of the conditions (default: the OT batch size)')
    parser.add_argument('--cond_scale', type=float, default=10.0,
                        help='c2ot_fixed / cluster: condition weight as a multiple of mean sample '
                             'cost / mean condition distance (papers: 10)')
    args = parser.parse_args()
    if args.pairing is None:
        args.pairing = 'independent' if args.no_ot else 'ot'
    elif args.no_ot and args.pairing != 'independent':
        parser.error('--no_ot contradicts --pairing ' + args.pairing)
    if not args.conditional and args.pairing not in ('independent', 'ot', 'global'):
        parser.error(f'--pairing {args.pairing} needs conditions; add --conditional')
    return args


def run_name(args):
    """Checkpoint stem, e.g. 'cond_pose_flow_matching_model_C2OT_NOCFG'."""
    model = 'cond_pose_flow_matching_model' if args.conditional else 'pose_flow_matching_model'
    return f"{model}{PAIRING_SUFFIX[args.pairing]}{'_NOCFG' if args.no_cfg else '_CFG'}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_training_and_model_config(args, start_dist_params=None, goal_dist_params=None, action_dist_params=None):
    # Distribution parameters (twist representation) come from the task file unless given.
    task = load_pose_task(args.task_config)
    continuous = task['type'] == 'continuous'
    if continuous and not args.conditional:
        raise ValueError(f"{args.task_config} is a continuous-condition task; add --conditional")
    task_name, mode_names = ((task['name'], task['mode_names']) if goal_dist_params is None
                             else ('custom', None))
    if start_dist_params is None:
        start_dist_params = task['start_dist_params']
    if goal_dist_params is None:
        goal_dist_params = task['goal_dist_params']
    if args.conditional and action_dist_params is None:
        action_dist_params = task['action_dist_params']

    if args.batch_size is None:
        batch_size = 128 if continuous else len(goal_dist_params['mu'])*32
    else:
        batch_size = args.batch_size

    save_path = os.path.join(args.save_path, run_name(args))
    # action token size (6 for twist actions)
    if continuous:
        obs_dim = task['obs_dim']
    else:
        obs_dim = len(action_dist_params['mu'][0][0]) if args.conditional else None

    # make a config object
    training_config = {
        'conditional': args.conditional,
        'use_ot': args.pairing != 'independent',
        'use_cfg': not args.no_cfg,
        'pairing': args.pairing,
        'r_tar': args.r_tar,
        'ot_batch_mult': args.ot_batch_mult,
        'num_clusters': args.num_clusters,
        'cond_scale': args.cond_scale,
        'seed': args.seed,
        'git_commit': git_commit(),
        'task': task_name,
        'task_type': task['type'],
        'task_spec': task['spec'] if continuous else None,
        'mode_names': mode_names,
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

    _tee = _Tee(os.path.join(args.save_path, run_name(args)) + '_log.txt')

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

    # The original 'ot' / 'independent' runs keep their exact training path (and RNG stream);
    # every other pairing, or a bigger OT batch, draws paired batches from a Pairer.
    pairer = None
    t = config.training
    if t.task_type == 'continuous':
        sampler = ContinuousGoalTask(OmegaConf.to_container(t.task_spec)).batch_sampler(
            t.start_dist_params, seq_len=t.seq_len, device=device)
        pairer = Pairer(t.pairing, sampler, t.batch_size, ot_batch_mult=t.ot_batch_mult,
                        r_tar=t.r_tar, num_clusters=t.num_clusters, cond_scale=t.cond_scale)
    elif t.pairing not in ('ot', 'independent') or t.ot_batch_mult > 1:
        sampler = lambda n: sample_pose_batch(n, t.start_dist_params, t.goal_dist_params,
                                              t.action_dist_params, seq_len=t.seq_len, device=device)
        pairer = Pairer(t.pairing, sampler, t.batch_size, ot_batch_mult=t.ot_batch_mult,
                        r_tar=t.r_tar, num_clusters=t.num_clusters, cond_scale=t.cond_scale)
    
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
        save_path=config.training.save_path,
        pairer=pairer,
    )
    
    # Plot loss history
    print("\nLoss history:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: {loss:.6f}")

    _tee.close()





