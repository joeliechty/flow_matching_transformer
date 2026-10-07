import os
import torch
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.tf_utils import sample_random_twist, convert_twist_to_pose, _quat_to_rot_mat
from utils.visualization_utils import visualize_trajectory, print_poses
from omegaconf import OmegaConf

# Action vocabulary of the four_corners task: name -> action twist token.
ACTION_MAP = {
    'top':    [0.0,  0.0, 1.0, 0.0, 0.0, 0.0],
    'bottom': [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    'right':  [0.0,  1.0, 0.0, 0.0, 0.0, 0.0],
    'left':   [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
}

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

def generate_from_start_poses(model, start_poses, obs=None, num_steps=100, return_trajectory=False, cfg_scale=3.0, device='cpu',
                              obs_mask=None):
    """
    Generate goal poses from start poses using the trained model.

    Args:
        model: trained FlowMatchingTransformerModel (conditional or non-conditional)
        start_poses: start poses as twists [batch, 6] or quaternions [batch, 7]
        obs: observation conditioning tensor [batch, obs_dim] (required for conditional models)
        obs_mask: optional bool [batch, M]; True replaces that obs token with the null token
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
                result = model.inference(x0, obs, num_steps=num_steps, return_trajectory=return_trajectory, cfg_scale=cfg_scale,
                                         obs_mask=obs_mask)
            else:
                result = model.inference(x0, num_steps=num_steps, return_trajectory=return_trajectory)
        except Exception as e:
            print("Error during model inference:", e)
            raise e

    return result

def generate_from_distribution(model, distribution_params, batch_size, obs=None, num_steps=100,
                               return_trajectory=False, cfg_scale=3.0, device='cpu', obs_mask=None):
    """
    Sample start poses from a distribution and generate goal poses.

    Args:
        model: trained FlowMatchingTransformerModel (conditional or non-conditional)
        distribution_params: dict with 'mu' and 'sigma' for twist distribution
        batch_size: number of samples to generate
        obs: tensor of observations
        obs_mask: optional bool [batch, M]; True replaces that obs token with the null token
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
        return_trajectory=return_trajectory, cfg_scale=cfg_scale, device=device,
        obs_mask=obs_mask,
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
    parser.add_argument('--cfg_scale', '-CFG', type=float, default=3.0, help="Classifier-free guidance scale (1.0 disables CFG, >1.0 amplifies conditioning)")
    parser.add_argument('--no_cfg', '-NOCFG', action='store_true', help="Disable classifier-free guidance at inference (equivalent to --cfg_scale 1.0)")
    # --compare_ot_fix: before/after-fix comparison figures (uses -CE, -N, -STEPS, -CFG above)
    parser.add_argument('--compare_ot_fix', action='store_true',
                        help="Save start->goal mapping figures comparing conditional models before vs after the OT pairing fix")
    parser.add_argument('--before_dir', type=str, default='checkpoints/pose_before_ot_fix/seed_1/',
                        help="Checkpoints from before the fix (for --compare_ot_fix)")
    parser.add_argument('--after_dir', type=str, default='checkpoints/pose/seed_1/',
                        help="Checkpoints from after the fix (for --compare_ot_fix)")
    parser.add_argument('--save_dir', type=str, default='experiments/results/pose/',
                        help="Results directory for --compare_ot_fix: epoch figures go to <save_dir>/epoch_<CE>/")
    parser.add_argument('--compare_seed', type=int, default=0, help="Seed for the shared start noise")
    parser.add_argument('--elev', type=float, default=18, help="3-D view elevation (degrees)")
    parser.add_argument('--azim', type=float, default=-35, help="3-D view azimuth (degrees)")
    parser.add_argument('--show', action='store_true', help="Show the figures instead of saving them")
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
    cfg_suffix = '_NOCFG' if args.no_cfg else '_CFG'
    model_name = 'cond_pose_flow_matching_model' if args.conditional else 'pose_flow_matching_model'

    config_path = f"{base_path}{model_name}{ot_suffix}{cfg_suffix}_training_config.yaml"
    checkpoint_path = f"{base_path}{model_name}{ot_suffix}{cfg_suffix}_epoch_{args.checkpoint_epoch}.pt"

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
    # goal_dist['mu'] entries may be doubly-nested ([[x,y,z,...]]) from YAML parsing
    mus = [m[0] if isinstance(m[0], list) else m for m in goal_dist['mu']]
    return [mu for mu in mus
            if all(conditions[a](mu) for a in actions if a in conditions)]

def build_obs_from_actions(actions, batch_size, device):
    """Dynamically builds [batch, M, 6] tensor based on requested conditions."""
    obs_tokens = []
    for a in actions:
        if a in ACTION_MAP:
            obs_tokens.append(ACTION_MAP[a])
        else:
            raise ValueError(f"Unknown action: {a}")

    # [M, 6] -> [1, M, 6] -> [batch, M, 6]
    obs_tensor = torch.tensor(obs_tokens, dtype=torch.float32, device=device)
    return obs_tensor.unsqueeze(0).repeat(batch_size, 1, 1)


# -- --compare_ot_fix: start -> goal mappings before vs after the conditional-OT fix ------

# (panel title, checkpoint dir: 'before' or 'after', checkpoint stem)
OT_FIX_PANELS = [
    ('OT, no CFG: before fix', 'before', 'cond_pose_flow_matching_model_OT_NOCFG'),
    ('OT, no CFG: after fix', 'after', 'cond_pose_flow_matching_model_OT_NOCFG'),
    ('no OT, no CFG (unaffected)', 'after', 'cond_pose_flow_matching_model_NOOT_NOCFG'),
    ('OT + CFG: before fix', 'before', 'cond_pose_flow_matching_model_OT_CFG'),
    ('OT + CFG: after fix', 'after', 'cond_pose_flow_matching_model_OT_CFG'),
    ('no OT + CFG (unaffected)', 'after', 'cond_pose_flow_matching_model_NOOT_CFG'),
]


def _condition_name(tokens, index):
    """'top + right' for a known token list, else 'condition <index>'."""
    names = {tuple(v): k for k, v in ACTION_MAP.items()}
    parts = [names.get(tuple(float(x) for x in t)) for t in tokens.tolist()]
    return ' + '.join(parts) if all(parts) else f'condition {index}'


def _sample_start_poses(start_dist, n, generator):
    mu = torch.tensor(start_dist['mu'], dtype=torch.float32).reshape(6)
    sigma = torch.tensor(start_dist['sigma'], dtype=torch.float32).reshape(6)
    twists = mu + sigma * torch.randn(n, 6, generator=generator)
    return convert_twist_to_pose(twists, dt=1.0, return_representation='quat')


def compare_ot_fix(args, device):
    """Three figures comparing conditional models before and after the OT pairing fix, all fed
    the same start noise, so any difference in the mappings comes from the model:
      mappings:  start -> goal sampling paths for each model in OT_FIX_PANELS
      goal_zoom: goal samples of one condition up close, with orientation frames, vs real data
      pairings:  how one training minibatch is OT-paired, globally (before) vs per condition (after)
    """
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.eval_utils import (goal_mode_poses_from_config, mode_offsets, pose_mode_distance,
                                  sample_goal_poses)
    from utils.pose_task import task_conditions
    from utils.train_utils import _mode_condition_ids, _pair_within_conditions, sequence_ot_pairing
    from utils.visualization_utils import plot_condition_mappings, plot_goal_cloud, plot_pairings

    dirs = {'before': args.before_dir, 'after': args.after_dir}
    epoch = args.checkpoint_epoch
    # Epoch-specific figures sit with that epoch's other results; the pairing figure doesn't
    # depend on the epoch.
    epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch}')

    def config_for(which, stem):
        return OmegaConf.load(os.path.join(dirs[which], f'{stem}_training_config.yaml'))

    def load(which, stem):
        config = config_for(which, stem)
        model, _ = load_model(os.path.join(dirs[which], f'{stem}_epoch_{epoch}.pt'), device=device,
                              model_config=OmegaConf.to_container(config.model, resolve=True),
                              conditional=True)
        return model

    # Task layout (identical for every panel) from the unaffected no-OT run.
    task = config_for('after', 'cond_pose_flow_matching_model_NOOT_NOCFG').training
    goal_dist = OmegaConf.to_container(task.goal_dist_params, resolve=True)
    start_dist = OmegaConf.to_container(task.start_dist_params, resolve=True)
    actions = OmegaConf.to_container(task.action_dist_params, resolve=True)
    mode_poses = goal_mode_poses_from_config(goal_dist)
    conditions, _ = task_conditions(actions)
    names = [_condition_name(c.tokens, i) for i, c in enumerate(conditions)]

    gen = torch.Generator().manual_seed(args.compare_seed)
    starts = [_sample_start_poses(start_dist, args.num_samples, gen) for _ in conditions]
    zoom_starts = _sample_start_poses(start_dist, 64, gen)

    def sample(model, stem, cond, start_poses, trajectory=True):
        scale = args.cfg_scale if stem.endswith('_CFG') else 1.0  # no-CFG models sampled unguided
        return generate_from_start_poses(model, start_poses, cond.obs(start_poses.shape[0], device),
                                         num_steps=args.num_steps, return_trajectory=trajectory,
                                         cfg_scale=scale, device=device).cpu()

    # 1. start -> goal mappings -------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    zoom_samples = {}
    for p, (title, which, stem) in enumerate(OT_FIX_PANELS):
        model = load(which, stem)
        trajs = [sample(model, stem, c, s) for c, s in zip(conditions, starts)]
        cond_idx = [i for i, t in enumerate(trajs) for _ in range(t.shape[0])]
        trajectory = torch.cat(trajs)
        targets = torch.tensor([conditions[i].valid_modes[0] for i in cond_idx])
        dist = pose_mode_distance(trajectory[:, -1], mode_poses, targets)
        if stem.endswith('NOCFG'):
            zoom_samples[(which, stem)] = sample(model, stem, conditions[0], zoom_starts, trajectory=False)
        ax = fig.add_subplot(2, 3, p + 1, projection='3d')
        guidance = f', guidance {args.cfg_scale:g}' if stem.endswith('_CFG') else ''
        plot_condition_mappings(ax, trajectory, cond_idx, mode_poses=mode_poses, condition_names=names,
                                title=f'{title}{guidance}\nmean twist distance to target {dist:.2f}')
        ax.set_xlim(-3, 7); ax.set_ylim(-7, 7); ax.set_zlim(-7, 7)
        ax.set_box_aspect((10, 14, 14))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.suptitle(f'Start -> goal mappings, epoch {epoch}: the same {args.num_samples} start poses per '
                 f'condition for every model', fontsize=13)
    _finish(fig, plt, args, os.path.join(epoch_dir, 'ot_fix_mappings.png'))

    # 2. goal samples of one condition, up close ------------------------------------
    real, real_idx = sample_goal_poses(goal_dist, 64, seed=args.compare_seed)
    k = conditions[0].valid_modes[0]
    panels = [('real goal samples', real[real_idx == k]),
              ('OT, no CFG: before fix', zoom_samples[('before', 'cond_pose_flow_matching_model_OT_NOCFG')]),
              ('OT, no CFG: after fix', zoom_samples[('after', 'cond_pose_flow_matching_model_OT_NOCFG')]),
              ('no OT, no CFG', zoom_samples[('after', 'cond_pose_flow_matching_model_NOOT_NOCFG')])]
    offsets = [mode_offsets(s, mode_poses[k]) for _, s in panels]
    # Row 1: one scale wide enough for every panel. Row 2: zoomed to the real data's spread.
    wide = max(0.4, 1.15 * max(torch.quantile(dp.norm(dim=-1), 0.95).item() for dp, _ in offsets))
    close = 4 * offsets[0][0].pow(2).sum(-1).mean().sqrt().item()
    fig = plt.figure(figsize=(18, 10))
    for row, radius in enumerate((wide, close)):
        for p, ((title, s), (dp, dr)) in enumerate(zip(panels, offsets)):
            ax = fig.add_subplot(2, 4, row * 4 + p + 1, projection='3d')
            rms_p = dp.pow(2).sum(-1).mean().sqrt().item()
            rms_r = dr.pow(2).sum(-1).mean().sqrt().item()
            label = (f'{title}\nposition error {rms_p:.2f}, rotation error {rms_r:.2f} rad (RMS)'
                     if row == 0 else f'{title}, zoomed')
            hidden = plot_goal_cloud(ax, s, mode_poses[k], radius,
                                     frame_length=0.12 if row == 0 else radius / 6, title=label)
            if hidden:
                ax.set_title(f'{label}\n({hidden} of {s.shape[0]} outside this view)', fontsize=10)
            ax.view_init(elev=args.elev, azim=args.azim)
    fig.suptitle(f'Goal samples for "{names[0]}" (64 each), epoch {epoch}; dark frame = mode centre, '
                 f'RGB = sample orientation x/y/z.\nTop row: same wide scale. '
                 f'Bottom row: zoomed to ±{close:.2f} (4× the real data\'s RMS position error)', fontsize=12)
    _finish(fig, plt, args, os.path.join(epoch_dir, 'ot_fix_goal_zoom.png'), hspace=0.32)

    # 3. training-time pairing of one minibatch ----------------------------------------
    mus = torch.tensor(goal_dist['mu'], dtype=torch.float32).reshape(-1, 6)
    sigmas = torch.tensor(goal_dist['sigma'], dtype=torch.float32).reshape(-1, 6)
    K, per_mode = mus.shape[0], 128 // mus.shape[0]
    start_tw = (torch.tensor(start_dist['mu'], dtype=torch.float32).reshape(6)
                + torch.tensor(start_dist['sigma'], dtype=torch.float32).reshape(6)
                * torch.randn(K * per_mode, 6, generator=gen)).unsqueeze(1)
    goal_tw = (mus.repeat_interleave(per_mode, 0) + sigmas.repeat_interleave(per_mode, 0)
               * torch.randn(K * per_mode, 6, generator=gen)).unsqueeze(1)
    cond_ids = torch.tensor(_mode_condition_ids(actions['mu'])).repeat_interleave(per_mode)
    se3 = lambda s, g: sequence_ot_pairing(s, g, manifold='se3')
    pairings = [('before fix: OT over the whole batch', se3(start_tw, goal_tw)),
                ('after fix: OT within each condition',
                 _pair_within_conditions(start_tw, goal_tw, cond_ids, se3))]
    goal_poses = convert_twist_to_pose(goal_tw[:, 0], dt=1.0, return_representation='quat')
    centroid = goal_poses[:, :3].mean(0)
    fig = plt.figure(figsize=(14, 6.5))
    for p, (title, paired) in enumerate(pairings):
        start_poses = convert_twist_to_pose(paired[:, 0], dt=1.0, return_representation='quat')
        # Mean offset of each condition's starts toward its own goals, in start-noise std units.
        toward = (goal_poses[:, :3] - centroid)
        toward = toward / toward.norm(dim=-1, keepdim=True)
        bias = (start_poses[:, :3] * toward).sum(-1).mean().item()
        ax = fig.add_subplot(1, 2, p + 1, projection='3d')
        plot_pairings(ax, start_poses, goal_poses, cond_ids.tolist(), condition_names=names,
                      title=f'{title}\nstarts sit {bias:+.2f} std toward their own goal')
        ax.set_xlim(-3, 7); ax.set_ylim(-7, 7); ax.set_zlim(-7, 7)
        ax.set_box_aspect((10, 14, 14))
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.suptitle(f'Training-time OT pairing of one {K * per_mode}-sample minibatch: '
                 f'start positions coloured by the condition of their paired goal', fontsize=12)
    _finish(fig, plt, args, os.path.join(args.save_dir, 'ot_fix_pairings.png'))


def _finish(fig, plt, args, path, hspace=None):
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    if hspace is not None:  # tight_layout under-spaces rows of 3-D axes
        fig.subplots_adjust(hspace=hspace)
    if args.show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    args = parse_args()

    if args.compare_ot_fix:
        compare_ot_fix(args, device)
        raise SystemExit(0)

    # Resolve effective CFG scale (--no_cfg forces 1.0, otherwise honor --cfg_scale)
    cfg_scale = 1.0 if args.no_cfg else args.cfg_scale
    print(f"Using CFG scale: {cfg_scale}")

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
            model, start_poses, obs, num_steps=args.num_steps, return_trajectory=False,
            cfg_scale=cfg_scale, device=device
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
        return_trajectory=True, cfg_scale=cfg_scale, device=device
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
        num_steps=args.num_steps, return_trajectory=False, cfg_scale=cfg_scale, device=device
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
