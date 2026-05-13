import torch
from omegaconf import OmegaConf

from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.visualization_utils import visualize_image_trajectory
from image_gen_trainer import (
    patch_encode, patch_decode,
    SEQ_LEN, PATCH_DIM, IMG_HW, NUM_CLASSES,
)


def load_model(checkpoint_path, device='cpu', model_config=None, conditional=False):
    if conditional:
        ModelClass = ConditionalFlowMatchingTransformerModel
    else:
        ModelClass = FlowMatchingTransformerModel
        if model_config is not None and 'obs_dim' in model_config:
            model_config = dict(model_config)
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


def generate_from_noise(model, batch_size, obs=None, num_steps=100,
                        return_trajectory=False, cfg_scale=3.0, device='cpu'):
    """
    Sample [B, SEQ_LEN, PATCH_DIM] noise and integrate the flow.

    Returns:
        If return_trajectory=False: images [B, 1, 28, 28]
        If return_trajectory=True:  trajectory [B, T+1, 1, 28, 28]
    """
    model.eval()
    x0 = torch.randn(batch_size, SEQ_LEN, PATCH_DIM, device=device)

    is_conditional = isinstance(model, ConditionalFlowMatchingTransformerModel)
    with torch.no_grad():
        if is_conditional:
            result = model.inference(
                x0, obs, num_steps=num_steps, return_trajectory=return_trajectory,
                cfg_scale=cfg_scale, manifold='euclidean',
            )
        else:
            result = model.inference(
                x0, num_steps=num_steps, return_trajectory=return_trajectory,
                manifold='euclidean',
            )

    if return_trajectory:
        # result: [B, T+1, SEQ_LEN, PATCH_DIM]
        B, T, S, D = result.shape
        flat = result.reshape(B * T, S, D)
        imgs = patch_decode(flat).view(B, T, 1, IMG_HW, IMG_HW)
        return imgs
    else:
        return patch_decode(result)  # [B, 1, 28, 28]


def build_obs_from_digit(digit, batch_size, device):
    """One-hot class obs of shape [B, 1, NUM_CLASSES]."""
    if not 0 <= digit <= 9:
        raise ValueError(f"--digit must be 0-9, got {digit}")
    obs = torch.zeros(batch_size, NUM_CLASSES, device=device)
    obs[:, digit] = 1.0
    return obs.unsqueeze(1)  # [B, 1, NUM_CLASSES]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Inference with MNIST flow matching model")
    parser.add_argument('--digit', '-D', type=int, default=None,
                        help='Digit class (0-9) for class-conditional generation. Omit for unconditional (null token).')
    parser.add_argument('--conditional', '-C', action='store_true')
    parser.add_argument('--checkpoint_epoch', '-CE', type=int, default=10)
    parser.add_argument('--checkpoint_path', '-CP', type=str, default=None,
                        help="Path to checkpoint directory (default: checkpoints/)")
    parser.add_argument('--num_samples', '-N', type=int, default=8)
    parser.add_argument('--num_steps', '-STEPS', type=int, default=100)
    parser.add_argument('--return_trajectory', '-RT', action='store_true')
    parser.add_argument('--no_ot', '-NOOT', action='store_true')
    parser.add_argument('--cfg_scale', '-CFG', type=float, default=3.0)
    parser.add_argument('--no_cfg', '-NOCFG', action='store_true')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional path to save the trajectory tile figure.')
    return parser.parse_args()


def get_config_and_checkpoint_paths(args):
    base_path = args.checkpoint_path if args.checkpoint_path else "checkpoints/"
    ot_suffix = '_NOOT' if args.no_ot else '_OT'
    cfg_suffix = '_NOCFG' if args.no_cfg else '_CFG'
    model_name = 'cond_image_flow_matching_model' if args.conditional else 'image_flow_matching_model'
    config_path = f"{base_path}{model_name}{ot_suffix}{cfg_suffix}_training_config.yaml"
    checkpoint_path = f"{base_path}{model_name}{ot_suffix}{cfg_suffix}_epoch_{args.checkpoint_epoch}.pt"
    return config_path, checkpoint_path


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    args = parse_args()
    cfg_scale = 1.0 if args.no_cfg else args.cfg_scale
    print(f"Using CFG scale: {cfg_scale}")

    config_path, checkpoint_path = get_config_and_checkpoint_paths(args)
    try:
        config = OmegaConf.load(config_path)
        print(f"Loaded training config from {config_path}")
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Train the model first with image_gen_trainer.py.")
        exit(1)

    model_config = OmegaConf.to_container(config.model, resolve=True)

    try:
        model, _ = load_model(
            checkpoint_path, device=device, model_config=model_config,
            conditional=args.conditional,
        )
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Train the model first with image_gen_trainer.py.")
        exit(1)

    if args.conditional and args.digit is not None:
        obs = build_obs_from_digit(args.digit, args.num_samples, device)
        print(f"Class-conditional generation for digit {args.digit}.")
    elif args.conditional:
        obs = None
        print("Conditional model with no --digit: using null token (unconditional sample).")
    else:
        obs = None

    print("=" * 60)
    print("EXAMPLE 1: Generate samples from noise")
    print("=" * 60)
    imgs = generate_from_noise(
        model, args.num_samples, obs=obs, num_steps=args.num_steps,
        return_trajectory=False, cfg_scale=cfg_scale, device=device,
    )
    print(f"Generated images: shape={tuple(imgs.shape)} "
          f"min={imgs.min().item():.3f} max={imgs.max().item():.3f}")

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generate trajectory and tile noise -> denoised")
    print("=" * 60)
    traj = generate_from_noise(
        model, args.num_samples, obs=obs, num_steps=args.num_steps,
        return_trajectory=True, cfg_scale=cfg_scale, device=device,
    )
    print(f"Trajectory shape: {tuple(traj.shape)}  "
          f"[batch, T+1, 1, {IMG_HW}, {IMG_HW}]")
    title = f"Digit {args.digit}" if (args.conditional and args.digit is not None) else "Unconditional MNIST"
    visualize_image_trajectory(
        traj, num_samples=min(args.num_samples, 8), num_timesteps=10,
        save_path=args.save_path, title=title,
    )

    print("\n" + "=" * 60)
    print("Inference complete.")
    print("=" * 60)
