import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from models.flow_matching_transformer import FlowMatchingTransformerModel
from utils.train_utils import train
from omegaconf import OmegaConf
from utils.logging_utils import _Tee


# MNIST tokenization: 28x28 -> 7x7 grid of 4x4 patches -> seq_len=49, patch_dim=16
PATCH_SIZE = 4
IMG_HW = 28
SEQ_LEN = (IMG_HW // PATCH_SIZE) ** 2  # 49
PATCH_DIM = PATCH_SIZE * PATCH_SIZE     # 16
NUM_CLASSES = 10


def patch_encode(images):
    """[B, 1, 28, 28] -> [B, 49, 16] (row-major patch order)."""
    B, C, H, W = images.shape
    assert C == 1 and H == IMG_HW and W == IMG_HW
    x = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    # x shape: [B, 1, 7, 7, 4, 4]
    x = x.contiguous().view(B, 1, SEQ_LEN, PATCH_DIM)
    return x.squeeze(1)  # [B, 49, 16]


def patch_decode(patches):
    """[B, 49, 16] -> [B, 1, 28, 28]."""
    B, S, D = patches.shape
    assert S == SEQ_LEN and D == PATCH_DIM
    grid = IMG_HW // PATCH_SIZE
    # [B, 49, 16] -> [B, 7, 7, 4, 4] -> [B, 1, 28, 28]
    x = patches.view(B, grid, grid, PATCH_SIZE, PATCH_SIZE)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, IMG_HW, IMG_HW)
    return x


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Flow Matching Transformer on MNIST")
    parser.add_argument('--num_epochs', '-E', type=int, default=10)
    parser.add_argument('--num_batches_per_epoch', '-BE', type=int, default=200)
    parser.add_argument('--batch_size', '-B', type=int, default=128)
    parser.add_argument('--conditional', '-C', action='store_true',
                        help='Class-conditional generation on digit labels 0-9')
    parser.add_argument('--n_steps', type=int, default=10,
                        help='Number of interpolation steps per trajectory')
    parser.add_argument('--save_path', type=str, default='checkpoints/')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--no_ot', '-NOOT', action='store_true')
    parser.add_argument('--no_cfg', '-NOCFG', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


def build_config(args):
    ot_suffix = '_NOOT' if args.no_ot else '_OT'
    cfg_suffix = '_NOCFG' if args.no_cfg else '_CFG'
    if args.conditional:
        save_path = os.path.join(args.save_path, f'cond_image_flow_matching_model{ot_suffix}{cfg_suffix}')
        obs_dim = NUM_CLASSES
    else:
        save_path = os.path.join(args.save_path, f'image_flow_matching_model{ot_suffix}{cfg_suffix}')
        obs_dim = None

    training_config = {
        'conditional': args.conditional,
        'manifold': 'euclidean',
        'use_ot': not args.no_ot,
        'use_cfg': not args.no_cfg,
        'num_epochs': args.num_epochs,
        'num_batches_per_epoch': args.num_batches_per_epoch,
        'batch_size': args.batch_size,
        'n_interp_steps': args.n_steps,
        'seq_len': SEQ_LEN,
        'save_path': save_path,
        'patch_size': PATCH_SIZE,
        'image_hw': IMG_HW,
        'num_classes': NUM_CLASSES,
        'lr': 2e-4,
        'weight_decay': 1e-5,
    }
    model_config = {
        'input_dim': PATCH_DIM,
        'output_dim': PATCH_DIM,
        'obs_dim': obs_dim,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'phase_dim': 128,
        'max_seq_len': SEQ_LEN,
    }
    config = OmegaConf.create({'training': training_config, 'model': model_config})

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    config_save_path = save_path + '_training_config.yaml'
    OmegaConf.save(config, config_save_path)
    print(f"Training and model configuration saved to {config_save_path}\n")
    return config


def build_dataloader(args):
    # Normalize roughly to [-1, 1] so noise scale (unit Gaussian) matches data scale.
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(args.data_root, train=True, download=True, transform=tf)
    return DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )


if __name__ == "__main__":
    args = parse_args()

    _ot = '_NOOT' if args.no_ot else '_OT'
    _cfg = '_NOCFG' if args.no_cfg else '_CFG'
    _name = 'cond_image_flow_matching_model' if args.conditional else 'image_flow_matching_model'
    _tee = _Tee(os.path.join(args.save_path, f'{_name}{_ot}{_cfg}') + '_log.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    config = build_config(args)
    dataloader = build_dataloader(args)

    print("=" * 60)
    print("TRAINING FLOW MATCHING TRANSFORMER ON MNIST")
    print("=" * 60)

    if args.conditional:
        model = ConditionalFlowMatchingTransformerModel(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            obs_dim=config.model.obs_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            phase_dim=config.model.phase_dim,
            max_seq_len=config.model.max_seq_len,
        ).to(device)
    else:
        model = FlowMatchingTransformerModel(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            phase_dim=config.model.phase_dim,
            max_seq_len=config.model.max_seq_len,
        ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    loss_history = train(
        model=model,
        optimizer=optimizer,
        num_epochs=config.training.num_epochs,
        num_batches_per_epoch=config.training.num_batches_per_epoch,
        batch_size=config.training.batch_size,
        n_steps=config.training.n_interp_steps,
        use_ot=config.training.use_ot,
        use_cfg=config.training.use_cfg,
        device=device,
        save_path=config.training.save_path,
        manifold='euclidean',
        dataloader=dataloader,
        num_classes=NUM_CLASSES,
        patch_encode=patch_encode,
    )

    print("\nLoss history:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"Epoch {epoch}: {loss:.6f}")

    _tee.close()
