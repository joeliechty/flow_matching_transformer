import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.tf_utils import add_twist_to_pose
from utils.euclid_utils import add_velocity_to_state
from models.support_models import AdaptiveLayerNorm, SinusoidalPosEmb, TransformerBlock, get_2d_sincos_pos_embed


def _step_state(x, v, dt, manifold):
    if manifold == 'se3':
        return add_twist_to_pose(x, v, dt)
    elif manifold == 'euclidean':
        return add_velocity_to_state(x, v, dt)
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}. Expected 'se3' or 'euclidean'.")

# class GaussianFourierProjection(nn.Module):
#     """
#     Project coordinates into a higher dimensional space using high-freq sinusoidal features.
#     Standard trick from Tancik et al. (NeurIPS 2020) / Song et al. (ICLR 2021).
#     """
#     def __init__(self, input_dim, embed_dim, scale=10.0):
#         super().__init__()
#         # Randomly sampled weights (fixed during training)
#         # scale: Higher values = higher frequency sensitivity
#         self.W = nn.Parameter(torch.randn(embed_dim // 2, input_dim) * scale, requires_grad=False)

#     def forward(self, x):
#         x_proj = (2 * torch.pi * x) @ self.W.T
#         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FlowMatchingTransformerModel(nn.Module):
    """
    Flow Matching Transformer with Adaptive Layer Normalization.
    
    This model uses AdaLN to inject phase information into the transformer blocks,
    enabling proper conditioning on the flow matching phase parameter.
    """
    
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        phase_dim=256,
        max_seq_len=1024,
        pos_emb_type='1d_learned',
        pos_emb_grid=None,
    ):
        """
        Initialize the Flow Matching Transformer model.
        Args:
            input_dim: dimension of input features
            hidden_dim: dimension of transformer hidden states
            num_layers: number of transformer layers
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dim to model dim
            dropout: dropout rate
            phase_dim: dimension of phase embeddings
            max_seq_len: maximum sequence length for positional embeddings
            pos_emb_type: '1d_learned' (default) or '2d_sincos'. The 2D variant uses
                fixed sin/cos embeddings over a (rows, cols) grid; required for image
                patches so the transformer gets spatial inductive bias for free.
            pos_emb_grid: (grid_h, grid_w) tuple required when pos_emb_type='2d_sincos'.
                grid_h * grid_w must equal max_seq_len.
        """
        super().__init__()
        self.input_dim = input_dim

        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.phase_dim = phase_dim
        self.max_seq_len = max_seq_len
        self.pos_emb_type = pos_emb_type
        self.pos_emb_grid = tuple(pos_emb_grid) if pos_emb_grid is not None else None

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional embeddings
        if pos_emb_type == '1d_learned':
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        elif pos_emb_type == '2d_sincos':
            if self.pos_emb_grid is None:
                raise ValueError("pos_emb_grid=(H, W) is required when pos_emb_type='2d_sincos'")
            gh, gw = self.pos_emb_grid
            if gh * gw != max_seq_len:
                raise ValueError(f"pos_emb_grid {gh}x{gw} must match max_seq_len={max_seq_len}")
            self.register_buffer('pos_emb', get_2d_sincos_pos_embed(hidden_dim, gh, gw))
        else:
            raise ValueError(f"Unknown pos_emb_type: {pos_emb_type!r}")
        
        # Phase embedding (converts scalar phase to embedding)
        self.phase_emb = nn.Sequential(
            SinusoidalPosEmb(phase_dim),
            nn.Linear(phase_dim, phase_dim),
            nn.SiLU(),
            nn.Linear(phase_dim, phase_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                phase_dim=phase_dim
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.final_norm = AdaptiveLayerNorm(hidden_dim, phase_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        if self.pos_emb_type == '1d_learned':
            nn.init.normal_(self.pos_emb, std=0.02)

        # Zero-initialize output projection for better training stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x, phase):
        """
        Forward pass through the flow matching transformer.
        
        Args:
            x: input tensor [batch, seq_len, input_dim]
            phase: flow matching phase parameter [batch] or [batch, 1], 
                   typically in range [0, 1]
        
        Returns:
            output: predicted vector field [batch, seq_len, input_dim]
        """
        B, N, _ = x.shape
        
        # Ensure phase is the right shape
        if phase.dim() == 1:
            phase = phase.view(-1)
        elif phase.dim() == 2:
            phase = phase.squeeze(-1)
            
        # Project input
        x = self.input_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_emb[:, :N, :]
        
        # Get phase embeddings
        phase_emb = self.phase_emb(phase)  # [batch, phase_dim]
        
        # Apply transformer blocks with phase conditioning
        for block in self.blocks:
            x = block(x, phase_emb)
        
        # Final normalization and projection
        x = self.final_norm(x, phase_emb)
        x = self.output_proj(x)
        
        return x
    
    def cfm_loss(self, x_t, t, v_target, reduction='mean'):
        """
        Compute conditional flow matching loss using optimal transport path.
        
        Args:
            x0: source samples [batch, seq_len, input_dim]
            x1: target samples [batch, seq_len, input_dim]
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: flow matching loss
        """
        
        # Predict vector field
        v_pred = self.forward(x_t, t)
        
        # Compute MSE loss
        loss = F.mse_loss(v_pred, v_target, reduction=reduction)
        
        return loss
    
    @torch.no_grad()
    def sample(self, x0, num_steps=100, method='euler', manifold='se3'):
        """
        Generate samples using ODE integration.

        Args:
            x0: initial noise samples [batch, seq_len, input_dim]
            num_steps: number of integration steps
            method: integration method ('euler' or 'midpoint')
            manifold: 'se3' (default) or 'euclidean' — controls the integrator.

        Returns:
            x1: generated samples [batch, seq_len, input_dim]
        """
        device = x0.device
        B = x0.shape[0]

        x = x0.clone()
        dt = torch.tensor(1.0 / num_steps, device=device)

        for step in range(num_steps):
            t = torch.full((B,), step * dt.item(), device=device)

            # Euler method
            v = self.forward(x, t)
            x = _step_state(x, v, dt, manifold)

        return x
    
    def save_checkpoint(self, filepath, optimizer=None, epoch=None, loss=None, **extra_info):
        """
        Save model checkpoint.
        
        Args:
            filepath: path to save checkpoint
            optimizer: optional optimizer state to save
            epoch: optional epoch number
            loss: optional loss value
            **extra_info: any additional information to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'mlp_ratio': self.mlp_ratio,
                'dropout': self.dropout,
                'phase_dim': self.phase_dim,
                'max_seq_len': self.max_seq_len,
                'pos_emb_type': self.pos_emb_type,
                'pos_emb_grid': list(self.pos_emb_grid) if self.pos_emb_grid is not None else None,
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        
        # Add any extra information
        checkpoint.update(extra_info)
        
        torch.save(checkpoint, filepath)
        
    @classmethod
    def load_checkpoint(cls, filepath, device='cpu', optimizer=None, model_config=None):
        """
        Load model from checkpoint.
        
        Args:
            filepath: path to checkpoint file
            device: device to load model on
            optimizer: optional optimizer to load state into
            model_config: optional model config dict. Required if checkpoint doesn't contain model_config
                         (for backward compatibility with old checkpoints)
            
        Returns:
            model: loaded model
            checkpoint: full checkpoint dict with epoch, loss, etc.
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Get model config from checkpoint or parameter
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        elif model_config is not None:
            config = model_config
        else:
            raise ValueError(
                "Checkpoint does not contain 'model_config'. "
                "Please provide model_config parameter to load_checkpoint() with the model configuration."
            )
        
        # Create model from config
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, checkpoint
    
    @torch.no_grad()
    def inference(self, start_poses, num_steps=100, return_trajectory=False, manifold='se3'):
        """
        Generate goal poses from start poses using the trained flow model.

        Args:
            start_poses: starting states. For manifold='se3': poses in quaternion
                format [batch, 7] or [batch, seq_len, 7]. For manifold='euclidean':
                noise samples [batch, input_dim] or [batch, seq_len, input_dim].
            num_steps: number of ODE integration steps
            return_trajectory: if True, return full trajectory; if False, only final state
            manifold: 'se3' (default) or 'euclidean'.

        Returns:
            If return_trajectory=False:
                final state [batch, D] or [batch, seq_len, D]
            If return_trajectory=True:
                trajectory: all intermediate states [batch, num_steps+1, D] or
                [batch, num_steps+1, seq_len, D]
        """
        self.eval()

        # Handle input shape
        if start_poses.dim() == 2:
            # [batch, D] -> [batch, 1, D]
            x = start_poses.unsqueeze(1)
            squeeze_output = True
        else:
            x = start_poses
            squeeze_output = False

        device = x.device
        B = x.shape[0]
        dt = torch.tensor(1.0 / num_steps, device=device)

        if return_trajectory:
            trajectory = [x.clone()]

        # Integrate ODE from t=0 to t=1
        for step in range(num_steps):
            t = torch.full((B,), step * dt.item(), device=device)
            v = self.forward(x, t)
            x = _step_state(x, v, dt, manifold)

            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            # Stack trajectory: [batch, num_steps+1, seq_len, 7]
            trajectory = torch.stack(trajectory, dim=1)
            if squeeze_output:
                # Remove seq_len dimension: [batch, num_steps+1, 7]
                trajectory = trajectory.squeeze(2)
            return trajectory
        else:
            if squeeze_output:
                # Remove seq_len dimension: [batch, 7]
                x = x.squeeze(1)
            return x