import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.tf_utils import add_twist_to_pose


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization that modulates scale and shift based on phase."""
    
    def __init__(self, dim, phase_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Project phase to scale and shift parameters
        self.phase_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(phase_dim, 2 * dim, bias=True)
        )
        
    def forward(self, x, phase_emb):
        """
        Args:
            x: input tensor [batch, seq_len, dim]
            phase_emb: phase embedding [batch, phase_dim]
        """
        # Get modulation parameters
        phase_params = self.phase_proj(phase_emb)  # [batch, 2*dim]
        scale, shift = phase_params.chunk(2, dim=-1)  # Each [batch, dim]
        
        # Apply layer norm and modulate
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN for phase conditioning."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, phase_dim=256):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(dim, phase_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = AdaptiveLayerNorm(dim, phase_dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x, phase_emb):
        """
        Args:
            x: input tensor [batch, seq_len, dim]
            phase_emb: phase embedding [batch, phase_dim]
        """
        # Attention block with residual
        x = x + self.attn(self.norm1(x, phase_emb))
        # MLP block with residual
        x = x + self.mlp(self.norm2(x, phase_emb))
        return x


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for phase/time."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: phase values [batch] or [batch, 1]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


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
        max_seq_len=1024
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

        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
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
    def sample(self, x0, num_steps=100, method='euler'):
        """
        Generate samples using ODE integration.
        
        Args:
            x0: initial noise samples [batch, seq_len, input_dim]
            num_steps: number of integration steps
            method: integration method ('euler' or 'midpoint')
            
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
            x = add_twist_to_pose(x, v, dt)
        
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
                'max_seq_len': self.max_seq_len
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
    def inference(self, start_poses, num_steps=100, return_trajectory=False):
        """
        Generate goal poses from start poses using the trained flow model.
        
        Args:
            start_poses: starting poses as tensors [batch, 7] in quaternion format
                        (x, y, z, qw, qx, qy, qz) or [batch, seq_len, 7]
            num_steps: number of ODE integration steps
            return_trajectory: if True, return full trajectory; if False, only final poses
            
        Returns:
            If return_trajectory=False:
                goal_poses: final poses [batch, 7] or [batch, seq_len, 7]
            If return_trajectory=True:
                trajectory: all intermediate poses [batch, num_steps+1, 7] or [batch, num_steps+1, seq_len, 7]
        """
        self.eval()
        
        # Handle input shape
        if start_poses.dim() == 2:
            # [batch, 7] -> [batch, 1, 7]
            x = start_poses.unsqueeze(1)
            squeeze_output = True
        else:
            # [batch, seq_len, 7]
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
            x = add_twist_to_pose(x, v, dt)
            
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