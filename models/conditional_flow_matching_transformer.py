import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tf_utils import add_twist_to_pose
from utils.euclid_utils import add_velocity_to_state
from models.support_models import AdaptiveLayerNorm, SinusoidalPosEmb, TransformerBlock


def _step_state(x, v, dt, manifold):
    if manifold == 'se3':
        return add_twist_to_pose(x, v, dt)
    elif manifold == 'euclidean':
        return add_velocity_to_state(x, v, dt)
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}. Expected 'se3' or 'euclidean'.")


class ConditionalFlowMatchingTransformerModel(nn.Module):
    """
    Conditional Flow Matching Transformer with Adaptive Layer Normalization.
    
    This model uses AdaLN to inject phase information into the transformer blocks,
    enabling proper conditioning on the flow matching phase parameter.
    """
    
    def __init__(
        self,
        input_dim,
        output_dim=None,
        obs_dim=6,
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
            obs_dim: dimension of observations
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
        self.obs_dim = obs_dim
        
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
        
        # Observation embedding 
        # We assume obs is a flat vector [batch, obs_dim] that becomes 1 token [batch, 1, hidden]
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
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

        # Learnable NULL token for unconditional generation/masking
        self.null_token = nn.Parameter(torch.rand(1, 1, hidden_dim))
        
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
        
    def forward(self, x, obs, phase, cond_mask=None):
        """
        Forward pass through the flow matching transformer.
        
        Args:
            x: input tensor [batch, seq_len, input_dim]
            obs: observation tensor [batch, obs_dim]
            phase: flow matching phase tensor [batch] or [batch, 1]
            cond_mask: optional boolean mask [batch] indicating which samples are conditioned (if None, all are conditioned)
        
        Returns:
            output: predicted vector field [batch, seq_len, input_dim]
        """
        B, N, _ = x.shape
        
        # Embed phase
        if phase.dim() == 1: phase = phase.view(-1)
        elif phase.dim() == 2: phase = phase.squeeze(-1)
        phase_emb = self.phase_emb(phase)  # [batch, phase_dim]

        # Embed input
        x_emb = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Add positional embeddings to denoising tokens
        x_emb = x_emb + self.pos_emb[:, :N, :]

        # Embed observations (Handles [batch, M, obs_dim] -> [batch, M, hidden_dim])
        o_emb = self.obs_emb(obs)  # [batch, 1, hidden_dim]
        if cond_mask is not None:
            # cond_mask is a boolean [batch, M]. True means repalce with NULL token
            expanded_null = self.null_token.expand(B, o_emb.shape[1], -1)  # [batch, M, hidden_dim]
            o_emb = torch.where(cond_mask.unsqueeze(-1), expanded_null, o_emb)  # [batch, M, hidden_dim]

        h = x_emb
        # pass o_emb as context for cross-attention
        for block in self.blocks:
            h = block(h, phase_emb, context=o_emb)
        
        # Final normalization and projection
        h = self.final_norm(h, phase_emb)
        x = self.output_proj(h) # [batch, seq_len, output_dim]
        
        return x
    
    def cfm_loss(self, x_t, t, v_target, obs, cond_mask=None, reduction='mean'):
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
        v_pred = self.forward(x_t, obs, t, cond_mask=cond_mask)
        
        # Compute MSE loss
        loss = F.mse_loss(v_pred, v_target, reduction=reduction)
        
        return loss
    
    @torch.no_grad()
    def sample(self, x0, obs, num_steps=100, method='euler', manifold='se3'):
        """
        Generate samples using ODE integration.

        Args:
            x0: initial noise samples [batch, seq_len, input_dim]
            num_steps: number of integration steps
            method: integration method ('euler' or 'midpoint')
            manifold: 'se3' (default) or 'euclidean'.

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
            v = self.forward(x, obs, t)
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
                'obs_dim': self.obs_dim
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
    def inference(self, start_poses, obs=None, num_steps=100, return_trajectory=False, cfg_scale=3.0, manifold='se3'):
        """
        Generate goal states from start states using the trained flow model.

        Args:
            start_poses: starting states. For manifold='se3': poses [batch, 7] or
                [batch, seq_len, 7]. For manifold='euclidean': noise samples
                [batch, input_dim] or [batch, seq_len, input_dim].
            obs: observation tensor [batch, M, obs_dim]; if None, all tokens replaced with null
            num_steps: number of ODE integration steps
            return_trajectory: if True, return full trajectory; if False, only final state
            cfg_scale: classifier-free guidance scale (if >1.0, amplifies the predicted vector field for more aggressive generation)
            manifold: 'se3' (default) or 'euclidean'.

        Returns:
            If return_trajectory=False:
                final state [batch, D] or [batch, seq_len, D]
            If return_trajectory=True:
                trajectory: all intermediate states [batch, num_steps+1, D] or
                [batch, num_steps+1, seq_len, D]
        """
        self.eval()

        x = start_poses.unsqueeze(1) if start_poses.dim() == 2 else start_poses  # Ensure shape [batch, seq_len, 7]
        squeeze_output = start_poses.dim() == 2

        device = x.device
        B = x.shape[0]
        dt = torch.tensor(1.0 / num_steps, device=device)

        # When obs is None, use a dummy obs and replace all tokens with null (unconditional)
        if obs is None:
            obs = torch.zeros(B, 1, self.obs_dim, device=device)
            cond_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            uncond_mask = cond_mask  # both passes identical; CFG is a no-op
        else:
            uncond_mask = torch.ones(B, obs.shape[1], dtype=torch.bool, device=device)
            cond_mask = torch.zeros(B, obs.shape[1], dtype=torch.bool, device=device)

        if return_trajectory: trajectory = [x.clone()]

        for step in range(num_steps):
            t = torch.full((B,), step * dt.item(), device=device)

            if cfg_scale > 1.0 and not torch.all(cond_mask):
                # double forward pass for CFG (skipped when fully unconditional)
                x_double = torch.cat([x, x], dim=0)  # [2*batch, seq_len, 7]
                t_double = torch.cat([t, t], dim=0)  # [2*batch]
                obs_double = torch.cat([obs, obs], dim=0)  # [2*batch, obs_dim]
                mask_double = torch.cat([cond_mask, uncond_mask], dim=0)  # [2*batch, M]

                v_double = self.forward(x_double, obs_double, t_double, cond_mask=mask_double)  # [2*batch, seq_len, 7]
                v_cond, v_uncond = v_double.chunk(2, dim=0)  # Each [batch, seq_len, 7]
                v = v_uncond + cfg_scale * (v_cond - v_uncond)  # Amplify the difference
            else:
                v = self.forward(x, obs, t, cond_mask=cond_mask)  # [batch, seq_len, 7]
            
            x = _step_state(x, v, dt, manifold)  # Integrate ODE step

            if return_trajectory: trajectory.append(x.clone())
        
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
        



        