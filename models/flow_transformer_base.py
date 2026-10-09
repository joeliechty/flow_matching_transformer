"""Shared core of the flow-matching transformers (`FlowMatchingTransformerModel` and
`ConditionalFlowMatchingTransformerModel`).

The network is a DiT-style transformer [Peebles & Xie 2023]: adaLN-Zero blocks with RMSNorm,
QK-normed attention through `F.scaled_dot_product_attention`, and a zero-initialised output layer.
On SE(3) it sees each pose as [standardised position, 6-D rotation] and predicts a standardised
twist; `forward` returns the twist in the task's units. Sampling integrates the predicted
velocity with Euler, midpoint or Heun steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tf_utils import add_twist_to_pose, _quat_to_ortho6d
from utils.euclid_utils import add_velocity_to_state
from models.support_models import FinalLayer, TimeEmbedding, TransformerBlock, get_2d_sincos_pos_embed

MANIFOLDS = ('se3', 'euclidean')

# ODE solvers for sampling, and the network evaluations each takes per step (CFG doubles them).
SOLVER_EVALS = {'euler': 1, 'midpoint': 2, 'heun': 2}


def _step_state(x, v, dt, manifold):
    if manifold == 'se3':
        return add_twist_to_pose(x, v, dt)
    elif manifold == 'euclidean':
        return add_velocity_to_state(x, v, dt)
    else:
        raise ValueError(f"Unknown manifold: {manifold!r}. Expected 'se3' or 'euclidean'.")


def ode_step(x, t, dt, velocity, method, manifold):
    """One step of `method` from state x at times t [batch]; velocity(x, t) -> v.

    On SE(3) every velocity is a twist applied from x through the exponential map, so midpoint
    and Heun are Runge–Kutta–Munthe-Kaas steps on R³×SO(3) with dexp⁻¹ taken as the identity,
    which keeps them second order.
    """
    v = velocity(x, t)
    if method == 'euler':
        return _step_state(x, v, dt, manifold)
    if method == 'midpoint':
        x_mid = _step_state(x, v, dt / 2, manifold)
        return _step_state(x, velocity(x_mid, t + dt / 2), dt, manifold)
    if method == 'heun':
        v_end = velocity(_step_state(x, v, dt, manifold), t + dt)
        return _step_state(x, (v + v_end) / 2, dt, manifold)
    raise ValueError(f"Unknown solver: {method!r}. Expected one of {tuple(SOLVER_EVALS)}.")


class FlowTransformerBase(nn.Module):
    """Embedding, transformer trunk, normalisation, checkpointing and ODE integration shared by
    the unconditional and conditional models. Subclasses build their condition pathway, call
    `_init_weights`, and implement `_predict` (the normalised velocity)."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, mlp_ratio,
                 dropout, phase_dim, max_seq_len, pos_emb_type, pos_emb_grid, manifold,
                 time_emb, cross_attn=False):
        super().__init__()
        if manifold not in MANIFOLDS:
            raise ValueError(f"Unknown manifold: {manifold!r}. Expected one of {MANIFOLDS}.")
        if manifold == 'se3' and input_dim != 7:
            raise ValueError("an SE(3) model's state is a 7-D pose (x, y, z, qw, qx, qy, qz)")
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.phase_dim = phase_dim
        self.max_seq_len = max_seq_len
        self.pos_emb_type = pos_emb_type
        self.pos_emb_grid = tuple(pos_emb_grid) if pos_emb_grid is not None else None
        self.manifold = manifold
        self.time_emb = time_emb

        # Input projection. An SE(3) pose enters as [standardised position, 6-D rotation]: the
        # 6-D rotation is continuous and the same for q and -q, unlike the quaternion [Zhou et al. 2019].
        self.input_proj = nn.Linear(9 if manifold == 'se3' else input_dim, hidden_dim)

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

        # Phase (flow time) embedding, which drives every block's adaLN modulation
        self.phase_emb = TimeEmbedding(phase_dim, time_emb)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                             cond_dim=phase_dim, cross_attn=cross_attn)
            for _ in range(num_layers)
        ])
        self.final_layer = FinalLayer(hidden_dim, phase_dim, self.output_dim)

        # The network sees standardised positions and predicts standardised velocities. Identity
        # until `set_normalizer`; the pose trainer sets it from the task's statistics.
        self.register_buffer('pos_mean', torch.zeros(3))
        self.register_buffer('pos_std', torch.ones(3))
        self.register_buffer('vel_mean', torch.zeros(self.output_dim))
        self.register_buffer('vel_std', torch.ones(self.output_dim))

    def _init_weights(self):
        """Xavier for every linear layer, then adaLN-Zero: every block starts as the identity and
        the raw output at zero, so the initial velocity is the mean velocity."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        if self.pos_emb_type == '1d_learned':
            nn.init.normal_(self.pos_emb, std=0.02)
        for block in self.blocks:
            block.zero_init()
        self.final_layer.zero_init()

    @torch.no_grad()
    def set_normalizer(self, pos_mean, pos_std, vel_mean, vel_std):
        """Position and velocity statistics (see `utils.train_utils.pose_normalizer_stats`)."""
        for name, value in (('pos_mean', pos_mean), ('pos_std', pos_std),
                            ('vel_mean', vel_mean), ('vel_std', vel_std)):
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def _embed(self, x):
        """State [batch, seq_len, input_dim] -> tokens [batch, seq_len, hidden_dim]."""
        if self.manifold == 'se3':
            pos = (x[..., :3] - self.pos_mean) / self.pos_std
            x = torch.cat([pos, _quat_to_ortho6d(x[..., 3:7].contiguous())], dim=-1)
        h = self.input_proj(x)
        return h + self.pos_emb[:, :h.shape[1], :]

    @staticmethod
    def _autocast(ref):
        """bf16 autocast for the network on CUDA [mixed precision, as in DiT/SD3 training];
        the pose maths around it stays in fp32."""
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=ref.is_cuda)

    def _time(self, phase):
        """Flow time [batch] or [batch, 1] -> [batch, phase_dim]."""
        if phase.dim() == 2:
            phase = phase.squeeze(-1)
        return self.phase_emb(phase)

    def _trunk(self, h, c, context=None):
        for block in self.blocks:
            h = block(h, c, context=context)
        return h

    def _denormalize(self, out):
        return self.vel_mean + self.vel_std * out

    def _velocity_loss(self, pred, v_target, reduction='mean'):
        """MSE between a normalised prediction and the target velocity, in normalised units."""
        return F.mse_loss(pred, (v_target - self.vel_mean) / self.vel_std, reduction=reduction)

    def model_config(self):
        """Constructor arguments, saved with every checkpoint."""
        return {
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
            'manifold': self.manifold,
            'time_emb': self.time_emb,
        }

    def save_checkpoint(self, filepath, optimizer=None, epoch=None, loss=None, **extra_info):
        """
        Save model checkpoint.

        Args:
            filepath: path to save checkpoint
            optimizer: optional optimizer state to save
            epoch: optional epoch number
            loss: optional loss value
            **extra_info: any additional information to save, e.g. ema_state_dict
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        checkpoint.update(extra_info)
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(cls, filepath, device='cpu', optimizer=None, model_config=None, use_ema=True):
        """
        Load model from checkpoint.

        Args:
            filepath: path to checkpoint file
            device: device to load model on
            optimizer: optional optimizer to load state into
            model_config: model config dict, used only if the checkpoint doesn't contain one
            use_ema: load the EMA weights when the checkpoint has them (the weights to sample from)

        Returns:
            model: loaded model
            checkpoint: full checkpoint dict with epoch, loss, etc.
        """
        checkpoint = torch.load(filepath, map_location=device)

        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        elif model_config is not None:
            config = model_config
        else:
            raise ValueError(
                "Checkpoint does not contain 'model_config'. "
                "Please provide model_config parameter to load_checkpoint() with the model configuration."
            )

        model = cls(**config)
        ema = checkpoint.get('ema_state_dict') if use_ema else None
        model.load_state_dict(ema if ema is not None else checkpoint['model_state_dict'])
        model.to(device)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, checkpoint

    @torch.no_grad()
    def _integrate(self, x0, velocity, num_steps, method, manifold, return_trajectory):
        """Integrate dx/dt = velocity(x, t) from t = 0 to 1 in `num_steps` steps of `method`.

        x0 is [batch, D] or [batch, seq_len, D]. Returns the final state, or with
        `return_trajectory` every step's state [batch, num_steps + 1, (seq_len,) D]; a 2-D x0
        gets its seq_len dimension squeezed back out.
        """
        if method not in SOLVER_EVALS:
            raise ValueError(f"Unknown solver: {method!r}. Expected one of {tuple(SOLVER_EVALS)}.")
        manifold = manifold or self.manifold
        squeeze_output = x0.dim() == 2
        x = x0.unsqueeze(1) if squeeze_output else x0
        B = x.shape[0]
        dt = torch.tensor(1.0 / num_steps, device=x.device)

        trajectory = [x.clone()] if return_trajectory else None
        for step in range(num_steps):
            t = torch.full((B,), step * dt.item(), device=x.device)
            x = ode_step(x, t, dt, velocity, method, manifold)
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=1)  # [batch, num_steps+1, seq_len, D]
            return trajectory.squeeze(2) if squeeze_output else trajectory
        return x.squeeze(1) if squeeze_output else x
