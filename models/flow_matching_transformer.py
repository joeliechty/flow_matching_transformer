import torch

from models.flow_transformer_base import FlowTransformerBase


class FlowMatchingTransformerModel(FlowTransformerBase):
    """
    Flow Matching Transformer with adaLN-Zero time conditioning.

    A DiT-style transformer (see `models.flow_transformer_base`) that predicts the velocity of
    the flow at state x and flow time (phase) t.
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
        manifold='se3',
        time_emb='sinusoidal',
    ):
        """
        Initialize the Flow Matching Transformer model.
        Args:
            input_dim: dimension of the state (7 for an SE(3) pose)
            output_dim: dimension of the predicted velocity (default: input_dim)
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
            manifold: 'se3' (poses, integrated with the twist exponential map) or 'euclidean'
            time_emb: featurisation of the flow time, one of `models.support_models.TIME_EMBEDDINGS`
        """
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, num_heads, mlp_ratio,
                         dropout, phase_dim, max_seq_len, pos_emb_type, pos_emb_grid, manifold,
                         time_emb)
        self._init_weights()

    def _predict(self, x, phase):
        """Normalised velocity [batch, seq_len, output_dim]."""
        with self._autocast(x):
            c = self._time(phase)
            h = self._trunk(self._embed(x), c)
            out = self.final_layer(h, c)
        return out.float()

    def forward(self, x, phase):
        """
        Forward pass through the flow matching transformer.

        Args:
            x: input tensor [batch, seq_len, input_dim]
            phase: flow matching phase parameter [batch] or [batch, 1], in [0, 1]

        Returns:
            output: predicted vector field [batch, seq_len, output_dim]
        """
        return self._denormalize(self._predict(x, phase))

    def cfm_loss(self, x_t, t, v_target, reduction='mean'):
        """
        Conditional flow matching loss: MSE between the predicted and target velocity at the
        path states x_t, in normalised units.

        Args:
            x_t: states on the paths [batch, seq_len, input_dim]
            t: their flow times [batch]
            v_target: target velocities [batch, seq_len, output_dim]
            reduction: 'mean', 'sum', or 'none'
        """
        return self._velocity_loss(self._predict(x_t, t), v_target, reduction)

    @torch.no_grad()
    def sample(self, x0, num_steps=100, method='euler', manifold=None):
        """Generate samples [batch, seq_len, input_dim] from x0 by ODE integration."""
        return self.inference(x0, num_steps=num_steps, method=method, manifold=manifold)

    @torch.no_grad()
    def inference(self, start_poses, num_steps=100, return_trajectory=False, manifold=None,
                  method='euler'):
        """
        Generate goal states from start states using the trained flow model.

        Args:
            start_poses: starting states. For manifold='se3': poses in quaternion
                format [batch, 7] or [batch, seq_len, 7]. For manifold='euclidean':
                noise samples [batch, input_dim] or [batch, seq_len, input_dim].
            num_steps: number of ODE integration steps
            return_trajectory: if True, return full trajectory; if False, only final state
            manifold: integrator manifold (default: the model's)
            method: ODE solver, 'euler', 'midpoint' or 'heun'

        Returns:
            If return_trajectory=False:
                final state [batch, D] or [batch, seq_len, D]
            If return_trajectory=True:
                trajectory: all intermediate states [batch, num_steps+1, D] or
                [batch, num_steps+1, seq_len, D]
        """
        self.eval()
        return self._integrate(start_poses, self.forward, num_steps, method, manifold,
                               return_trajectory)
