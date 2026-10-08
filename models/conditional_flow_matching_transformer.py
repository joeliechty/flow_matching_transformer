import torch
import torch.nn as nn

from models.flow_transformer_base import FlowTransformerBase

# How the condition tokens reach the transformer (an ablation axis):
#   adaln       flattened and added to the time embedding, so they drive every block's adaLN
#               modulation [DiT's class conditioning, Peebles & Xie 2023]
#   cross_attn  attended to by a cross-attention branch in every block
#   joint       appended to the sequence, so self-attention mixes them in [DiT's in-context
#               conditioning; SD3's MM-DiT and π0 also attend jointly, with separate weights]
COND_MODES = ('adaln', 'cross_attn', 'joint')


class ConditionalFlowMatchingTransformerModel(FlowTransformerBase):
    """
    Conditional Flow Matching Transformer with adaLN-Zero time conditioning.

    A DiT-style transformer (see `models.flow_transformer_base`) that predicts the velocity of
    the flow given condition tokens (observations or action tokens), injected per `cond_mode`.
    Any token can be replaced by a learned null token, for classifier-free guidance and
    partial conditioning.
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
        max_seq_len=1024,
        pos_emb_type='1d_learned',
        pos_emb_grid=None,
        manifold='se3',
        time_emb='sinusoidal',
        cond_mode='adaln',
        num_obs_tokens=1,
    ):
        """
        Initialize the Flow Matching Transformer model.
        Args:
            input_dim: dimension of the state (7 for an SE(3) pose)
            output_dim: dimension of the predicted velocity (default: input_dim)
            hidden_dim: dimension of transformer hidden states
            obs_dim: dimension of each condition token
            num_layers: number of transformer layers
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dim to model dim
            dropout: dropout rate
            phase_dim: dimension of phase embeddings
            max_seq_len: maximum sequence length for positional embeddings
            manifold: 'se3' (poses, integrated with the twist exponential map) or 'euclidean'
            time_emb: featurisation of the flow time, one of `models.support_models.TIME_EMBEDDINGS`
            cond_mode: how the condition tokens enter, one of `COND_MODES`
            num_obs_tokens: number of condition tokens M per sample
        """
        if cond_mode not in COND_MODES:
            raise ValueError(f"Unknown cond_mode: {cond_mode!r}. Expected one of {COND_MODES}.")
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, num_heads, mlp_ratio,
                         dropout, phase_dim, max_seq_len, pos_emb_type, pos_emb_grid, manifold,
                         time_emb, cross_attn=cond_mode == 'cross_attn')
        self.obs_dim = obs_dim
        self.cond_mode = cond_mode
        self.num_obs_tokens = num_obs_tokens

        # Observation embedding: [batch, M, obs_dim] -> [batch, M, hidden_dim]
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Learnable NULL token for unconditional generation/masking
        self.null_token = nn.Parameter(torch.rand(1, 1, hidden_dim))

        # Which slot each condition token fills, added after the null-token swap so the model
        # also knows which slots are nulled
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, num_obs_tokens, hidden_dim))

        if cond_mode == 'adaln':
            self.cond_proj = nn.Linear(num_obs_tokens * hidden_dim, phase_dim)

        self._init_weights()
        nn.init.normal_(self.obs_pos_emb, std=0.02)

    def model_config(self):
        return {**super().model_config(), 'obs_dim': self.obs_dim, 'cond_mode': self.cond_mode,
                'num_obs_tokens': self.num_obs_tokens}

    def _predict(self, x, obs, phase, cond_mask=None):
        """Normalised velocity [batch, seq_len, output_dim]."""
        B, N, _ = x.shape
        c = self._time(phase)  # [batch, phase_dim]
        h = self._embed(x)     # [batch, seq_len, hidden_dim]

        o = self.obs_emb(obs)  # [batch, M, hidden_dim]
        if cond_mask is not None:
            # cond_mask is a boolean [batch, M]. True means replace with NULL token
            expanded_null = self.null_token.expand(B, o.shape[1], -1)
            o = torch.where(cond_mask.unsqueeze(-1), expanded_null, o)
        o = o + self.obs_pos_emb[:, :o.shape[1]]

        context = None
        if self.cond_mode == 'adaln':
            c = c + self.cond_proj(o.flatten(1))
        elif self.cond_mode == 'cross_attn':
            context = o
        else:  # joint: the condition tokens join the sequence and are dropped at the output
            h = torch.cat([h, o], dim=1)
        h = self._trunk(h, c, context=context)
        return self.final_layer(h[:, :N], c)

    def forward(self, x, obs, phase, cond_mask=None):
        """
        Forward pass through the flow matching transformer.

        Args:
            x: input tensor [batch, seq_len, input_dim]
            obs: condition tokens [batch, M, obs_dim]
            phase: flow matching phase tensor [batch] or [batch, 1]
            cond_mask: optional boolean mask [batch, M]; True replaces that token with the null
                token (if None, every token is visible)

        Returns:
            output: predicted vector field [batch, seq_len, output_dim]
        """
        return self._denormalize(self._predict(x, obs, phase, cond_mask))

    def cfm_loss(self, x_t, t, v_target, obs, cond_mask=None, reduction='mean'):
        """
        Conditional flow matching loss: MSE between the predicted and target velocity at the
        path states x_t, in normalised units.

        Args:
            x_t: states on the paths [batch, seq_len, input_dim]
            t: their flow times [batch]
            v_target: target velocities [batch, seq_len, output_dim]
            obs: condition tokens [batch, M, obs_dim]
            cond_mask: optional boolean mask [batch, M] of tokens replaced by the null token
            reduction: 'mean', 'sum', or 'none'
        """
        return self._velocity_loss(self._predict(x_t, obs, t, cond_mask), v_target, reduction)

    @torch.no_grad()
    def sample(self, x0, obs, num_steps=100, method='euler', manifold=None):
        """Generate samples [batch, seq_len, input_dim] from x0 by unguided ODE integration."""
        return self.inference(x0, obs, num_steps=num_steps, cfg_scale=1.0, method=method,
                              manifold=manifold)

    @torch.no_grad()
    def inference(self, start_poses, obs=None, num_steps=100, return_trajectory=False, cfg_scale=3.0,
                  manifold=None, obs_mask=None, method='euler', cfg_interval=None):
        """
        Generate goal states from start states using the trained flow model.

        Args:
            start_poses: starting states. For manifold='se3': poses [batch, 7] or
                [batch, seq_len, 7]. For manifold='euclidean': noise samples
                [batch, input_dim] or [batch, seq_len, input_dim].
            obs: observation tensor [batch, M, obs_dim]; if None, all tokens replaced with null
            obs_mask: optional bool [batch, M]; True replaces that obs token with the null token,
                e.g. to condition on a subset of the tokens. Ignored when obs is None.
            num_steps: number of ODE integration steps
            return_trajectory: if True, return full trajectory; if False, only final state
            cfg_scale: classifier-free guidance scale (if >1.0, amplifies the predicted vector field for more aggressive generation)
            manifold: integrator manifold (default: the model's)
            method: ODE solver, 'euler', 'midpoint' or 'heun'
            cfg_interval: optional (lo, hi); guidance is applied only at flow times lo <= t < hi,
                and the plain conditional velocity elsewhere [Kynkäänniemi et al. 2024]

        Returns:
            If return_trajectory=False:
                final state [batch, D] or [batch, seq_len, D]
            If return_trajectory=True:
                trajectory: all intermediate states [batch, num_steps+1, D] or
                [batch, num_steps+1, seq_len, D]
        """
        self.eval()
        B, device = start_poses.shape[0], start_poses.device

        # When obs is None, use a dummy obs and replace all tokens with null (unconditional)
        if obs is None:
            obs = torch.zeros(B, self.num_obs_tokens, self.obs_dim, device=device)
            cond_mask = torch.ones(B, self.num_obs_tokens, dtype=torch.bool, device=device)
            uncond_mask = cond_mask  # both passes identical; CFG is a no-op
        else:
            uncond_mask = torch.ones(B, obs.shape[1], dtype=torch.bool, device=device)
            if obs_mask is None:
                cond_mask = torch.zeros(B, obs.shape[1], dtype=torch.bool, device=device)
            else:
                cond_mask = obs_mask.to(device=device, dtype=torch.bool)

        guide = cfg_scale > 1.0 and not torch.all(cond_mask)
        lo, hi = cfg_interval if cfg_interval is not None else (float('-inf'), float('inf'))

        def velocity(x, t):
            if guide and lo <= t[0].item() < hi:
                # double forward pass for CFG
                v_double = self.forward(torch.cat([x, x]), torch.cat([obs, obs]), torch.cat([t, t]),
                                        cond_mask=torch.cat([cond_mask, uncond_mask]))
                v_cond, v_uncond = v_double.chunk(2, dim=0)
                return v_uncond + cfg_scale * (v_cond - v_uncond)  # Amplify the difference
            return self.forward(x, obs, t, cond_mask=cond_mask)

        return self._integrate(start_poses, velocity, num_steps, method, manifold, return_trajectory)
