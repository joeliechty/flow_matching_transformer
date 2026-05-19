import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHA(nn.Module):
    """Multi-head attention supporting both self- and cross-attention."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, Nq, C = q.shape
        Nk = k.shape[1]
        q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)


class _EncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer (DETR / standard ACT style)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_attn = _MHA(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pos=None):
        q = k = x if pos is None else x + pos
        x = self.norm1(x + self.drop(self.self_attn(q, k, x)))
        x = self.norm2(x + self.drop(self.mlp(x)))
        return x


class _DecoderLayer(nn.Module):
    """Post-norm Transformer decoder layer with self- and cross-attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.self_attn = _MHA(dim, num_heads, dropout)
        self.cross_attn = _MHA(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos=None, memory_pos=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt = self.norm1(tgt + self.drop(self.self_attn(q, k, tgt)))

        q = tgt if query_pos is None else tgt + query_pos
        k = memory if memory_pos is None else memory + memory_pos
        tgt = self.norm2(tgt + self.drop(self.cross_attn(q, k, memory)))

        tgt = self.norm3(tgt + self.drop(self.mlp(tgt)))
        return tgt


class ActionChunkingTransformerModel(nn.Module):
    """
    Action Chunking Transformer (ACT) with CVAE, following Zhao et al. (ALOHA, 2023).

    Architecture:
      - CVAE encoder (BERT-style): consumes [CLS, qpos, a_1..a_K] and produces (mu, logvar).
        Used at training time only; at inference, z is set to the prior mean (zeros).
      - Transformer encoder: consumes [z, obs] tokens and produces memory.
      - Transformer decoder: K learnable position queries attend to memory and predict
        the action chunk (K actions).

    Loss = L1(pred_actions, true_actions) + kl_weight * KL(q(z|obs, a) || N(0, I)).
    """

    def __init__(
        self,
        action_dim,
        obs_dim,
        chunk_size=100,
        hidden_dim=512,
        num_encoder_layers=4,
        num_decoder_layers=7,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        latent_dim=32,
        kl_weight=10.0,
    ):
        """
        Args:
            action_dim: dimension of a single action.
            obs_dim: dimension of the observation vector (qpos / proprioception).
            chunk_size: number of actions K predicted per forward pass.
            hidden_dim: transformer model dimension.
            num_encoder_layers: encoder depth for both the CVAE encoder and the main encoder.
            num_decoder_layers: decoder depth.
            num_heads: number of attention heads.
            mlp_ratio: feed-forward expansion ratio.
            dropout: dropout rate.
            latent_dim: dimension of the CVAE latent z.
            kl_weight: weight on the KL divergence term in the loss.
        """
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # ---------------- CVAE encoder (training only) ----------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cvae_qpos_proj = nn.Linear(obs_dim, hidden_dim)
        self.cvae_action_proj = nn.Linear(action_dim, hidden_dim)
        # Learned positional embedding for [CLS, qpos, a_1..a_K]
        self.cvae_pos_emb = nn.Parameter(torch.zeros(1, 2 + chunk_size, hidden_dim))
        self.cvae_encoder = nn.ModuleList([
            _EncoderLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)  # -> (mu, logvar)

        # ---------------- Main transformer encoder ----------------
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        # Two input tokens: [z, obs]. Learned positional embeddings.
        self.encoder_pos_emb = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.encoder = nn.ModuleList([
            _EncoderLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_encoder_layers)
        ])

        # ---------------- Transformer decoder ----------------
        # K learnable query position embeddings (one per action in the chunk).
        self.query_embed = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        self.decoder = nn.ModuleList([
            _DecoderLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cvae_pos_emb, std=0.02)
        nn.init.normal_(self.encoder_pos_emb, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)

    # ------------------------------------------------------------------
    # CVAE encoder: q(z | obs, actions)
    # ------------------------------------------------------------------
    def encode(self, obs, actions):
        """
        Args:
            obs: [batch, obs_dim]
            actions: [batch, chunk_size, action_dim]
        Returns:
            mu, logvar: each [batch, latent_dim]
        """
        B = obs.shape[0]
        cls = self.cls_token.expand(B, -1, -1)                          # [B, 1, H]
        q_tok = self.cvae_qpos_proj(obs).unsqueeze(1)                    # [B, 1, H]
        a_tok = self.cvae_action_proj(actions)                           # [B, K, H]
        tokens = torch.cat([cls, q_tok, a_tok], dim=1)                   # [B, 2+K, H]
        tokens = tokens + self.cvae_pos_emb[:, : tokens.shape[1], :]

        for layer in self.cvae_encoder:
            tokens = layer(tokens)

        cls_out = tokens[:, 0]                                           # [B, H]
        mu, logvar = self.latent_proj(cls_out).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Main encoder + decoder
    # ------------------------------------------------------------------
    def decode(self, obs, z):
        """
        Args:
            obs: [batch, obs_dim]
            z:   [batch, latent_dim]
        Returns:
            actions: [batch, chunk_size, action_dim]
        """
        B = obs.shape[0]
        z_tok = self.z_proj(z).unsqueeze(1)                              # [B, 1, H]
        o_tok = self.obs_proj(obs).unsqueeze(1)                          # [B, 1, H]
        memory = torch.cat([z_tok, o_tok], dim=1)                        # [B, 2, H]
        memory_pos = self.encoder_pos_emb                                # [1, 2, H]

        for layer in self.encoder:
            memory = layer(memory, pos=memory_pos)

        # Decoder: queries start at zero, positional embeddings carry identity.
        tgt = torch.zeros(B, self.chunk_size, self.hidden_dim, device=obs.device)
        query_pos = self.query_embed                                     # [1, K, H]
        for layer in self.decoder:
            tgt = layer(tgt, memory, query_pos=query_pos, memory_pos=memory_pos)

        tgt = self.decoder_norm(tgt)
        return self.action_head(tgt)

    def forward(self, obs, actions=None):
        """
        Args:
            obs: [batch, obs_dim]
            actions: optional [batch, chunk_size, action_dim]. If provided, the CVAE
                encoder is used and z is sampled via reparameterization; otherwise z=0.
        Returns:
            pred_actions: [batch, chunk_size, action_dim]
            mu, logvar: each [batch, latent_dim] (or None if actions is None)
        """
        if actions is not None:
            mu, logvar = self.encode(obs, actions)
            z = self.reparameterize(mu, logvar)
        else:
            mu = logvar = None
            z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)

        pred = self.decode(obs, z)
        return pred, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def act_loss(self, obs, actions, reduction='mean'):
        """
        Standard ACT loss: L1 reconstruction + kl_weight * KL(q(z|x,a) || N(0,I)).

        Args:
            obs: [batch, obs_dim]
            actions: [batch, chunk_size, action_dim]
        Returns:
            total_loss, dict with 'recon', 'kl', 'total'
        """
        pred, mu, logvar = self.forward(obs, actions)
        recon = F.l1_loss(pred, actions, reduction=reduction)
        # KL(N(mu, sigma^2) || N(0, I)) per-sample, then mean over batch.
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl = kl.mean() if reduction == 'mean' else kl.sum() if reduction == 'sum' else kl
        total = recon + self.kl_weight * kl
        return total, {'recon': recon.detach(), 'kl': kl.detach(), 'total': total.detach()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, obs):
        """
        Generate an action chunk from an observation, using z = 0 (prior mean).

        Args:
            obs: [batch, obs_dim] or [obs_dim]
        Returns:
            actions: [batch, chunk_size, action_dim] (batch dim preserved as given)
        """
        self.eval()
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device)
        pred = self.decode(obs, z)
        if squeeze:
            pred = pred.squeeze(0)
        return pred

    # ------------------------------------------------------------------
    # Checkpoint helpers (match conventions in flow_matching_transformer.py)
    # ------------------------------------------------------------------
    def save_checkpoint(self, filepath, optimizer=None, epoch=None, loss=None, **extra_info):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'action_dim': self.action_dim,
                'obs_dim': self.obs_dim,
                'chunk_size': self.chunk_size,
                'hidden_dim': self.hidden_dim,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'num_heads': self.num_heads,
                'mlp_ratio': self.mlp_ratio,
                'dropout': self.dropout,
                'latent_dim': self.latent_dim,
                'kl_weight': self.kl_weight,
            },
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
    def load_checkpoint(cls, filepath, device='cpu', optimizer=None, model_config=None):
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, checkpoint
