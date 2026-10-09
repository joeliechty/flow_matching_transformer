import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def modulate(x, shift, scale):
    """adaLN modulation of a normalised x [batch, seq_len, dim] by per-sample shift and scale [batch, dim]."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    """Multi-head self-attention with QK-norm: queries and keys are RMS-normalised per head
    before the dot product, which bounds the attention logits [SD3, Esser et al. 2024]."""
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V: each [B, heads, N, head_dim]
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        x = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        
        # Combine heads
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(x))


class CrossAttention(nn.Module):
    """Cross-attention from the sequence to the condition tokens, with QK-norm."""
    def __init__(self, dim, context_dim=None, num_heads=8, dropout=0.0):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        """context: [batch, M, context_dim], or a (contexts [U, M, context_dim], index [batch])
        pair when many samples share a few contexts (`index` picks each sample's)."""
        if isinstance(context, tuple):
            return self._shared_context(x, *context)
        B, N, C = x.shape
        B_c, M, C_c = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # K and V come from the context (the condition tokens)
        k, v = self.kv(context).reshape(B_c, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        out = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v,
                                             dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(out))

    def _shared_context(self, x, contexts, index):
        """Attention to contexts[index[b]] with the keys and values computed once per context:
        scores and outputs are taken against every context and the sample's one picked out,
        which is cheap for few contexts and keeps no per-sample keys or values."""
        B, N, C = x.shape
        U, M, _ = contexts.shape
        h, d = self.num_heads, self.head_dim
        q = self.q_norm(self.q(x).reshape(B, N, h, d).transpose(1, 2))         # [B, h, N, d]
        k, v = self.kv(contexts).reshape(U, M, 2, h, d).permute(2, 0, 3, 1, 4)  # [U, h, M, d]
        pick = index.view(B, 1, 1, 1, 1)
        scores = torch.einsum('bhnd,uhmd->bhnum', q, self.k_norm(k)) * d ** -0.5
        attn = scores.gather(3, pick.expand(B, h, N, 1, M)).squeeze(3).softmax(-1)
        if self.training and self.attn_dropout > 0:
            attn = F.dropout(attn, self.attn_dropout)
        out = torch.einsum('bhnm,uhmd->bhnud', attn, v).gather(3, pick.expand(B, h, N, 1, d))
        out = out.squeeze(3).transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(out))
    

class TransformerBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning [DiT, Peebles & Xie 2023].

    Each branch (self-attention, optional cross-attention to condition tokens, MLP) sees an
    RMS-normalised input modulated by a per-sample shift and scale, and its output is scaled by
    a per-sample gate before the residual add. Shift, scale and gate are regressed from the
    conditioning vector c by a zero-initialised layer (`zero_init`), so the block starts as
    the identity.
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=256, cross_attn=False):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout) if cross_attn else None
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.num_branches = 3 if cross_attn else 2
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * self.num_branches * dim))

    def zero_init(self):
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        
    def forward(self, x, c, context=None):
        """
        Args:
            x: input tensor [batch, seq_len, dim]
            c: conditioning vector [batch, cond_dim] (time embedding, plus the condition in adaLN mode)
            context: condition tokens [batch, M, dim] for the cross-attention branch
        """
        mods = self.modulation(c).chunk(3 * self.num_branches, dim=-1)
        branches = [self.attn]
        if self.cross_attn is not None:
            branches.append(lambda h: self.cross_attn(h, context))
        branches.append(self.mlp)
        for i, branch in enumerate(branches):
            shift, scale, gate = mods[3 * i:3 * i + 3]
            x = x + gate.unsqueeze(1) * branch(modulate(self.norm(x), shift, scale))
        return x


class FinalLayer(nn.Module):
    """adaLN-modulated RMSNorm and the output projection, both zero-initialised (`zero_init`),
    so the model's raw output starts at zero."""

    def __init__(self, dim, cond_dim, output_dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        self.proj = nn.Linear(dim, output_dim)

    def zero_init(self):
        for layer in (self.modulation[-1], self.proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


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


class GaussianFourierProjection(nn.Module):
    """Random Fourier features of t: sin and cos of 2π·t·w with fixed w ~ N(0, scale²)
    [Tancik et al. 2020; Song et al. 2021]. w is a buffer, so it is saved with the model."""

    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer('W', torch.randn(dim // 2) * scale)

    def forward(self, t):
        proj = 2 * math.pi * t[:, None] * self.W[None, :]
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


# Featurisations of the flow time t in [0, 1] (an ablation axis):
#   sinusoidal        sinusoids of t itself; the highest frequency is 1 rad per unit of t
#   sinusoidal_x1000  sinusoids of 1000·t, the scale DiT trains at and SD3 / Flux use
#   fourier           Gaussian Fourier features, scale 16
TIME_EMBEDDINGS = ('sinusoidal', 'sinusoidal_x1000', 'fourier')


class TimeEmbedding(nn.Module):
    """Flow time t [batch] -> [batch, dim]: a fixed featurisation (`TIME_EMBEDDINGS`), then a
    two-layer MLP."""

    def __init__(self, dim, kind='sinusoidal'):
        super().__init__()
        if kind not in TIME_EMBEDDINGS:
            raise ValueError(f"Unknown time embedding: {kind!r}. Expected one of {TIME_EMBEDDINGS}.")
        self.scale = 1000.0 if kind == 'sinusoidal_x1000' else 1.0
        self.features = GaussianFourierProjection(dim) if kind == 'fourier' else SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.mlp(self.features(t * self.scale))


def _sincos_1d(embed_dim, positions):
    """1D sin/cos embedding from a 1D float tensor of positions -> [N, embed_dim]."""
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / float(half)
    omega = 1.0 / (10000.0 ** omega)                       # [half]
    out = positions.float().reshape(-1, 1) * omega.reshape(1, -1)  # [N, half]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)      # [N, embed_dim]


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """Fixed 2D sin/cos positional embedding -> [1, grid_h*grid_w, embed_dim].

    Half of the channels encode the row index, the other half the column index.
    Row-major ordering matches the patch_encode flatten in image_gen_trainer.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    # row-major: outer = row, inner = col
    row_idx = rows.repeat_interleave(grid_w)   # [H*W]
    col_idx = cols.repeat(grid_h)              # [H*W]

    emb_h = _sincos_1d(embed_dim // 2, row_idx)   # [H*W, embed_dim/2]
    emb_w = _sincos_1d(embed_dim // 2, col_idx)   # [H*W, embed_dim/2]
    emb = torch.cat([emb_h, emb_w], dim=1)        # [H*W, embed_dim]
    return emb.unsqueeze(0)                        # [1, H*W, embed_dim]


def farthest_point_sample(points, m):
    """Indices [batch, m] of m farthest-point-sampled points of points [batch, P, 3], starting
    from point 0 (the points are random surface samples, so the start needs no randomness)."""
    B, P, _ = points.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=points.device)
    dist = torch.full((B, P), float('inf'), device=points.device)
    rows = torch.arange(B, device=points.device)
    farthest = torch.zeros(B, dtype=torch.long, device=points.device)
    for i in range(m):
        idx[:, i] = farthest
        d = ((points - points[rows, farthest].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


class PlainBlock(nn.Module):
    """Pre-norm transformer block (RMSNorm, QK-normed self-attention, MLP), without adaLN."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PointPatchEncoder(nn.Module):
    """Point cloud [batch, P, 3] -> patch tokens [batch, num_patches, dim], the Point-BERT /
    Point-MAE tokeniser [Yu et al. 2022; Pang et al. 2022]: farthest point sampling picks the
    patch centres, each centre's patch_size nearest points (relative to the centre) go through a
    shared mini-PointNet and are max-pooled, an MLP of the centre adds its position, and a few
    plain transformer blocks mix the patches.

    A batch usually holds several samples of one object, so identical clouds are encoded once.
    """

    def __init__(self, dim, num_patches=64, patch_size=32, num_layers=4, num_heads=4):
        super().__init__()
        self.num_patches, self.patch_size = num_patches, patch_size
        self.point_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 256))
        self.patch_proj = nn.Sequential(nn.Linear(256, dim), nn.GELU(), nn.Linear(dim, dim))
        self.centre_emb = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, dim))
        self.blocks = nn.ModuleList([PlainBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)

    def _encode(self, points):
        B = points.shape[0]
        rows = torch.arange(B, device=points.device).unsqueeze(1)
        centres = points[rows, farthest_point_sample(points, self.num_patches)]   # [B, M, 3]
        d = ((centres.unsqueeze(2) - points.unsqueeze(1)) ** 2).sum(-1)          # [B, M, P]
        knn = d.topk(self.patch_size, dim=-1, largest=False).indices             # [B, M, k]
        patches = points[rows.unsqueeze(-1), knn] - centres.unsqueeze(2)         # [B, M, k, 3]
        h = self.patch_proj(self.point_mlp(patches).amax(dim=2)) + self.centre_emb(centres)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, points):
        """Tokens of each cloud [batch, num_patches, dim]."""
        tokens, index = self.encode_unique(points)
        return tokens[index]

    def encode_unique(self, points):
        """Tokens of each distinct cloud [U, num_patches, dim] and each sample's index into them
        [batch]."""
        # Group the clouds by their first few points (distinct random surface samplings never
        # share them), check the grouping on the whole clouds, and encode one of each group
        _, inverse = torch.unique(points[:, :4].flatten(1), dim=0, return_inverse=True)
        first = torch.empty(int(inverse.max()) + 1, dtype=torch.long, device=points.device)
        first.scatter_(0, inverse, torch.arange(len(points), device=points.device))
        if not torch.equal(points[first][inverse], points):
            _, inverse = torch.unique(points.flatten(1), dim=0, return_inverse=True)
            first = torch.empty(int(inverse.max()) + 1, dtype=torch.long, device=points.device)
            first.scatter_(0, inverse, torch.arange(len(points), device=points.device))
        return self._encode(points[first]), inverse
