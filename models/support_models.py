import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class CrossAttention(nn.Module):
    """Cross-attention mechanism for conditioning."""
    def __init__(self, dim, context_dim=None, num_heads=8, dropout=0.0):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        B, N, C = x.shape
        B_c, M, C_c = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # K and V come from the context (the condition tokens)
        kv = self.kv(context).reshape(B_c, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.dropout(out)
    

class TransformerBlock(nn.Module):
    """Transformer block with Self-Attention, Cross-Attention, and AdaLN."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, phase_dim=256):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(dim, phase_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = AdaptiveLayerNorm(dim, phase_dim)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)
        self.norm3 = AdaptiveLayerNorm(dim, phase_dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x, phase_emb, context=None):
        """
        Args:
            x: input tensor [batch, seq_len, dim]
            phase_emb: phase embedding [batch, phase_dim]
        """
        # Self-attention block with residual
        x = x + self.attn(self.norm1(x, phase_emb))
        # Cross-attention to inject conditions
        if context is not None:
            x = x + self.cross_attn(self.norm2(x, phase_emb), context)
        # MLP block with residual
        x = x + self.mlp(self.norm3(x, phase_emb))
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
