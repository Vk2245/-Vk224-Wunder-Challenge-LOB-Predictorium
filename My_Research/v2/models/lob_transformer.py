"""
lob_transformer.py — Causal Transformer with Rotary Positional Embeddings
==========================================================================
Novel for LOB:
- Causal self-attention (strictly no future leakage)
- RoPE (Rotary Position Embeddings) — better than learned positional embeddings
  for financial time-series because they encode RELATIVE position, not absolute.
- Pre-LN (more stable gradient flow than post-LN)
- 4 layers, 4 heads, hidden=128

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


LOB_TRANSFORMER_DEFAULTS = {
    "hidden_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 5,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


def build_rope_cache(seq_len: int, dim: int, device: torch.device):
    """Pre-compute RoPE sin/cos caches."""
    half = dim // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)       # (seq_len, half)
    freqs = torch.cat([freqs, freqs], dim=-1)   # (seq_len, dim)
    cos = freqs.cos()[None, None, :, :]         # (1, 1, seq_len, dim)
    sin = freqs.sin()[None, None, :, :]
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to query/key tensors."""
    # x: (B, heads, T, head_dim)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with RoPE.
    Uses register_buffer for causal mask so it moves to GPU automatically.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Causal mask: upper-triangular = -inf
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale    # (B, heads, T, T)
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block (Attention + FFN)."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class LOBTransformerModel(nn.Module):
    """
    Causal Transformer for LOB prediction.
    Input → Linear proj → N x (CausalAttn + FFN) → Last token → Head
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # RoPE cache (computed once, reused)
        # We pre-compute this for sequence length 100 to avoid torch.compile / CUDA Graphs mutation bugs.
        cos, sin = build_rope_cache(100, self.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)

        # Slice just in case T < 100 (though typically T=100)
        cos = self.rope_cos[:, :, :T, :]
        sin = self.rope_sin[:, :, :T, :]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        out = x[:, -1, :]   # Last causal token
        return self.head(out)
