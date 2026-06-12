"""
patch_tst.py — Patch-based Transformer for LOB
=================================================
Inspired by PatchTST (Nie et al., 2023) and Google TimesFM.

Key insight: Instead of attending to 100 individual timesteps,
group them into 10 PATCHES of 10. Each patch captures local
LOB microstructure as a single token. Attention between patches
captures inter-pattern relationships at the right timescale for t1.

Benefits:
  - 10x shorter sequence → attention is 100x cheaper → TPU-friendly
  - Patches capture local dynamics (5-10 tick patterns)
  - Patch-level attention captures regime transitions → t1
  - Patch embedding is a 1D convolution → dense TPU op

Architecture:
  Input → PatchEmbed(stride=10) → Transformer(10 tokens) → Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS, SEQ_LEN


PATCH_TST_DEFAULTS = {
    "hidden_dim": 192,
    "patch_size": 10,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "epochs": 3,
    "patience": 5,
    "use_amp": True,
    "use_compile": False,
}


class PatchEmbedding(nn.Module):
    """Convert (B, T, C) → (B, num_patches, hidden_dim) via strided 1D conv."""
    def __init__(self, input_dim: int, hidden_dim: int, patch_size: int = 10):
        super().__init__()
        self.patch_size = patch_size
        # Conv1d with kernel_size=stride=patch_size → non-overlapping patches
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, C) → transpose → (B, C, T)
        x = x.transpose(1, 2)
        x = self.proj(x)        # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, hidden_dim)
        return self.norm(x)


class PatchTSTModel(nn.Module):
    """
    Patch-based Transformer for LOB prediction.
    Groups 100 timesteps into 10 patches → lightweight attention.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 192,
        patch_size: int = 10,
        n_heads: int = 4,
        n_layers: int = 3,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = SEQ_LEN // patch_size  # 100 // 10 = 10

        self.patch_embed = PatchEmbedding(input_dim, hidden_dim, patch_size)

        # Learned positional embeddings for patches
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

        # Standard Transformer encoder on patch tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        # Head: last patch (most recent) + global mean
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.patch_embed(x)          # (B, 10, hidden_dim)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.transformer(x)          # (B, 10, hidden_dim)
        x = self.norm(x)
        last = x[:, -1, :]               # Most recent patch → t0
        mean = x.mean(dim=1)             # Global context → t1
        return self.head(torch.cat([last, mean], dim=-1))
