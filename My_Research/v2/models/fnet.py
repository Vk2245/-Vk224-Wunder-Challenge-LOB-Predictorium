"""
fnet.py — Fourier Transform Network for LOB
=============================================
Replaces self-attention entirely with FFT (Google Research, 2021).

Why this helps:
  - FFT decomposes signal into frequency components in O(n log n)
  - Low-frequency components → slow-moving trends → t1 signal
  - High-frequency components → immediate microstructure → t0 signal
  - ZERO learned attention weights → less overfitting on small signal
  - Pure dense ops → perfect for TPU

Architecture:
  Input → Proj → N x (FFT mixing + FFN) → Pool → Head

Reference: "FNet: Mixing Tokens with Fourier Transforms" (Google, 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


FNET_DEFAULTS = {
    "hidden_dim": 192,
    "n_layers": 4,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "epochs": 3,
    "patience": 5,
    "use_amp": True,
    "use_compile": False,  # FFT + compile can clash
}


class FourierMixing(nn.Module):
    """Replace attention with 2D FFT along sequence and feature dims."""
    def forward(self, x):
        # x: (B, T, C)
        # Apply FFT along sequence dim (dim=1) and feature dim (dim=2)
        # Take real part as the "mixed" representation
        return torch.fft.fft2(x.float()).real.to(x.dtype)


class FNetBlock(nn.Module):
    """Pre-LN FNet block: Fourier mixing + FFN."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fourier = FourierMixing()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.fourier(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FNetModel(nn.Module):
    """
    FNet for LOB prediction.
    Frequency decomposition naturally separates t0 (high-freq) from t1 (low-freq).
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 192,
        n_layers: int = 4,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FNetBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Use both last token (causal/t0) and global mean (context/t1)
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        return self.head(torch.cat([last, mean], dim=-1))
