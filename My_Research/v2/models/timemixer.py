"""
timemixer.py — TimeMixer: Multiscale MLP-based Time Series Model
================================================================
Ported from _archive_phase1/src/train_timemixer.py into v2 harness.
Based on TSMixer/TimeMixer (Google 2023, ICLR 2024).

Key innovation for LOB:
- Decomposes 100-step sequence into 3 scales: 100, 50, 25
- Separates seasonal/trend at each scale via moving average
- Pure MLP mixing (no RNN, no attention) → ultra-fast on GPU
- Fuses multi-scale features for final prediction

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


TIMEMIXER_DEFAULTS = {
    "hidden_dim": 256,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 6,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class MovingAvg(nn.Module):
    """Moving average for trend extraction. Pads to preserve sequence length."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].expand(-1, pad_left, -1)
        end   = x[:, -1:, :].expand(-1, pad_right, -1)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.transpose(1, 2)).transpose(1, 2)[:, :x.size(1), :]


class SeriesDecomp(nn.Module):
    """Decompose into (seasonal, trend) via moving average."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend  # seasonal, trend


class MixingBlock(nn.Module):
    """MLP mixing along the time dimension."""
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, C)
        x_norm = self.norm(x)
        mixed = self.mlp(x_norm.transpose(1, 2)).transpose(1, 2)
        return x + mixed


class TimeMixerModel(nn.Module):
    """
    TimeMixer for LOB prediction.

    Multiscale decomposition at 3 resolutions (100, 50, 25 steps).
    Seasonal/trend separation at each scale.
    MLP mixing at each scale, then concat-fuse → prediction head.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 256,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Decomposition at each scale
        self.decomp_100 = SeriesDecomp(kernel_size=25)
        self.decomp_50  = SeriesDecomp(kernel_size=13)
        self.decomp_25  = SeriesDecomp(kernel_size=7)

        # Mixing blocks
        self.mix_100 = MixingBlock(100, hidden_dim, dropout)
        self.mix_50  = MixingBlock(50,  hidden_dim, dropout)
        self.mix_25  = MixingBlock(25,  hidden_dim, dropout)

        # Fusion + head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # x: (B, 100, input_dim)
        x = self.input_proj(x)  # (B, 100, hidden_dim)

        # Scale 1: Full 100 steps
        s100, t100 = self.decomp_100(x)
        feat_100 = self.mix_100(s100 + t100)[:, -1, :]

        # Scale 2: 50 steps
        x50 = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        s50, t50 = self.decomp_50(x50)
        feat_50 = self.mix_50(s50 + t50)[:, -1, :]

        # Scale 3: 25 steps
        x25 = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
        s25, t25 = self.decomp_25(x25)
        feat_25 = self.mix_25(s25 + t25)[:, -1, :]

        fused = torch.cat([feat_100, feat_50, feat_25], dim=-1)
        return self.head(self.fusion(fused))
