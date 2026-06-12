"""
mlp_mixer.py — MLP-Mixer for LOB
==================================
Google's pure-MLP architecture (Tolstikhin et al., 2021).
Designed literally for TPUs — zero attention, zero convolutions.

Two types of mixing:
  - Token-mixing MLP: operates across TIME dimension (learns temporal patterns)
  - Channel-mixing MLP: operates across FEATURE dimension (learns feature interactions)

Why this helps:
  - Token-mixing captures temporal dependencies without positional bias → t1
  - Channel-mixing captures cross-feature correlations (spread↔volume) → t0 + t1
  - Pure dense matrix multiplications = TPU nirvana

Architecture:
  Input → Proj → N x (TokenMix + ChannelMix) → Pool → Head
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS, SEQ_LEN


MLP_MIXER_DEFAULTS = {
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
    "use_compile": False,
}


class MixerMLP(nn.Module):
    """Standard 2-layer MLP with GELU."""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """
    One Mixer block: TokenMix → ChannelMix.
    TokenMix: transpose → MLP across time → transpose back
    ChannelMix: MLP across features (standard)
    """
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = MixerMLP(seq_len, expansion=2, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = MixerMLP(hidden_dim, expansion=2, dropout=dropout)

    def forward(self, x):
        # Token mixing: (B, T, C) → transpose → MLP on T → transpose back
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.token_mix(x)
        x = x.transpose(1, 2)          # (B, T, C)
        x = x + residual

        # Channel mixing: standard MLP on C
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual

        return x


class MLPMixerModel(nn.Module):
    """
    MLP-Mixer for LOB prediction.
    Google's TPU-native architecture.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 192,
        n_layers: int = 4,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(seq_len, hidden_dim, dropout) for _ in range(n_layers)
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
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        return self.head(torch.cat([last, mean], dim=-1))
