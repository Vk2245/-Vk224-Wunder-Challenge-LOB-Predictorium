"""
wavenet_dense.py — Dilated Causal Convolutions for LOB
=======================================================
Why this is perfect for TPU:
  - ZERO recurrence (unlike GRU/LSTM)
  - ZERO attention (unlike Transformer)
  - Pure dense convolutions → maps perfectly to TPU's matrix units
  - Exponentially growing receptive field: 2^0 + 2^1 + ... + 2^6 = 127 > 100 timesteps
  - Extremely fast inference (~0.5ms)

Architecture:
  Input → 1D Conv proj → 7 x DilatedResBlock(dilation=1,2,4,8,16,32,64)
        → Global pooling → Head

Each DilatedResBlock uses:
  - Causal (left-padded) dilated conv
  - Gated activation (tanh ⊙ sigmoid) — the WaveNet trick
  - 1x1 conv residual + skip connections

Architecture only — training handled by engine.py or train_tpu.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


WAVENET_DENSE_DEFAULTS = {
    "hidden_dim": 128,
    "n_blocks": 7,
    "dropout": 0.1,
    # Training config
    "batch_size": 4096,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 8,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class GatedDilatedResBlock(nn.Module):
    """
    Single dilated causal convolution block with gated activation.
    Uses the WaveNet gating mechanism: tanh(filter) ⊙ sigmoid(gate).
    """

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Causal padding: we pad left only
        self.pad = (kernel_size - 1) * dilation

        # Filter and gate convolutions (combined for efficiency)
        self.conv_fg = nn.Conv1d(channels, channels * 2, kernel_size,
                                 dilation=dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.skip_1x1 = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T)"""
        residual = x

        # Causal padding (left only)
        x_pad = F.pad(x, (self.pad, 0))
        fg = self.conv_fg(x_pad)

        # Gated activation
        f, g = fg.chunk(2, dim=1)
        z = torch.tanh(f) * torch.sigmoid(g)
        z = self.dropout(z)

        # Skip connection output
        skip = self.skip_1x1(z)

        # Residual output
        out = self.conv_1x1(z) + residual

        # LayerNorm (needs (B, T, C) format)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        return out, skip


class WaveNetDenseModel(nn.Module):
    """
    WaveNet-style dilated causal convolution model for LOB prediction.
    Pure dense operations — no recurrence, no attention.
    Ideal for TPU execution.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_blocks: int = 7,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection: (B, T, input_dim) → (B, hidden_dim, T)
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Dilated blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            GatedDilatedResBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Head: global representations → output
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        """x: (B, T, input_dim) → (B, output_dim)"""
        B, T, _ = x.shape

        # Transpose for Conv1d: (B, C, T)
        x = x.transpose(1, 2)
        x = self.input_conv(x)

        # Normalize
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)

        # Stack of dilated blocks with skip connections
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        # Aggregate: use both last timestep AND global mean
        skip_out = skip_sum.transpose(1, 2)  # (B, T, C)
        last = skip_out[:, -1, :]             # (B, C) — causal last token
        mean = skip_out.mean(dim=1)           # (B, C) — global context

        combined = torch.cat([last, mean], dim=-1)  # (B, C*2)
        return self.head(combined)
