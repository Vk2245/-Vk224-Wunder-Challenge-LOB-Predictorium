"""
triple_fusion.py — Triple Gated Fusion: Transformer + BiGRU + TimeMixer
=========================================================================
Architecture:
  Input → [Transformer] ─┐
  Input → [BiGRU]       ─┤→ Learned Gated Fusion → Head
  Input → [TimeMixer]   ─┘

Rationale: Each model captures a fundamentally different signal:
  - Transformer: Global attention patterns, regime context
  - BiGRU:       Local bidirectional temporal dynamics  
  - TimeMixer:   Multiscale seasonal/trend decomposition

A lightweight gating network learns to adaptively weight each branch's
contribution per-sample, based on the input itself.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS
from models.timemixer import SeriesDecomp, MixingBlock


TRIPLE_FUSION_DEFAULTS = {
    "hidden_dim": 96,
    "n_heads": 4,
    "dropout": 0.1,
    # Training config
    "batch_size": 1024,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 5,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class TripleFusionModel(nn.Module):
    """
    Three-branch gated fusion model.
    Each branch is a lightweight version of the standalone architecture.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 96,
        n_heads: int = 4,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Branch 1: Transformer ─────────────────────────────────────────────
        self.t_proj = nn.Linear(input_dim, hidden_dim)
        self.t_pos = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ── Branch 2: BiGRU ───────────────────────────────────────────────────
        self.g_proj = nn.Linear(input_dim, hidden_dim)
        self.bigru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim * 2)
        self.gru_compress = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Branch 3: TimeMixer ───────────────────────────────────────────────
        self.tm_proj = nn.Linear(input_dim, hidden_dim)
        self.decomp_100 = SeriesDecomp(kernel_size=25)
        self.decomp_50  = SeriesDecomp(kernel_size=13)
        self.decomp_25  = SeriesDecomp(kernel_size=7)
        self.mix_100 = MixingBlock(100, hidden_dim, dropout)
        self.mix_50  = MixingBlock(50,  hidden_dim, dropout)
        self.mix_25  = MixingBlock(25,  hidden_dim, dropout)
        self.tm_fuse = nn.Linear(hidden_dim * 3, hidden_dim)

        # ── Gating Network ────────────────────────────────────────────────────
        # Takes global stats of input to decide branch weights
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # global mean + std of input
            nn.GELU(),
            nn.Linear(64, 3),              # 3 branch weights
        )

        # ── Output Head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape

        # ── Transformer Branch ────────────────────────────────────────────────
        t = self.t_proj(x) + self.t_pos[:, :T, :]
        t = self.transformer(t)
        t_out = t[:, -1, :]                              # (B, hidden_dim)

        # ── BiGRU Branch ──────────────────────────────────────────────────────
        g = self.g_proj(x)
        g, _ = self.bigru(g)
        g = self.gru_norm(g)
        g_out = self.gru_compress(g[:, -1, :])           # (B, hidden_dim)

        # ── TimeMixer Branch ──────────────────────────────────────────────────
        tm = self.tm_proj(x)
        s100, t100 = self.decomp_100(tm)
        f100 = self.mix_100(s100 + t100)[:, -1, :]
        x50 = F.avg_pool1d(tm.transpose(1, 2), 2, 2).transpose(1, 2)
        s50, t50 = self.decomp_50(x50)
        f50 = self.mix_50(s50 + t50)[:, -1, :]
        x25 = F.avg_pool1d(tm.transpose(1, 2), 4, 4).transpose(1, 2)
        s25, t25 = self.decomp_25(x25)
        f25 = self.mix_25(s25 + t25)[:, -1, :]
        tm_out = self.tm_fuse(torch.cat([f100, f50, f25], dim=-1))  # (B, hidden_dim)

        # ── Gated Fusion ──────────────────────────────────────────────────────
        gate_input = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        weights = F.softmax(self.gate(gate_input), dim=-1)  # (B, 3)

        fused = (
            weights[:, 0:1] * t_out +
            weights[:, 1:2] * g_out +
            weights[:, 2:3] * tm_out
        )                                                    # (B, hidden_dim)

        return self.head(fused)
