"""
transformer_timemixer.py — Parallel Fusion: Transformer + TimeMixer
====================================================================
Architecture:
  Input → [Transformer Branch]  ─┐
  Input → [TimeMixer Branch]    ─┤→ Concat → Fusion → Head

Rationale: Transformer is powerful but slow. TimeMixer is weak on global
context but ultra-fast. Running both in parallel and concatenating lets
us get the best of both worlds without the cost of a sequential fusion.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS
from models.timemixer import MovingAvg, SeriesDecomp, MixingBlock


TRANSFORMER_TIMEMIXER_DEFAULTS = {
    "hidden_dim": 128,
    "n_heads": 4,
    "n_transformer_layers": 2,
    "dropout": 0.1,
    # Training config
    "batch_size": 2048,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 6,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class TransformerTimeMixerModel(nn.Module):
    """
    Parallel Transformer + TimeMixer fusion.
    Both branches see the same raw input independently.
    Their final representations are concatenated and fused.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Transformer Branch ────────────────────────────────────────────────
        self.t_proj = nn.Linear(input_dim, hidden_dim)
        self.t_pos_emb = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # ── TimeMixer Branch ───────────────────────────────────────────────────
        self.tm_proj = nn.Linear(input_dim, hidden_dim)
        self.decomp_100 = SeriesDecomp(kernel_size=25)
        self.decomp_50  = SeriesDecomp(kernel_size=13)
        self.decomp_25  = SeriesDecomp(kernel_size=7)
        self.mix_100 = MixingBlock(100, hidden_dim, dropout)
        self.mix_50  = MixingBlock(50,  hidden_dim, dropout)
        self.mix_25  = MixingBlock(25,  hidden_dim, dropout)
        self.tm_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Fusion Head ────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape

        # Transformer branch
        t_x = self.t_proj(x) + self.t_pos_emb[:, :T, :]
        t_x = self.transformer(t_x)
        t_out = t_x[:, -1, :]          # (B, hidden_dim)

        # TimeMixer branch
        tm_x = self.tm_proj(x)

        s100, t100 = self.decomp_100(tm_x)
        feat_100 = self.mix_100(s100 + t100)[:, -1, :]

        x50 = F.avg_pool1d(tm_x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        s50, t50 = self.decomp_50(x50)
        feat_50 = self.mix_50(s50 + t50)[:, -1, :]

        x25 = F.avg_pool1d(tm_x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)
        s25, t25 = self.decomp_25(x25)
        feat_25 = self.mix_25(s25 + t25)[:, -1, :]

        tm_out = self.tm_fuse(torch.cat([feat_100, feat_50, feat_25], dim=-1))  # (B, hidden_dim)

        # Concat + head
        fused = torch.cat([t_out, tm_out], dim=-1)   # (B, hidden_dim * 2)
        return self.head(fused)
