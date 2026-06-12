"""
transformer_bigru.py — Sequential Fusion: Transformer → BiGRU
===============================================================
Architecture:
  Input → LOBFeatureEngineer → Transformer Encoder (global context)
       → BiGRU (local temporal refinement) → Head

Rationale: The Transformer sees the full 100-step window via attention
and extracts regime-level context. The BiGRU then refines this
attended representation with bidirectional temporal dynamics —
combining global pattern detection with fine-grained sequential modeling.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS
from models.enc_dec import LOBFeatureEngineer


TRANSFORMER_BIGRU_DEFAULTS = {
    "hidden_dim": 128,
    "n_heads": 4,
    "n_transformer_layers": 2,
    "n_gru_layers": 2,
    "dropout": 0.1,
    # Training config
    "batch_size": 2048,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 5,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class TransformerBiGRUModel(nn.Module):
    """
    Sequential Transformer → BiGRU fusion.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_gru_layers: int = 2,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feature engineering
        self.fe = LOBFeatureEngineer()
        enhanced_dim = self.fe.out_dim

        # Input projection
        self.input_proj = nn.Linear(enhanced_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)

        # Transformer Encoder (non-causal — BiGRU handles causality)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

        # BiGRU refinement
        self.bigru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim * 2)

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.fe(x)
        x = self.input_proj(x) + self.pos_emb[:, :T, :]
        x = self.transformer(x)          # (B, T, hidden_dim)
        x, _ = self.bigru(x)             # (B, T, hidden_dim * 2)
        x = self.gru_norm(x)
        out = x[:, -1, :]                # Last timestep
        return self.head(out)
