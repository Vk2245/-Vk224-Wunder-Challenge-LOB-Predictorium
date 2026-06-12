"""
itransformer.py — Inverted Transformer for LOB
=================================================
Inspired by iTransformer (Liu et al., 2024).

KEY INSIGHT: Standard transformers apply attention across TIME.
iTransformer applies attention across FEATURES.

Why this is powerful for LOB:
  - Each of 32 features becomes a "token"
  - Attention learns: "When spread (p0-p6) narrows AND bid volume (v6) surges,
    price will move up" — these CROSS-FEATURE correlations are exactly
    what drives both t0 and t1
  - Features are embedded via their FULL temporal context (100 timesteps)
  - Extremely TPU-friendly: 32 tokens × 32 tokens attention = tiny

Architecture:
  Input (B,100,32) → transpose → embed each feature's timeline → 
  Transformer(32 tokens) → project → Head

This is the opposite of what every other model does, and that's exactly
why it might capture signals they all miss.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS, SEQ_LEN


ITRANSFORMER_DEFAULTS = {
    "hidden_dim": 128,
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


class iTransformerModel(nn.Module):
    """
    Inverted Transformer: attention across FEATURES, not time.
    Each feature's full 100-step timeline becomes a token embedding.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Embed each feature's timeline (100 timesteps) into hidden_dim
        self.feature_embed = nn.Linear(seq_len, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # Learned feature-type embeddings (like token type embeddings in BERT)
        self.feature_type_emb = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)

        # Transformer on feature tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Project back: each feature token → scalar predictions
        # Pool all feature tokens → final prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (B, T=100, C=32)
        B, T, C = x.shape

        # INVERT: treat each feature as a token with timeline as its embedding
        x = x.transpose(1, 2)               # (B, C=32, T=100)
        x = self.feature_embed(x)            # (B, 32, hidden_dim)
        x = self.feature_norm(x)
        x = x + self.feature_type_emb[:, :C, :]

        # Attention across features
        x = self.transformer(x)              # (B, 32, hidden_dim)
        x = self.norm(x)

        # Flatten all feature representations
        x = x.reshape(B, -1)                 # (B, 32 * hidden_dim)
        return self.head(x)
