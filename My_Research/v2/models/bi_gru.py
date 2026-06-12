"""
bi_gru.py — Bidirectional GRU (Competition-Proven Architecture)
===============================================================
Re-built inside v2 harness from _archive_phase1/src/train_dual_optimized.py.
3-layer BiGRU with LayerNorm after bidirectional merge.
Best architecture from Phase 1 competition.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


BI_GRU_DEFAULTS = {
    "hidden_dim": 256,
    "n_layers": 3,
    "dropout": 0.15,
    # Training config
    "batch_size": 4096,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 5,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class BiGRUModel(nn.Module):
    """
    Bidirectional GRU for LOB prediction.

    3 BiGRU layers with LayerNorm between each layer.
    Forward + backward hidden states concatenated at each layer output.
    Final hidden state → 2-layer MLP head.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 256,
        n_layers: int = 3,
        output_dim: int = N_TARGETS,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Stack of BiGRU layers with LayerNorm in between
        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_dim
        for i in range(n_layers):
            self.gru_layers.append(
                nn.GRU(in_dim, hidden_dim, batch_first=True, bidirectional=True)
            )
            # After BiGRU: output dim = hidden_dim * 2 (fwd + bwd)
            self.norms.append(nn.LayerNorm(hidden_dim * 2))
            in_dim = hidden_dim * 2  # Input to next layer

        self.dropout = nn.Dropout(dropout)

        # Head: takes the final timestep's representation
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        for gru, norm in zip(self.gru_layers, self.norms):
            x, _ = gru(x)          # (B, T, hidden_dim * 2)
            x = norm(x)
            x = self.dropout(x)

        # Take the last time step
        out = x[:, -1, :]          # (B, hidden_dim * 2)
        return self.head(out)
