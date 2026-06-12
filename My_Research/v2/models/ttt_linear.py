"""
ttt_linear.py — Test-Time Training Linear Layers
==================================================
Novel for LOB: TTT layers (Sun et al., ICML 2024) replace the hidden state
with a learnable model that adapts DURING INFERENCE via self-supervised
reconstruction. No prior work applies TTT to financial time series.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


TTT_LINEAR_DEFAULTS = {
    "hidden_dim": 96,
    "n_blocks": 2,
    "dropout": 0.1,
    # Training config
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 3,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
}


class TTTLinearLayer(nn.Module):
    """
    Test-Time Training Linear Layer (Sun et al., 2024).

    Hidden state is a linear model W that gets updated per-token via
    a self-supervised reconstruction loss.

    At each timestep t:
        1. Compute key k_t, query q_t, value v_t from x_t
        2. Self-supervised loss: L = ||W * k_t - v_t||^2
        3. Update W via gradient step: W <- W - eta * grad_W(L)
        4. Output: y_t = W * q_t
    """

    def __init__(self, dim: int, inner_dim: int = None, ttt_lr: float = 0.1):
        super().__init__()
        self.dim = dim
        inner_dim = inner_dim or dim

        self.W_K = nn.Linear(dim, inner_dim, bias=False)
        self.W_V = nn.Linear(dim, inner_dim, bias=False)
        self.W_Q = nn.Linear(dim, inner_dim, bias=False)

        self.W0 = nn.Parameter(torch.eye(inner_dim) * 0.01)
        self.ttt_lr = nn.Parameter(torch.tensor(ttt_lr))

        self.inner_dim = inner_dim
        self.out_proj = nn.Linear(inner_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        d = self.inner_dim

        K = self.W_K(x)
        V = self.W_V(x)
        Q = self.W_Q(x)

        outputs = []
        W = self.W0.unsqueeze(0).expand(B, -1, -1).clone()
        lr = torch.sigmoid(self.ttt_lr) * 0.1

        step = max(1, T // 20)
        for t in range(0, T, step):
            k_t = K[:, t, :]
            v_t = V[:, t, :]

            pred = torch.bmm(W, k_t.unsqueeze(-1)).squeeze(-1)
            error = v_t - pred

            delta_W = lr * torch.bmm(
                error.unsqueeze(-1),
                k_t.unsqueeze(-2),
            )
            W = W + delta_W

            del pred, error, delta_W

        q_final = Q[:, -1, :]
        y_final = torch.bmm(W, q_final.unsqueeze(-1)).squeeze(-1)

        out = y_final.unsqueeze(1).expand(-1, T, -1)
        out = self.out_proj(out)
        out = self.norm(out + x)

        return out


class TTTLinearBlock(nn.Module):
    """TTT-Linear block with feed-forward network."""

    def __init__(self, dim: int, inner_dim: int = None, ff_mult: int = 2):
        super().__init__()
        self.ttt = TTTLinearLayer(dim, inner_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        x = self.ttt(x)
        x = x + self.ffn(x)
        return x


class TTTLinearModel(nn.Module):
    """
    Full TTT-Linear model for LOB prediction.
    Input projection -> TTTLinearBlock x N -> Pool last -> Head
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 96,
        n_blocks: int = 2,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TTTLinearBlock(hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = x[:, -1, :]
        return self.head(self.dropout(out))
