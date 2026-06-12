"""
xlstm.py — xLSTM (Extended LSTM) with Exponential Gating
==========================================================
Novel for LOB: Exponential gating enables unbounded forget gates,
allowing faster memory adaptation to regime shifts than standard LSTM.
sLSTM variant (scalar memory) for efficiency.

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


# Default hyperparameters + training config
XLSTM_DEFAULTS = {
    "hidden_dim": 128,
    "proj_dim": 256,
    "n_blocks": 2,
    "dropout": 0.15,
    # Training config
    "batch_size": 1024,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "epochs": 3,
    "patience": 7,
    "use_amp": False,  # AMP causes NaN with exp gates
    "use_compile": True,
}


class ExpGateSLSTMCell(nn.Module):
    """
    sLSTM cell with STABILIZED exponential gating.
    Hybrid sigmoid-exp gates for stability while maintaining
    the unbounded memory capacity benefit of exponential gating.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_norm = nn.LayerNorm(input_size)
        self.hidden_norm = nn.LayerNorm(hidden_size)

        self.W = nn.Linear(input_size, 4 * hidden_size)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        with torch.no_grad():
            self.W.weight.data *= 0.1
            self.U.weight.data *= 0.1
            self.W.bias.data.zero_()
            self.W.bias[hidden_size:2*hidden_size].fill_(0.0)

    def forward(self, x_t, state):
        h, c, n = state

        # Differentiable, export-friendly NaN handling
        h = torch.nan_to_num(h, nan=0.0)
        c = torch.nan_to_num(c, nan=0.0)
        n = torch.nan_to_num(n, nan=1.0)

        x_norm = self.input_norm(x_t)
        h_norm = self.hidden_norm(h)

        gates = self.W(x_norm) + self.U(h_norm)
        i_pre, f_pre, o_pre, z_pre = gates.chunk(4, dim=-1)

        i_base = torch.sigmoid(i_pre)
        f_base = torch.sigmoid(f_pre)

        i_boost = torch.exp(torch.clamp(i_pre, -2, 2)) * 0.1
        f_boost = torch.exp(torch.clamp(f_pre, -2, 2)) * 0.1

        i_t = i_base + i_boost
        f_t = f_base + f_boost

        o_t = torch.sigmoid(o_pre)
        z_t = torch.tanh(z_pre)

        c_new = f_t * c + i_t * z_t
        c_new = torch.clamp(c_new, -10, 10)

        n_new = torch.max(torch.abs(f_t * n + i_t), torch.ones_like(n))

        h_new = o_t * (c_new / n_new)

        return h_new, (h_new, c_new, n_new)


class xLSTMBlock(nn.Module):
    """xLSTM block: sLSTM + post-up projection for cross-variate mixing."""

    def __init__(self, input_size: int, hidden_size: int, proj_size: int):
        super().__init__()
        self.cell = ExpGateSLSTMCell(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.up_proj = nn.Linear(hidden_size, proj_size)
        self.down_proj = nn.Linear(proj_size, hidden_size)
        self.gate = nn.Linear(hidden_size, proj_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        n = torch.ones(B, self.hidden_size, device=device)

        for t in range(T):
            h, (h, c, n) = self.cell(x[:, t, :], (h, c, n))

        h_norm = self.norm(h)
        up = self.up_proj(h_norm)
        gate = torch.sigmoid(self.gate(h_norm))
        out = self.down_proj(F.gelu(up) * gate)

        return out


class xLSTMModel(nn.Module):
    """
    Full xLSTM model for LOB prediction.
    Input projection -> xLSTM Block x N -> Head
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        proj_dim: int = 256,
        n_blocks: int = 2,
        output_dim: int = N_TARGETS,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            xLSTMBlock(hidden_dim, hidden_dim, proj_dim)
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
            x_out = block(x)
        out = self.head(self.dropout(x_out))
        return out
