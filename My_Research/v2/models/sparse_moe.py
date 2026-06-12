"""
sparse_moe.py — Sparse Mixture of Experts with Regime-Aware Routing
=====================================================================
Novel for LOB: Standard ensembles use fixed weights. MoE learns a router
that activates different expert networks for different market conditions.
The router learns to detect regimes (high-vol, trending, mean-reverting).

Architecture only — training handled by engine.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS


SPARSE_MOE_DEFAULTS = {
    "hidden_dim": 128,
    "n_experts": 4,
    # Training config
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 3,
    "patience": 7,
    "use_amp": True,
    "use_compile": True,
    "aux_weight": 0.01,  # MoE load-balancing loss weight
}


class Expert(nn.Module):
    """Single expert: GRU temporal encoding + 2-layer MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        _, h = self.gru(x)
        h = h.squeeze(0)
        return self.mlp(h)


class RegimeRouter(nn.Module):
    """
    Lightweight regime-aware router.
    Uses last K steps + global stats to decide which experts to activate.
    """

    def __init__(self, input_dim: int, n_experts: int, lookback: int = 20):
        super().__init__()
        self.lookback = lookback
        router_input_dim = input_dim * lookback + input_dim * 2
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_experts),
        )
        self.n_experts = n_experts

    def forward(self, x):
        B, T, D = x.shape

        last_k = x[:, -self.lookback:, :].reshape(B, -1)
        mu = x.mean(dim=1)
        std = x.std(dim=1)

        router_input = torch.cat([last_k, mu, std], dim=-1)
        logits = self.router(router_input)

        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        top2_vals, top2_idx = logits.topk(2, dim=-1)
        top2_weights = F.softmax(top2_vals, dim=-1)

        return logits, top2_weights, top2_idx


class SparseMoEModel(nn.Module):
    """
    Sparse Mixture-of-Experts for LOB prediction.

    Input -> Router (selects top-2 experts)
    Input -> All Experts (but only top-2 contribute to output)
    Output = weighted sum of top-2 expert outputs

    Returns (output, aux_loss) during training.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 128,
        n_experts: int = 4,
        output_dim: int = N_TARGETS,
    ):
        super().__init__()
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(n_experts)
        ])

        self.router = RegimeRouter(input_dim, n_experts, lookback=20)

    def forward(self, x, return_routing_info: bool = False):
        B = x.size(0)

        logits, top2_weights, top2_idx = self.router(x)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )

        idx_expanded = top2_idx.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        selected = torch.gather(expert_outputs, 1, idx_expanded)

        weights_expanded = top2_weights.unsqueeze(-1)
        output = (selected * weights_expanded).sum(dim=1)

        # Load balancing auxiliary loss
        routing_probs = F.softmax(logits, dim=-1)
        avg_routing = routing_probs.mean(dim=0)
        aux_loss = self.n_experts * (avg_routing * avg_routing).sum()

        if return_routing_info:
            return output, aux_loss, top2_idx
        return output, aux_loss


class SparseMoEForExport(nn.Module):
    """Wrapper that drops aux_loss for ONNX export."""
    def __init__(self, moe):
        super().__init__()
        self.moe = moe
    def forward(self, x):
        pred, _ = self.moe(x)
        return pred
