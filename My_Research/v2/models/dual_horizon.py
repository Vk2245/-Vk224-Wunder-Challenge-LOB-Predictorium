"""
dual_horizon.py — Dual-Horizon Specialist for LOB
====================================================
THE KEY INSIGHT: t0 and t1 are fundamentally DIFFERENT tasks.
  - t0 = "What is the LOB telling us RIGHT NOW?" → last few timesteps matter
  - t1 = "How will the LOB EVOLVE?" → velocity, momentum, trajectory matter

Every other model treats them as the same regression target.
This model has SEPARATE EXPERT PATHWAYS:
  - Shared backbone extracts common representations
  - t0 expert: shallow, focuses on recent timesteps
  - t1 expert: deep, uses temporal differencing (velocity features)

Additional innovation: TEMPORAL AUGMENTATION
  - Computes first-order differences (velocity of each feature)
  - Computes second-order differences (acceleration)
  - These are computed in forward() — zero preprocessing needed
  - Velocity/acceleration signals are far more predictive of t1

Architecture:
  Input → TemporalAugment → SharedBackbone
        → t0_expert(last timestep focus) → t0 prediction
        → t1_expert(velocity + global context) → t1 prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_FEATURES, N_TARGETS, SEQ_LEN


DUAL_HORIZON_DEFAULTS = {
    "hidden_dim": 160,
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


class TemporalAugment(nn.Module):
    """
    Compute velocity and acceleration features on-the-fly.
    Input:  (B, T, C)  → 32 features
    Output: (B, T, C*3) → 32 raw + 32 velocity + 32 acceleration = 96 features

    Velocity = x[t] - x[t-1]  (how fast is each feature changing?)
    Acceleration = v[t] - v[t-1]  (is the change speeding up or slowing down?)

    First timesteps are zero-padded.
    """
    def forward(self, x):
        B, T, C = x.shape

        # Velocity: first-order difference
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        # Acceleration: second-order difference
        accel = torch.zeros_like(x)
        accel[:, 2:, :] = velocity[:, 2:, :] - velocity[:, 1:-1, :]

        return torch.cat([x, velocity, accel], dim=-1)  # (B, T, C*3)


class DualHorizonModel(nn.Module):
    """
    Dual-Horizon model with separate expert pathways for t0 and t1.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dim: int = 160,
        n_layers: int = 3,
        output_dim: int = N_TARGETS,
        dropout: float = 0.1,
    ):
        super().__init__()
        augmented_dim = input_dim * 3  # raw + velocity + acceleration

        # Temporal augmentation
        self.augment = TemporalAugment()

        # Shared backbone: lightweight transformer
        self.proj = nn.Linear(augmented_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.backbone_norm = nn.LayerNorm(hidden_dim)

        # ── t0 Expert: focuses on RECENT timesteps ─────────────────────────
        # t0 is about the current LOB state → last 10 timesteps matter most
        self.t0_attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),  # attention score per timestep
        )
        self.t0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # predict t0 only
        )

        # ── t1 Expert: focuses on TRAJECTORY + global context ──────────────
        # t1 needs velocity/momentum → use temporal conv to capture trends
        self.t1_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, padding=5, groups=4),
            nn.GELU(),
        )
        self.t1_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # predict t1 only
        )

    def forward(self, x):
        B, T, _ = x.shape

        # Augment with velocity + acceleration
        x = self.augment(x)               # (B, T, C*3)
        x = self.proj(x)                  # (B, T, hidden_dim)

        # Shared backbone
        x = self.backbone(x)
        x = self.backbone_norm(x)         # (B, T, hidden_dim)

        # ── t0 Expert ─────────────────────────────────────────────────────
        # Exponential recency weighting: later timesteps matter more
        t0_scores = self.t0_attn_pool(x).squeeze(-1)   # (B, T)
        # Mask: only last 20 timesteps contribute to t0
        mask = torch.zeros(T, device=x.device)
        mask[-20:] = 1.0
        t0_scores = t0_scores * mask.unsqueeze(0) + (1 - mask.unsqueeze(0)) * (-1e9)
        t0_weights = F.softmax(t0_scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        t0_repr = (x * t0_weights).sum(dim=1)          # (B, hidden_dim)
        t0_pred = self.t0_head(t0_repr)                # (B, 1)

        # ── t1 Expert ─────────────────────────────────────────────────────
        # Temporal convolutions capture velocity/momentum patterns
        t1_x = x.transpose(1, 2)                       # (B, hidden_dim, T)
        t1_x = self.t1_conv(t1_x)                      # (B, hidden_dim, T)
        t1_x = t1_x.transpose(1, 2)                    # (B, T, hidden_dim)
        t1_last = t1_x[:, -1, :]                       # (B, hidden_dim)
        t1_mean = t1_x.mean(dim=1)                     # (B, hidden_dim)
        t1_repr = torch.cat([t1_last, t1_mean], dim=-1)
        t1_pred = self.t1_head(t1_repr)                # (B, 1)

        # Stack: (B, 2) = [t0, t1]
        return torch.cat([t0_pred, t1_pred], dim=-1)
