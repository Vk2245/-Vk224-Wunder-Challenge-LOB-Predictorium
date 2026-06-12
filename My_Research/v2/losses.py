"""
losses.py — Novel loss functions for LOB prediction (v2)
==========================================================
Every LOB model in the literature uses MSE. This is wrong:
LOB returns are heavy-tailed, heteroscedastic, and non-Gaussian.

Provides:
  1. HeteroscedasticGaussianLoss — predicts mean + variance, downweights noisy steps
  2. PinballLoss — quantile regression, more robust to outliers
  3. MultiQuantileLoss — multi-quantile simultaneous prediction
  4. CombinedLoss — weighted blend of MSE + heteroscedastic NLL
  5. DualHeadWrapper — wraps any backbone with mean + variance heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroscedasticGaussianLoss(nn.Module):
    """
    NLL under heteroscedastic Gaussian assumption.
    Model predicts both mu and log(sigma^2).
    Loss = 0.5 * log(sigma^2) + 0.5 * (y - mu)^2 / sigma^2

    Naturally downweights noisy timesteps where sigma^2 is large.
    """

    def __init__(self, min_log_var: float = -6.0, max_log_var: float = 6.0):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, mu, log_var, target):
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        precision = torch.exp(-log_var)
        loss = 0.5 * log_var + 0.5 * precision * (target - mu) ** 2
        return loss.mean()


class PinballLoss(nn.Module):
    """
    Pinball (quantile) loss for robust regression.
    At tau=0.5, equivalent to MAE (median regression).
    More robust to outliers than MSE.
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, pred, target):
        error = target - pred
        loss = torch.where(error >= 0, self.tau * error, (self.tau - 1.0) * error)
        return loss.mean()


class MultiQuantileLoss(nn.Module):
    """
    Multi-quantile loss — predict multiple quantiles simultaneously.
    At inference, use the median (tau=0.5) prediction.
    """

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

    def forward(self, pred, target):
        n_targets = target.shape[1]
        total_loss = 0.0

        for i, tau in enumerate(self.quantiles):
            q_pred = pred[:, i * n_targets : (i + 1) * n_targets]
            error = target - q_pred
            loss = torch.where(error >= 0, tau * error, (tau - 1.0) * error)
            total_loss = total_loss + loss.mean()

        return total_loss / self.n_quantiles


class CombinedLoss(nn.Module):
    """
    Weighted blend: alpha * MSE + (1-alpha) * PinballLoss.
    Default alpha=0.7 emphasises MSE but adds robustness from pinball.
    """

    def __init__(self, alpha: float = 0.7, pinball_tau: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.pinball = PinballLoss(tau=pinball_tau)

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        pin = self.pinball(pred, target)
        return self.alpha * mse + (1 - self.alpha) * pin


class DualHeadWrapper(nn.Module):
    """
    Wraps any backbone model with a dual head for heteroscedastic prediction.

    backbone(x) -> hidden
    hidden -> mu_head -> mu (mean prediction)
    hidden -> var_head -> log_var (uncertainty)
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int, n_targets: int = 2):
        super().__init__()
        self.backbone = backbone
        self.mu_head = nn.Linear(hidden_dim, n_targets)
        self.var_head = nn.Linear(hidden_dim, n_targets)

        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -2.0)

    def forward(self, x, return_variance: bool = False):
        hidden = self.backbone(x)
        mu = self.mu_head(hidden)

        if return_variance:
            log_var = self.var_head(hidden)
            return mu, log_var

        return mu
