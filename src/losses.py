"""JEPA-style L2 prediction loss, VC regularizers, plus an auxiliary position CE."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_pred, z_target)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / d


def world_model_loss(z_t, z_pred, z_target, pos_logits_t, pos_logits_tp1,
                     pos_t, pos_tp1,
                     w_pred: float = 1.0, w_var: float = 1.0, w_cov: float = 0.04,
                     w_pos: float = 1.0):
    l_pred = prediction_loss(z_pred, z_target)
    l_var = 0.5 * (variance_loss(z_t) + variance_loss(z_pred))
    l_cov = 0.5 * (covariance_loss(z_t) + covariance_loss(z_pred))
    l_pos_t = F.cross_entropy(pos_logits_t, pos_t)
    l_pos_tp1 = F.cross_entropy(pos_logits_tp1, pos_tp1)
    l_pos = 0.5 * (l_pos_t + l_pos_tp1)
    total = w_pred * l_pred + w_var * l_var + w_cov * l_cov + w_pos * l_pos
    return total, {"pred": l_pred.item(), "var": l_var.item(), "cov": l_cov.item(),
                   "pos": l_pos.item()}
