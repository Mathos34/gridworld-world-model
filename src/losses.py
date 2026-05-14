"""Multi-step JEPA loss + VC regularizer + position grounding."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / d


def multi_step_world_loss(model, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                          pos_seq: torch.Tensor,
                          w_pred: float = 1.0, w_var: float = 1.0, w_cov: float = 0.04,
                          w_pos: float = 1.0):
    """JEPA-style per-step loss aggregated over K transitions.

    For each transition (obs_t, action_t, obs_{t+1}) in the trajectory we encode
    obs_t and obs_{t+1}, predict the next latent from z_t and the action, and
    apply MSE to a stop-gradient encoding of obs_{t+1}. We also decode the agent
    cell from every encoded latent and supervise it with cross-entropy. K
    independent single-step signals per trajectory, so errors don't compound.

    obs_seq: (B, K+1, 3, H, W)
    act_seq: (B, K)
    pos_seq: (B, K+1)
    """
    B, Kp1 = obs_seq.shape[0], obs_seq.shape[1]
    K = Kp1 - 1
    flat_obs = obs_seq.reshape(B * Kp1, *obs_seq.shape[2:])
    z_all = model.encoder(flat_obs).reshape(B, Kp1, -1)
    z_target_all = z_all.detach()

    pos_logits_all = model.position_head(z_all.reshape(B * Kp1, -1)).reshape(B, Kp1, -1)
    pos_loss = F.cross_entropy(pos_logits_all.reshape(B * Kp1, -1), pos_seq.reshape(-1))

    z_t = z_all[:, :K].reshape(B * K, -1)
    a_t = act_seq.reshape(-1)
    z_pred = model.predictor(z_t, a_t)
    z_target = z_target_all[:, 1:].reshape(B * K, -1)
    pred_loss = F.mse_loss(z_pred, z_target)

    z_flat = z_all.reshape(B * Kp1, -1)
    var = variance_loss(z_flat)
    cov = covariance_loss(z_flat)

    total = w_pred * pred_loss + w_var * var + w_cov * cov + w_pos * pos_loss
    return total, {"pred": pred_loss.item(), "var": var.item(), "cov": cov.item(),
                   "pos": pos_loss.item()}
