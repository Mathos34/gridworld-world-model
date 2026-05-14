"""Latent world model: encoder + transition predictor in latent space (JEPA-style).

We add a small auxiliary `position head` that decodes the agent cell from the
latent. On an 8x8 gridworld with pure latent-only objectives the geometry of
the latent is too unconstrained to plan reliably with L2-to-goal, so we ground
the latent with this auxiliary supervision while keeping the JEPA-style L2+VC
loss as the main objective.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import GRID_SIZE, N_ACTIONS, N_CHANNELS

OBS_DIM = N_CHANNELS * GRID_SIZE * GRID_SIZE
LATENT_DIM = 64
N_CELLS = GRID_SIZE * GRID_SIZE


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, 16)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        emb = self.action_emb(a)
        return self.net(torch.cat([z, emb], dim=-1))


class PositionHead(nn.Module):
    """Decodes the agent cell index (0..63) from a latent — auxiliary grounding."""

    def __init__(self, latent_dim: int = LATENT_DIM, n_cells: int = N_CELLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_cells),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.position_head = PositionHead(latent_dim)

    def forward(self, obs_t: torch.Tensor, action: torch.Tensor, obs_tp1: torch.Tensor):
        z_t = self.encoder(obs_t)
        z_tp1_pred = self.predictor(z_t, action)
        with torch.no_grad():
            z_tp1_target = self.encoder(obs_tp1)
        pos_logits_t = self.position_head(z_t)
        pos_logits_tp1 = self.position_head(z_tp1_pred)
        return z_t, z_tp1_pred, z_tp1_target, pos_logits_t, pos_logits_tp1
