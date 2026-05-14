"""Cross-Entropy Method planner operating in latent space.

Reward shaping: we score plans by Manhattan distance between the position
predicted by the auxiliary position head and the goal cell. The latent
transition predictor handles the dynamics, the position head handles the
grounding. This keeps the planner principled while staying in latent space.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import GRID_SIZE, GridWorld, N_ACTIONS, cell_to_idx, encode_obs
from .model import WorldModel


@torch.no_grad()
def predicted_cell(model: WorldModel, z: torch.Tensor) -> torch.Tensor:
    logits = model.position_head(z)
    return logits.argmax(dim=-1)


def _cell_to_rc(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cells // GRID_SIZE, cells % GRID_SIZE


@torch.no_grad()
def cem_plan(model: WorldModel, obs: np.ndarray, goal: tuple[int, int],
             horizon: int = 15, n_candidates: int = 100, n_elite: int = 10,
             n_iter: int = 5, rng: np.random.Generator | None = None) -> int:
    """Plan a horizon of actions; return the first action of the converged distribution.

    Cost(plan) = sum_t Manhattan(decoded_pos(z_t), goal). We also bonus reaching the goal.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    device = next(model.parameters()).device
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    z0 = model.encoder(obs_t)
    goal_r, goal_c = goal

    logits = np.zeros((horizon, N_ACTIONS), dtype=np.float32)

    for _ in range(n_iter):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        actions = np.zeros((n_candidates, horizon), dtype=np.int64)
        for t in range(horizon):
            actions[:, t] = rng.choice(N_ACTIONS, size=n_candidates, p=probs[t])

        z = z0.expand(n_candidates, -1).clone()
        cost = torch.zeros(n_candidates, device=device)
        reached_bonus = torch.zeros(n_candidates, device=device)
        for t in range(horizon):
            a = torch.from_numpy(actions[:, t]).to(device)
            z = model.predictor(z, a)
            cells = predicted_cell(model, z).cpu().numpy()
            rs, cs = _cell_to_rc(cells)
            manh = np.abs(rs - goal_r) + np.abs(cs - goal_c)
            cost = cost + torch.from_numpy(manh.astype(np.float32)).to(device)
            reached = torch.from_numpy((manh == 0).astype(np.float32)).to(device)
            reached_bonus = torch.maximum(reached_bonus, reached * (horizon - t))
        score = -cost + 5.0 * reached_bonus

        elite_idx = torch.topk(score, k=n_elite).indices.cpu().numpy()
        elite_actions = actions[elite_idx]
        new_logits = np.zeros_like(logits)
        for t in range(horizon):
            counts = np.bincount(elite_actions[:, t], minlength=N_ACTIONS).astype(np.float32)
            counts = counts + 0.1
            new_logits[t] = np.log(counts / counts.sum())
        logits = 0.3 * logits + 0.7 * new_logits

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return int(np.argmax(probs[0]))


def evaluate_planner(model: WorldModel, n_episodes: int = 100, max_steps: int = 30,
                     seed: int = 1234, **plan_kwargs) -> tuple[float, list[bool]]:
    rng = np.random.default_rng(seed)
    env = GridWorld(seed=seed)
    successes = []
    for ep in range(n_episodes):
        obs = env.reset()
        success = False
        for _ in range(max_steps):
            action = cem_plan(model, obs, env.goal, rng=rng, **plan_kwargs)
            obs, done = env.step(action)
            if done:
                success = True
                break
        successes.append(success)
    return float(np.mean(successes)), successes


def evaluate_random(n_episodes: int = 100, max_steps: int = 30, seed: int = 1234) -> float:
    rng = np.random.default_rng(seed)
    env = GridWorld(seed=seed)
    n_success = 0
    for _ in range(n_episodes):
        env.reset()
        for _ in range(max_steps):
            a = int(rng.integers(0, N_ACTIONS))
            _, done = env.step(a)
            if done:
                n_success += 1
                break
    return n_success / n_episodes
