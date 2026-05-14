"""Gridworld environment and random-trajectory dataset.

Observations are 3-channel one-hot grids: (agent, wall, goal).
Actions: 0=up, 1=down, 2=left, 3=right.
We also record agent cell indices for the auxiliary position head.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

GRID_SIZE = 8
N_ACTIONS = 4
N_CHANNELS = 3
ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)


def cell_to_idx(r: int, c: int) -> int:
    return r * GRID_SIZE + c


def _free_cell(rng: np.random.Generator, walls: np.ndarray, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        r, c = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        if walls[r, c] == 0 and (r, c) not in forbidden:
            return r, c


def make_walls(rng: np.random.Generator) -> np.ndarray:
    """Build a vertical or horizontal wall with a single random gap."""
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    vertical = bool(rng.integers(0, 2))
    line = int(rng.integers(2, GRID_SIZE - 2))
    gap = int(rng.integers(0, GRID_SIZE))
    if vertical:
        walls[:, line] = 1
        walls[gap, line] = 0
    else:
        walls[line, :] = 1
        walls[line, gap] = 0
    return walls


def encode_obs(agent: tuple[int, int], walls: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    obs = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    obs[0, agent[0], agent[1]] = 1.0
    obs[1] = walls.astype(np.float32)
    obs[2, goal[0], goal[1]] = 1.0
    return obs


class GridWorld:
    """Minimal deterministic gridworld. Walls block movement; out-of-bounds blocks."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, walls: np.ndarray | None = None, agent: tuple[int, int] | None = None,
              goal: tuple[int, int] | None = None) -> np.ndarray:
        self.walls = walls if walls is not None else make_walls(self.rng)
        if agent is None:
            agent = _free_cell(self.rng, self.walls, set())
        if goal is None:
            goal = _free_cell(self.rng, self.walls, {agent})
        self.agent = agent
        self.goal = goal
        return self.obs()

    def obs(self) -> np.ndarray:
        return encode_obs(self.agent, self.walls, self.goal)

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        dr, dc = ACTIONS[action]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and self.walls[nr, nc] == 0:
            self.agent = (nr, nc)
        done = self.agent == self.goal
        return self.obs(), done


class TrajectoryDataset(Dataset):
    """Collect (obs_t, action, obs_{t+1}, pos_t, pos_{t+1}) tuples."""

    def __init__(self, n_traj: int = 10_000, traj_len: int = 20, seed: int = 42):
        rng = np.random.default_rng(seed)
        env = GridWorld(seed=seed)
        obs_t, acts, obs_tp1, pos_t, pos_tp1 = [], [], [], [], []
        for _ in range(n_traj):
            o = env.reset()
            for _ in range(traj_len):
                a = int(rng.integers(0, N_ACTIONS))
                prev_agent = env.agent
                o_next, done = env.step(a)
                obs_t.append(o)
                acts.append(a)
                obs_tp1.append(o_next)
                pos_t.append(cell_to_idx(*prev_agent))
                pos_tp1.append(cell_to_idx(*env.agent))
                o = o_next
                if done:
                    break
        self.obs_t = np.stack(obs_t).astype(np.float32)
        self.acts = np.asarray(acts, dtype=np.int64)
        self.obs_tp1 = np.stack(obs_tp1).astype(np.float32)
        self.pos_t = np.asarray(pos_t, dtype=np.int64)
        self.pos_tp1 = np.asarray(pos_tp1, dtype=np.int64)

    def __len__(self) -> int:
        return self.obs_t.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.obs_t[idx]),
            torch.tensor(self.acts[idx], dtype=torch.long),
            torch.from_numpy(self.obs_tp1[idx]),
            torch.tensor(self.pos_t[idx], dtype=torch.long),
            torch.tensor(self.pos_tp1[idx], dtype=torch.long),
        )
