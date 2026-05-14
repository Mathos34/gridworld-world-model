"""Gridworld environment + multi-step trajectory dataset.

Observations: 3-channel one-hot grids (agent, wall, goal). Actions: 0=up,
1=down, 2=left, 3=right. We expose K-step rollouts so the predictor can be
trained to be self-consistent over short horizons (multi-step prediction loss).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

GRID_SIZE = 8
N_ACTIONS = 4
N_CHANNELS = 3
N_CELLS = GRID_SIZE * GRID_SIZE
ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)


def cell_to_idx(r: int, c: int) -> int:
    return r * GRID_SIZE + c


def idx_to_cell(idx: int) -> tuple[int, int]:
    return idx // GRID_SIZE, idx % GRID_SIZE


def _free_cell(rng: np.random.Generator, walls: np.ndarray, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        r, c = int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE))
        if walls[r, c] == 0 and (r, c) not in forbidden:
            return r, c


def make_walls(rng: np.random.Generator) -> np.ndarray:
    """Vertical or horizontal wall with a random gap."""
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


def step_pos(walls: np.ndarray, agent: tuple[int, int], action: int) -> tuple[int, int]:
    dr, dc = ACTIONS[action]
    nr, nc = agent[0] + dr, agent[1] + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and walls[nr, nc] == 0:
        return nr, nc
    return agent


class TrajectoryDataset(Dataset):
    """Per item, a length-K *synthetic* trajectory starting from a uniformly
    sampled free cell on a freshly sampled (walls, goal). Every step picks a
    uniform random action and updates the agent position by the true dynamics.
    This gives uniform coverage of (walls, agent_position, action) instead of
    the biased coverage you get from rolling out a random policy in a single
    fixed environment.
    """

    def __init__(self, n_traj: int = 10_000, traj_len: int = 8, seed: int = 42):
        self.K = traj_len
        rng = np.random.default_rng(seed)
        obs_buf = np.zeros((n_traj, traj_len + 1, N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        act_buf = np.zeros((n_traj, traj_len), dtype=np.int64)
        pos_buf = np.zeros((n_traj, traj_len + 1), dtype=np.int64)
        for i in range(n_traj):
            walls = make_walls(rng)
            agent = _free_cell(rng, walls, set())
            goal = _free_cell(rng, walls, {agent})
            obs_buf[i, 0] = encode_obs(agent, walls, goal)
            pos_buf[i, 0] = cell_to_idx(*agent)
            for t in range(traj_len):
                a = int(rng.integers(0, N_ACTIONS))
                agent = step_pos(walls, agent, a)
                obs_buf[i, t + 1] = encode_obs(agent, walls, goal)
                act_buf[i, t] = a
                pos_buf[i, t + 1] = cell_to_idx(*agent)
        self.obs = obs_buf
        self.act = act_buf
        self.pos = pos_buf

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.pos[idx]),
        )


class UniformTransitionDataset(Dataset):
    """Per item, an (obs_t, action, obs_{t+1}, pos_t, pos_{t+1}) tuple sampled
    by drawing fresh (walls, agent, goal, action) uniformly at random. Best
    coverage of the state-action space; also fastest to generate."""

    def __init__(self, n_samples: int = 80_000, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.K = 1
        obs_buf = np.zeros((n_samples, 2, N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        act_buf = np.zeros((n_samples, 1), dtype=np.int64)
        pos_buf = np.zeros((n_samples, 2), dtype=np.int64)
        for i in range(n_samples):
            walls = make_walls(rng)
            agent = _free_cell(rng, walls, set())
            goal = _free_cell(rng, walls, {agent})
            a = int(rng.integers(0, N_ACTIONS))
            next_agent = step_pos(walls, agent, a)
            obs_buf[i, 0] = encode_obs(agent, walls, goal)
            obs_buf[i, 1] = encode_obs(next_agent, walls, goal)
            act_buf[i, 0] = a
            pos_buf[i, 0] = cell_to_idx(*agent)
            pos_buf[i, 1] = cell_to_idx(*next_agent)
        self.obs = obs_buf
        self.act = act_buf
        self.pos = pos_buf

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.pos[idx]),
        )
