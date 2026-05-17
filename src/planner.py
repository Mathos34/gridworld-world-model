"""BFS planner that runs entirely on the learned simulator.

Per planning step we precompute a 64x4 transition table T[cell, action] -> next_cell
by encoding 64 synthetic observations (one per possible agent position, with the
*current* walls and goal) and advancing each by each of the 4 actions. The next
agent cell is the argmax of the position head on the resulting latent. BFS over
this table finds the shortest predicted action sequence to the goal; we play its
first action and replan the next step (MPC).

If the BFS cannot reach the goal in the predicted graph (rare with a well-trained
model), we fall back to a Manhattan-greedy action.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .data import ACTIONS, GRID_SIZE, N_ACTIONS, N_CELLS, GridWorld, encode_obs
from .model import WorldModel


@torch.no_grad()
def build_transition_table(model: WorldModel, walls: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Returns T of shape (N_CELLS, N_ACTIONS) with T[c, a] = predicted next cell."""
    obs_batch = np.zeros((N_CELLS, 3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            obs_batch[r * GRID_SIZE + c] = encode_obs((r, c), walls, goal)
    obs_t = torch.from_numpy(obs_batch)
    z = model.encoder(obs_t)
    T = np.zeros((N_CELLS, N_ACTIONS), dtype=np.int64)
    for a in range(N_ACTIONS):
        a_t = torch.full((N_CELLS,), a, dtype=torch.long)
        z_next = model.predictor(z, a_t)
        T[:, a] = model.position_head(z_next).argmax(dim=-1).cpu().numpy()
    return T


def bfs_first_action(T: np.ndarray, start: int, goal: int) -> int | None:
    """BFS over the predicted dynamics graph. Returns the first action of a
    shortest path from start to goal, or None if unreachable."""
    if start == goal:
        return None
    parent_action = -np.ones(N_CELLS, dtype=np.int64)
    parent_state = -np.ones(N_CELLS, dtype=np.int64)
    visited = np.zeros(N_CELLS, dtype=bool)
    visited[start] = True
    q = deque([start])
    while q:
        s = q.popleft()
        for a in range(N_ACTIONS):
            ns = int(T[s, a])
            if visited[ns]:
                continue
            visited[ns] = True
            parent_state[ns] = s
            parent_action[ns] = a
            if ns == goal:
                # Reconstruct: walk back to start, return first action.
                cur = ns
                while parent_state[cur] != start:
                    cur = int(parent_state[cur])
                return int(parent_action[cur])
            q.append(ns)
    return None


def manhattan_greedy(walls: np.ndarray, agent: tuple[int, int], goal: tuple[int, int]) -> int:
    best_a, best_d = 0, 1e9
    for a, (dr, dc) in enumerate(ACTIONS):
        nr, nc = agent[0] + dr, agent[1] + dc
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            continue
        if walls[nr, nc] == 1:
            continue
        d = abs(nr - goal[0]) + abs(nc - goal[1])
        if d < best_d:
            best_d, best_a = d, a
    return best_a


@torch.no_grad()
def plan_action(model: WorldModel, agent: tuple[int, int], walls: np.ndarray,
                goal: tuple[int, int]) -> int:
    T = build_transition_table(model, walls, goal)
    s = agent[0] * GRID_SIZE + agent[1]
    g = goal[0] * GRID_SIZE + goal[1]
    a = bfs_first_action(T, s, g)
    if a is None:
        return manhattan_greedy(walls, agent, goal)
    return a


def evaluate_planner(model: WorldModel, n_episodes: int = 100, max_steps: int = 50,
                     seed: int = 1234) -> tuple[float, list[int]]:
    env = GridWorld(seed=seed)
    successes = []
    steps_taken = []
    for _ in range(n_episodes):
        env.reset()
        success = False
        for step in range(max_steps):
            a = plan_action(model, env.agent, env.walls, env.goal)
            _, done = env.step(a)
            if done:
                success = True
                steps_taken.append(step + 1)
                break
        successes.append(success)
        if not success:
            steps_taken.append(max_steps)
    return float(np.mean(successes)), steps_taken


def evaluate_random(n_episodes: int = 100, max_steps: int = 50, seed: int = 1234) -> float:
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


def evaluate_oracle_bfs(n_episodes: int = 100, max_steps: int = 50, seed: int = 1234) -> float:
    """Upper bound: BFS over the *true* dynamics. Useful to know how much
    reachable success rate is possible for these random initial states."""
    env = GridWorld(seed=seed)
    n_success = 0
    for _ in range(n_episodes):
        env.reset()
        # True transition: try the action; if blocked (wall or oob) state stays.
        for _ in range(max_steps):
            T = np.zeros((N_CELLS, N_ACTIONS), dtype=np.int64)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    s = r * GRID_SIZE + c
                    for a, (dr, dc) in enumerate(ACTIONS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or env.walls[nr, nc] == 1:
                            T[s, a] = s
                        else:
                            T[s, a] = nr * GRID_SIZE + nc
            s0 = env.agent[0] * GRID_SIZE + env.agent[1]
            g = env.goal[0] * GRID_SIZE + env.goal[1]
            a = bfs_first_action(T, s0, g)
            if a is None:
                break
            _, done = env.step(a)
            if done:
                n_success += 1
                break
    return n_success / n_episodes
