"""Unit tests for the BFS planner over a learned transition table."""
import numpy as np

from src.data import GRID_SIZE, N_ACTIONS, N_CELLS
from src.planner import bfs_first_action, manhattan_greedy


def _identity_table() -> np.ndarray:
    """Transition table where every action keeps the agent in place."""
    T = np.zeros((N_CELLS, N_ACTIONS), dtype=np.int64)
    for s in range(N_CELLS):
        T[s, :] = s
    return T


def test_bfs_returns_none_when_start_equals_goal():
    T = _identity_table()
    assert bfs_first_action(T, start=10, goal=10) is None


def test_bfs_returns_none_when_unreachable():
    T = _identity_table()
    assert bfs_first_action(T, start=0, goal=63) is None


def test_bfs_finds_one_step_neighbor():
    T = _identity_table()
    T[0, 3] = 1  # action 3 (right) from cell 0 moves to cell 1
    assert bfs_first_action(T, start=0, goal=1) == 3


def test_bfs_finds_two_step_path_first_action():
    T = _identity_table()
    T[0, 3] = 1
    T[1, 3] = 2
    assert bfs_first_action(T, start=0, goal=2) == 3


def test_bfs_picks_shortest_first_action_among_ties():
    T = _identity_table()
    T[0, 3] = 1
    T[1, 3] = 5
    T[0, 1] = 8
    T[8, 3] = 5
    first = bfs_first_action(T, start=0, goal=5)
    assert first in (3, 1)


def test_manhattan_greedy_moves_toward_goal_on_open_board():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    assert manhattan_greedy(walls, agent=(0, 0), goal=(7, 7)) in (1, 3)
    assert manhattan_greedy(walls, agent=(7, 7), goal=(0, 0)) in (0, 2)


def test_manhattan_greedy_avoids_walls():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    walls[0, 1] = 1
    a = manhattan_greedy(walls, agent=(0, 0), goal=(0, 7))
    assert a != 3
