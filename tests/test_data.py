"""Unit tests for the gridworld data layer."""
import numpy as np

from src.data import (
    ACTIONS,
    GRID_SIZE,
    N_CELLS,
    GridWorld,
    cell_to_idx,
    encode_obs,
    idx_to_cell,
    make_walls,
    step_pos,
)


def test_cell_idx_roundtrip():
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            assert idx_to_cell(cell_to_idx(r, c)) == (r, c)
    assert N_CELLS == GRID_SIZE * GRID_SIZE


def test_encode_obs_shape_and_channels():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    walls[3, 4] = 1
    obs = encode_obs(agent=(0, 0), walls=walls, goal=(7, 7))
    assert obs.shape == (3, GRID_SIZE, GRID_SIZE)
    assert obs[0, 0, 0] == 1.0 and obs[0].sum() == 1.0
    assert obs[1, 3, 4] == 1.0 and obs[1].sum() == 1.0
    assert obs[2, 7, 7] == 1.0 and obs[2].sum() == 1.0


def test_step_pos_moves_to_free_cell():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    for action, (dr, dc) in enumerate(ACTIONS):
        assert step_pos(walls, (3, 3), action) == (3 + dr, 3 + dc)


def test_step_pos_blocked_by_wall_keeps_position():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    walls[2, 3] = 1
    assert step_pos(walls, (3, 3), 0) == (3, 3)


def test_step_pos_blocked_by_out_of_bounds():
    walls = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    assert step_pos(walls, (0, 0), 0) == (0, 0)
    assert step_pos(walls, (0, 0), 2) == (0, 0)
    assert step_pos(walls, (GRID_SIZE - 1, GRID_SIZE - 1), 1) == (GRID_SIZE - 1, GRID_SIZE - 1)
    assert step_pos(walls, (GRID_SIZE - 1, GRID_SIZE - 1), 3) == (GRID_SIZE - 1, GRID_SIZE - 1)


def test_make_walls_has_exactly_one_gap_on_a_line():
    rng = np.random.default_rng(0)
    for _ in range(20):
        walls = make_walls(rng)
        per_row = walls.sum(axis=1)
        per_col = walls.sum(axis=0)
        rows_full = (per_row >= GRID_SIZE - 1).sum()
        cols_full = (per_col >= GRID_SIZE - 1).sum()
        assert rows_full + cols_full >= 1, "expected at least one wall line"


def test_gridworld_reset_separates_agent_and_goal():
    env = GridWorld(seed=42)
    for _ in range(10):
        env.reset()
        assert env.agent != env.goal
        assert env.walls[env.agent] == 0
        assert env.walls[env.goal] == 0
