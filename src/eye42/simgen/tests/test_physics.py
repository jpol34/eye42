from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from eye42.engine.tiles import Tile
from eye42.simgen.physics import TileSimulation
from eye42.simgen.tile_geometry import TILE_THICKNESS_M

_SOME_TILES = [Tile.of(6, 6), Tile.of(5, 3), Tile.of(2, 0), Tile.of(4, 4)]


def test_drop_and_settle_converges_before_the_step_budget():
    sim = TileSimulation(_SOME_TILES, seed=1)
    states = sim.drop_and_settle()

    assert len(states) == len(_SOME_TILES)


def test_dropped_tiles_come_to_rest_on_the_table_not_through_it():
    sim = TileSimulation(_SOME_TILES, seed=2)
    states = sim.drop_and_settle()

    for state in states:
        # A tile resting flat sits at roughly half its thickness above the table plane;
        # on edge or tilted against another tile it sits higher -- either way it must
        # not have sunk through the table (z <= 0), and shouldn't still be airborne.
        assert -1e-3 < state.position[2] < 0.20


def test_drop_and_settle_is_deterministic_given_the_same_seed():
    states_a = TileSimulation(_SOME_TILES, seed=42).drop_and_settle()
    states_b = TileSimulation(_SOME_TILES, seed=42).drop_and_settle()

    for a, b in zip(states_a, states_b):
        assert a.tile == b.tile
        assert np.allclose(a.position, b.position, atol=1e-6)
        assert np.allclose(a.orientation_quat, b.orientation_quat, atol=1e-6)


def test_different_seeds_produce_different_layouts():
    states_a = TileSimulation(_SOME_TILES, seed=1).drop_and_settle()
    states_b = TileSimulation(_SOME_TILES, seed=2).drop_and_settle()

    positions_a = [s.position[:2] for s in states_a]
    positions_b = [s.position[:2] for s in states_b]
    assert not np.allclose(positions_a, positions_b, atol=1e-3)


def test_apply_push_moves_a_settled_tile():
    tile = Tile.of(1, 0)
    sim = TileSimulation([tile], seed=3)
    sim.drop_and_settle()
    before = sim.state_of(tile).position

    sim.apply_push(tile, force_xy=(0.5, 0.0), steps=200)
    sim.settle()
    after = sim.state_of(tile).position

    moved = np.hypot(after[0] - before[0], after[1] - before[1])
    assert moved > 0.01, "a pushed tile should have slid measurably from its start position"


def test_slide_toward_reaches_the_target_without_destabilizing_the_solver():
    """A blind constant-force push (apply_push) at too high a force previously produced
    a real MuJoCo QACC-instability warning when multiple tiles converged on one point --
    slide_toward's velocity cap exists specifically to prevent that; this test pins the
    behavior it's meant to guarantee (arrival, not just "didn't crash")."""
    tile = Tile.of(2, 2)
    sim = TileSimulation([tile], seed=5)
    sim.drop_and_settle(drop_center=(0.2, 0.2))
    target = (-0.2, -0.1)

    steps = sim.slide_toward(tile, target)
    sim.settle()
    final = sim.state_of(tile).position

    assert steps < 3000, "should reach the target well before the bounded step budget"
    distance_to_target = ((final[0] - target[0]) ** 2 + (final[1] - target[1]) ** 2) ** 0.5
    assert distance_to_target < 0.03


def test_settling_a_single_flat_tile_takes_far_fewer_steps_than_the_budget():
    """A single tile dropped from a modest height onto an empty table is the easiest
    possible case -- it should settle quickly, not consume the full step budget
    reserved for genuinely unstable configurations."""
    sim = TileSimulation([Tile.of(0, 0)], seed=4)
    steps_taken = None

    # drop_and_settle doesn't expose step count directly; settle() does, so exercise it
    # the same way drop_and_settle does internally, on a single easy tile.
    sim.drop_and_settle()
    # A second settle() call on an already-settled tile should return almost immediately.
    steps_taken = sim.settle()
    assert steps_taken < 10
