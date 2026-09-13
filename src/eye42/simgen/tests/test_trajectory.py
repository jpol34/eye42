from __future__ import annotations

import random

import pytest

pytest.importorskip("mujoco")

from eye42.engine.tiles import Tile
from eye42.simgen.director import DealIntent, PlayIntent, SweepIntent, script_one_hand
from eye42.simgen.trajectory import simulate_hand

_SMALL_HANDS = (
    (Tile.of(6, 6), Tile.of(5, 4)),
    (Tile.of(3, 3), Tile.of(2, 1)),
    (Tile.of(0, 0), Tile.of(6, 3)),
    (Tile.of(4, 2), Tile.of(1, 1)),
)


def _small_intents():
    """A minimal, fast, hand-written Intent sequence (not a full 7-trick hand) --
    exercises the same DealIntent -> PlayIntent -> SweepIntent shape director.py
    produces, without the ~24s cost of a real 28-tile/7-trick simulation."""
    return [
        DealIntent(hands=_SMALL_HANDS),
        PlayIntent(seat=0, tile=Tile.of(6, 6), trick_index=0),
        PlayIntent(seat=1, tile=Tile.of(3, 3), trick_index=0),
        PlayIntent(seat=2, tile=Tile.of(0, 0), trick_index=0),
        PlayIntent(seat=3, tile=Tile.of(4, 2), trick_index=0),
        SweepIntent(trick_index=0, winner=0),
    ]


def test_frame_count_matches_deal_plus_each_play_plus_each_sweep():
    intents = _small_intents()
    frames = simulate_hand(intents, seed=0)

    # 1 (initial deal) + 4 plays + 1 sweep = 6 frames
    assert len(frames) == 6


def test_a_played_tile_leaves_its_seats_rack():
    intents = _small_intents()
    frames = simulate_hand(intents, seed=0)

    assert Tile.of(6, 6) in frames[0].seat_racks[0]
    after_first_play = frames[1]
    assert Tile.of(6, 6) not in after_first_play.seat_racks[0]
    assert len(after_first_play.seat_racks[0]) == 1


def test_all_tiles_stay_accounted_for_across_every_frame():
    intents = _small_intents()
    frames = simulate_hand(intents, seed=0)
    all_tiles = {t for hand in _SMALL_HANDS for t in hand}

    for frame in frames:
        rack_tiles = {t for hand in frame.seat_racks.values() for t in hand}
        sim_tiles = {ts.tile for ts in frame.tile_states}
        assert sim_tiles == all_tiles, "physics must always track every tile, played or not"
        assert rack_tiles <= all_tiles


def test_same_seed_produces_the_same_trajectory():
    intents = _small_intents()
    frames_a = simulate_hand(intents, seed=1)
    frames_b = simulate_hand(intents, seed=1)

    for a, b in zip(frames_a, frames_b):
        for ts_a, ts_b in zip(a.tile_states, b.tile_states):
            assert ts_a.tile == ts_b.tile
            assert ts_a.position == pytest.approx(ts_b.position, abs=1e-6)


@pytest.mark.slow
def test_a_full_scripted_hand_simulates_end_to_end_without_error():
    """Integration test across director.py + trajectory.py + physics.py together: a
    real 7-trick, 28-tile hand, fully played out and physically simulated. Slow (~20s)
    because it's real physics over 7 tricks -- kept as a single seed/run, not a sweep."""
    intents = script_one_hand(random.Random(0))
    frames = simulate_hand(intents, seed=0)

    assert len(frames) == 1 + 28 + 7  # deal + one frame per play + one per sweep
    assert all(len(rack) == 0 for rack in frames[-1].seat_racks.values())
