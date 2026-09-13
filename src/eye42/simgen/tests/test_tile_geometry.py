from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

from eye42.simgen.tile_geometry import TABLE_SIZE_M, TILE_HALF_EXTENTS_M
from eye42.simgen.trajectory import (
    _RACK_DROP_SPREAD_M,
    _SEAT_RACK_CENTER,
    _SEAT_WON_PILE_CENTER,
    _TRICK_AREA_CENTER,
    _TRICK_AREA_SPREAD_M,
    _WON_PILE_SPREAD_M,
)

_TILE_HALF_DIAGONAL_M = math.hypot(TILE_HALF_EXTENTS_M[0], TILE_HALF_EXTENTS_M[1])
_MIN_MARGIN_M = 0.05


def _worst_case_reach(center: tuple, spread: float) -> float:
    """A zone's farthest point from the table origin: its center plus the spread
    jitter's worst case on BOTH axes at once (trajectory.py's ``_jittered``/
    ``drop_and_settle`` apply spread independently per axis, not as one radial
    jitter), plus a tile's own half-diagonal so a tile's far corner, not just its
    center, is accounted for."""
    return math.hypot(abs(center[0]) + spread, abs(center[1]) + spread) + _TILE_HALF_DIAGONAL_M


def test_table_size_has_margin_beyond_every_placement_zones_worst_case_reach():
    half_table = min(TABLE_SIZE_M) / 2
    zone_reaches = (
        [_worst_case_reach(c, _RACK_DROP_SPREAD_M) for c in _SEAT_RACK_CENTER.values()]
        + [_worst_case_reach(c, _WON_PILE_SPREAD_M) for c in _SEAT_WON_PILE_CENTER.values()]
        + [_worst_case_reach(_TRICK_AREA_CENTER, _TRICK_AREA_SPREAD_M)]
    )

    worst_reach = max(zone_reaches)

    # The rack zone is the binding constraint today (largest center offset, 0.35m,
    # dwarfs the won-pile zone's 0.30m and the trick area's 0.0m) -- assert that
    # explicitly so a future edit to any zone can't silently make a DIFFERENT zone
    # the binding one without this test's margin check still covering it.
    assert worst_reach == max(_worst_case_reach(c, _RACK_DROP_SPREAD_M) for c in _SEAT_RACK_CENTER.values())
    assert half_table - worst_reach >= _MIN_MARGIN_M
