"""Drives a director-scripted hand's Intents through physics, producing a sequence of
settled FrameSnapshots -- the record render.py (or a ground-truth consumer that doesn't
need pixels at all) needs, decoupled from both the game-logic layer above and the
rendering layer below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from eye42.engine.tiles import Tile
from eye42.simgen.director import DealIntent, Intent, PlayIntent, SweepIntent
from eye42.simgen.physics import TileSimulation, TileState

# Placeholder table-space layout (meters, table-centered) pending a real camera
# calibration -- see tile_geometry.py / RESEARCH.md's "calibration.json is a homography,
# not a camera calibration" finding. Good enough to produce plausible, non-overlapping
# rack/pile/trick zones for Phase 0.
_SEAT_RACK_CENTER: Dict[int, Tuple[float, float]] = {
    0: (0.0, 0.35),
    1: (0.35, 0.0),
    2: (0.0, -0.35),
    3: (-0.35, 0.0),
}
_SEAT_WON_PILE_CENTER: Dict[int, Tuple[float, float]] = {
    0: (0.12, 0.30),
    1: (0.30, -0.12),
    2: (-0.12, -0.30),
    3: (-0.30, 0.12),
}
_TRICK_AREA_CENTER = (0.0, 0.0)
# A real trick's 4 tiles land scattered, not stacked at one point (see RESEARCH.md's
# footage review) -- jittering each play's target is both more realistic AND load-
# bearing for physics stability: pushing every tile at the exact same point causes
# tiles to collide and interpenetrate deeply, which is what produced a real QACC
# instability warning during development.
_TRICK_AREA_SPREAD_M = 0.06
_WON_PILE_SPREAD_M = 0.04
_RACK_DROP_SPREAD_M = 0.08


@dataclass(frozen=True)
class FrameSnapshot:
    """One settled moment's full scene state (a deal, a play, or a sweep) -- not a
    dense frame-by-frame trajectory; a renderer or validator can sample/interpolate
    between snapshots as needed."""

    index: int
    tile_states: Tuple[TileState, ...]
    seat_racks: Dict[int, Tuple[Tile, ...]]  # remaining un-played tiles per seat


def simulate_hand(intents: Sequence[Intent], seed: int) -> List[FrameSnapshot]:
    """Physically executes a director-scripted hand's Intents (see director.py) against
    a TileSimulation: deals tiles into each seat's rack, then pushes tiles from rack to
    the central trick area for each play and from the trick area to the winner's pile
    for each sweep -- letting physics own the resulting motion/settling rather than
    teleporting tiles between poses."""
    deal = intents[0]
    assert isinstance(deal, DealIntent)
    all_tiles = [tile for hand in deal.hands for tile in hand]
    sim = TileSimulation(all_tiles, seed=seed)
    jitter = random.Random(seed)

    racks: Dict[int, List[Tile]] = {seat: list(hand) for seat, hand in enumerate(deal.hands)}
    for seat, hand in racks.items():
        sim.drop_and_settle(hand, drop_center=_SEAT_RACK_CENTER[seat], spread=_RACK_DROP_SPREAD_M)

    frames: List[FrameSnapshot] = [
        FrameSnapshot(index=0, tile_states=tuple(sim.states()), seat_racks={s: tuple(r) for s, r in racks.items()})
    ]

    def _jittered(center: Tuple[float, float], spread: float) -> Tuple[float, float]:
        return (center[0] + jitter.uniform(-spread, spread), center[1] + jitter.uniform(-spread, spread))

    played_by_trick: Dict[int, List[Tile]] = {}
    for intent in intents[1:]:
        if isinstance(intent, PlayIntent):
            racks[intent.seat].remove(intent.tile)
            played_by_trick.setdefault(intent.trick_index, []).append(intent.tile)
            sim.slide_toward(intent.tile, _jittered(_TRICK_AREA_CENTER, _TRICK_AREA_SPREAD_M))
            sim.settle()
        elif isinstance(intent, SweepIntent):
            for tile in played_by_trick[intent.trick_index]:
                sim.slide_toward(tile, _jittered(_SEAT_WON_PILE_CENTER[intent.winner], _WON_PILE_SPREAD_M))
            sim.settle()
        else:  # DealIntent already handled above
            continue
        frames.append(
            FrameSnapshot(index=len(frames), tile_states=tuple(sim.states()), seat_racks={s: tuple(r) for s, r in racks.items()})
        )

    return frames
