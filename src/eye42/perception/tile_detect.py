"""Phase 2 (not implemented yet — depends on real recorded footage to design and
validate against). Interface stubs matching the project plan so real detection
logic has a slot to drop into.

Pipeline shape (see plan for full rationale):
  homography rectification -> tile localization -> 28-class identity classifier
  -> settle-time-debounced event segmentation -> trick-sweep / hand-boundary
  detection, feeding eye42.engine.events.TilePlayed / IrregularEndSignal into a
  HandState.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from eye42.engine.tiles import Tile


@dataclass(frozen=True)
class TileObservation:
    """A single detected tile on the rectified table, before it's decided to be
    a genuine "play" (see the settle-time debounce in EventSegmenter)."""

    tile: Tile
    confidence: float
    position: Tuple[float, float]  # rectified-plane coordinates
    # No shared clock convention with speech.bid_parser (which timestamps in
    # start_time/end_time seconds) exists yet -- ordering a tile play against a
    # spoken bid across modalities isn't possible until one is defined.
    frame_index: int


class TableRectifier:
    """Wraps the one-time 4-corner homography computed for a recording session.
    See tools/crop_rectify.py for the throwaway version used to validate a
    capture setup before committing to real sessions.
    """

    def __init__(self, corners: Sequence[Tuple[float, float]], output_size: Tuple[int, int]) -> None:
        raise NotImplementedError("Phase 2 — needs real footage to calibrate against")

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TileLocalizer:
    """Finds candidate tile bounding boxes/orientations on a rectified frame."""

    def find_tiles(self, rectified_frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Returns (x, y, width, height) boxes for candidate tile regions."""
        raise NotImplementedError("Phase 2 — classical contour detection first, per the plan")


class TileIdentityClassifier(Protocol):
    """A per-tile-half or whole-tile classifier over the 28 known tile classes.

    Per the plan: perspective-corrected crop -> small template-match or
    lightweight CNN, NOT Hough-circle pip counting (known failure modes: the
    center spinner pin reads as a false pip, blanks are indistinguishable from
    a failed detection).
    """

    def classify(self, tile_crop: np.ndarray) -> List[Tuple[Tile, float]]:
        """Returns candidate tile identities ranked by confidence (best guess
        first, score 0-1 each), not just a single committed identity -- the
        same "track weighted candidates, don't commit to one guess" convention
        engine.trump_inference.TrumpHypothesisTracker uses for ambiguity."""
        ...


class EventSegmenter:
    """Turns a stream of per-frame TileObservations into discrete play events,
    using a settle-time debounce (a tile only counts as "played" after N stable
    frames — same pattern as DGT chessboard debouncing) and explicit trick-sweep
    handling: if all 4 tiles of a trick aren't logged before the winner's sweep
    clears the area, this degrades gracefully rather than raising or silently
    under-counting -- the same force-closed pattern engine.trick.Trick and
    engine.hand.HandState use (see IrregularityKind.TRICK_FORCE_CLOSED), so a
    missed sweep-tile is flagged as a real-event irregularity, not a crash.
    """

    def __init__(self, settle_frames: int = 15) -> None:
        raise NotImplementedError("Phase 2 — needs real footage to tune settle_frames against")

    def feed(self, observations: List[TileObservation]) -> Optional[Tile]:
        """Feed one frame's worth of observations; returns a confirmed play if
        one just settled, else None."""
        raise NotImplementedError


class HandBoundaryDetector:
    """Detects shuffle/reshuffle (new hand started) via large-scale, sustained
    tile motion across the whole table, distinct from in-trick motion."""

    def is_new_hand_starting(self, motion_field: np.ndarray) -> bool:
        raise NotImplementedError("Phase 2 — threshold needs tuning against real footage")
