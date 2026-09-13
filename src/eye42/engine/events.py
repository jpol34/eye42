"""Abstract input events the engine consumes.

These are the "interface" between perception/speech (Phases 2-3, not built yet) and
the rules engine (Phase 1). Keeping them plain dataclasses means the engine can be
built and tested entirely with hand-typed event sequences, independent of any CV or
speech code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .tiles import Tile


@dataclass(frozen=True)
class BidMade:
    player: int
    amount: int  # points (30-41), or 42/marks-equivalent — see bidding.BidType
    marks: int = 0  # 0 = a point bid; >=1 = an explicit marks bid


@dataclass(frozen=True)
class Passed:
    player: int


@dataclass(frozen=True)
class TrumpCalled:
    """An explicit, unambiguous trump declaration (e.g. parsed from clear speech)."""

    caller: int
    trump: int


@dataclass(frozen=True)
class TrumpCueHeard:
    """A weak verbal cue ('low end' / 'high end') accompanying a lead — not a full
    declaration on its own; only resolves trump combined with the led tile."""

    speaker: int
    cue: str  # "low" or "high"


@dataclass(frozen=True)
class TilePlayed:
    player: int
    tile: Tile
    confidence: float = 1.0  # CV classifier confidence in this tile's identity, 0-1
    # Separate from `confidence`: how sure perception is that `player` is the seat
    # that actually played it, not just what the tile is -- a settled tile's resting
    # position on a shared table doesn't by itself reveal who played it.
    player_confidence: float = 1.0


@dataclass(frozen=True)
class IrregularEndSignal:
    """Perception observed mass/irregular tile motion (concession or misdeal-shaped
    behavior) rather than normal one-tile-per-turn play."""

    tricks_played_so_far: int
    # Optional richer misdeal signal (e.g. a seat visibly holding the wrong tile
    # count): Phase 2 perception doesn't exist yet, so this is defaulted and
    # unused until it does.
    tiles_visible_by_seat: Optional[Dict[int, int]] = None


@dataclass(frozen=True)
class TilesDealt:
    """The deal itself, when perception can observe it (Phase 2) or a hand
    transcript supplies it directly. The missing deal-event representation --
    this is what lets a misdeal (wrong tile count, an exposed tile) be modeled
    at all, rather than only inferred after the fact."""

    dealer: int
    counts: Dict[int, int]  # seat -> tiles dealt
    exposed_tiles: Tuple[Tile, ...] = ()
