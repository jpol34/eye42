"""Provenance-tracked conflict/repair log for noisy perception input, and for
real human irregularities during play (out-of-turn plays, revokes, misdeals,
wrong-seat trump calls, dealer-rotation mismatches, ...).

Corrections are logged as reversible proposals, never applied as silent in-place
overwrites — an edit made under a later-proven-wrong trump hypothesis must be
undoable, not permanent. ``Conflict``/``log_conflict`` cover tile-identity
conflicts (the original use case); ``Irregularity``/``log_irregularity`` cover
the broader "a human really did something the rules don't allow, and the engine
must keep going anyway" case. They're kept as separate shapes rather than one
merged hierarchy: ``Conflict`` is frozen and only ever describes a ``TilePlayed``
event, while an ``Irregularity`` needs to describe things a tile-played event
can't (a trump call, a deal, a dealer mismatch) and carry confirmation/confidence
metadata that would otherwise silently change what ``has_conflicts`` means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

from .events import TilePlayed


@dataclass(frozen=True)
class Conflict:
    event: TilePlayed
    reason: str


class IrregularityKind(Enum):
    OUT_OF_TURN = auto()
    TRICK_FORCE_CLOSED = auto()
    REVOKE = auto()
    SEAT_TILE_COUNT_ANOMALY = auto()
    MISDEAL_TILE_COUNT = auto()
    TRUMP_CALLED_BY_WRONG_SEAT = auto()
    PLAY_BEFORE_CONTRACT = auto()
    DEALER_ROTATION_MISMATCH = auto()
    BID_NOT_LEGAL = auto()  # a bid/pass BiddingRound rejected; dropped, hand continues
    TRICK_COUNT_EXCEEDED = auto()  # a play arriving after all 7 tricks are complete
    TRICK_CLOSED_EARLY = auto()  # a seat-already-played read force-started the next trick
    AMBIGUOUS_ATTRIBUTION = auto()  # a play recorded on a shaky guess at which seat played it


@dataclass(frozen=True)
class Irregularity:
    kind: IrregularityKind
    reason: str
    player: Optional[int] = None
    needs_confirmation: bool = False
    confidence: float = 1.0


@dataclass
class RepairLog:
    conflicts: List[Conflict] = field(default_factory=list)
    irregularities: List[Irregularity] = field(default_factory=list)
    sink: Optional[Callable[[object], None]] = None

    def log_conflict(self, event: TilePlayed, reason: str) -> None:
        conflict = Conflict(event=event, reason=reason)
        self.conflicts.append(conflict)
        if self.sink is not None:
            self.sink(conflict)

    def log_irregularity(self, irregularity: Irregularity) -> None:
        self.irregularities.append(irregularity)
        if self.sink is not None:
            self.sink(irregularity)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def open_questions(self) -> List[Irregularity]:
        """Irregularities that can't be auto-resolved and need a human answer
        via the post-hoc confirmation channel (never a live interruption)."""
        return [i for i in self.irregularities if i.needs_confirmation]
