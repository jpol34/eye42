"""Single-trick play: legality and winner determination.

A played tile is treated as ground truth (see engine.hand): ``play()`` supports
a ``strict`` mode (the default, used for direct/API-level use and by
``engine.probability``'s playout policy, where an illegal play means a bug in
the caller) and a non-strict mode (used by ``HandState`` for real event
sequences) that never raises -- it annotates violations instead of rejecting
the play, so a genuine human mistake degrades gracefully rather than crashing
whatever is processing the hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .tiles import Tile, effective_suit, follows_suit, trick_rank


class IllegalPlayError(ValueError):
    pass


class PlayViolation(Enum):
    OUT_OF_TURN = auto()
    REVOKE = auto()
    SEAT_ALREADY_PLAYED = auto()
    TRICK_FULL = auto()


@dataclass
class Trick:
    leader: int
    trump: int
    plays: List[Tuple[int, Tile]] = field(default_factory=list)
    _led_suit: Optional[int] = field(default=None, init=False)
    entitled_leader: Optional[int] = field(default=None, init=False)
    violations: Dict[int, Set[PlayViolation]] = field(default_factory=dict, init=False)
    superseded_plays: List[Tuple[int, Tile]] = field(default_factory=list, init=False)
    force_closed: bool = field(default=False, init=False)
    closed_early_for_new_trick: bool = field(default=False, init=False)
    """Closed by ``HandState`` because the next trick demonstrably started: ≥3
    distinct seats were already in and the seat entitled to lead next led it.

    Deliberately a *separate* flag from ``force_closed``, not a reuse of it.
    ``force_closed`` means "we gave up waiting, tiles are probably missing" and
    suppresses the probability estimator when it fires short; this one means
    "resolved, we simply saw the boundary" and must not suppress anything. One
    flag for both would blank the equity display on an ordinary next-trick lead.
    """

    @property
    def led_suit(self) -> Optional[int]:
        return self._led_suit

    @property
    def seats_played(self) -> Set[int]:
        return {p for p, _ in self.plays}

    @property
    def is_complete(self) -> bool:
        return len(self.seats_played) == 4 or self.force_closed or self.closed_early_for_new_trick

    @property
    def next_player(self) -> Optional[int]:
        """First seat, walking the rotation from ``leader``, that hasn't played
        yet -- this (rather than a fixed ``leader + len(plays)``) is what lets a
        seat that gets skipped by an out-of-turn play recover on its own turn
        instead of the rotation staying permanently off by one."""
        if self.is_complete:
            return None
        played = self.seats_played
        for offset in range(4):
            seat = (self.leader + offset) % 4
            if seat not in played:
                return seat
        return None

    def legal_plays(self, hand: Sequence[Tile]) -> List[Tile]:
        """Which tiles in ``hand`` are legal to play right now."""
        if self._led_suit is None:
            return list(hand)  # leading: anything is legal
        followers = [t for t in hand if follows_suit(t, self.trump, self._led_suit)]
        return followers if followers else list(hand)

    def check_play(
        self, player: int, tile: Tile, hand: Optional[Sequence[Tile]] = None
    ) -> Set[PlayViolation]:
        """Pure predicate: what (if anything) is wrong with this play, without
        mutating any state."""
        violations: Set[PlayViolation] = set()
        if self.is_complete:
            violations.add(PlayViolation.TRICK_FULL)
            return violations
        if player in self.seats_played:
            violations.add(PlayViolation.SEAT_ALREADY_PLAYED)
            return violations

        expected = self.next_player
        if expected is not None and player != expected:
            violations.add(PlayViolation.OUT_OF_TURN)
        if self._led_suit is not None and hand is not None and tile not in self.legal_plays(hand):
            violations.add(PlayViolation.REVOKE)
        return violations

    def play(
        self,
        player: int,
        tile: Tile,
        hand: Optional[Sequence[Tile]] = None,
        *,
        strict: bool = True,
    ) -> Set[PlayViolation]:
        violations = self.check_play(player, tile, hand)

        if strict:
            if PlayViolation.TRICK_FULL in violations:
                raise IllegalPlayError("trick already has 4 plays")
            if PlayViolation.SEAT_ALREADY_PLAYED in violations:
                raise IllegalPlayError(f"seat {player} already played this trick")
            if PlayViolation.OUT_OF_TURN in violations:
                raise IllegalPlayError(f"expected seat {self.next_player} to play next, got {player}")
            if PlayViolation.REVOKE in violations:
                raise IllegalPlayError(
                    f"seat {player} must follow suit {self._led_suit} if able; "
                    f"played {tile} instead"
                )
            self._record_play(player, tile)
            return violations

        # Non-strict: the tile on the table is ground truth. Never raise --
        # annotate and keep going.
        if PlayViolation.TRICK_FULL in violations or PlayViolation.SEAT_ALREADY_PLAYED in violations:
            self.superseded_plays.append((player, tile))
        else:
            if PlayViolation.OUT_OF_TURN in violations and not self.plays:
                # Out-of-turn lead: the tile physically on the table IS the lead.
                self.entitled_leader = self.leader
                self.leader = player

            if violations:
                self.violations.setdefault(player, set()).update(violations)
            self._record_play(player, tile)

        # Bounded force-close: without this, a duplicate/extra play combined
        # with an occluded/missed seat could leave a trick open forever,
        # silently swallowing every later play in the hand into
        # superseded_plays. The threshold is deliberately above 4 (the normal
        # count) so an ordinary duplicate-then-the-real-4th-seat sequence still
        # completes naturally rather than getting force-closed prematurely.
        # Checked here too (not just after a real play) since a run of nothing
        # but rejected duplicates would otherwise never trip it at all.
        FORCE_CLOSE_AFTER = 6
        if len(self.plays) + len(self.superseded_plays) >= FORCE_CLOSE_AFTER and not self.is_complete:
            self.force_closed = True

        return violations

    def _record_play(self, player: int, tile: Tile) -> None:
        if self._led_suit is None:
            self._led_suit = effective_suit(tile, self.trump, None)
        self.plays.append((player, tile))

    @property
    def winner(self) -> Optional[int]:
        if not self.plays:
            return None
        assert self._led_suit is not None
        best_player, best_tile = self.plays[0]
        best_rank = trick_rank(best_tile, self.trump, self._led_suit)
        for player, tile in self.plays[1:]:
            rank = trick_rank(tile, self.trump, self._led_suit)
            if rank > best_rank:
                best_player, best_tile, best_rank = player, tile, rank
        return best_player

    @property
    def count_value(self) -> int:
        return sum(t.count_value for _, t in self.plays)
