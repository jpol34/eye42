"""Count-point tallying and mathematical set detection.

Set detection is a pure arithmetic fact about the running score, evaluated after
each *completed* trick (the in-progress trick counts as "remaining" until it
finishes — documented here to avoid an off-by-one). It is deliberately NOT the
concession trigger: real concessions happen on strategic certainty, well before
the arithmetic locks (see the perception layer, Phase 2 — not built yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .bidding import Contract, team_of
from .tiles import full_set
from .trick import Trick

TOTAL_COUNT_POINTS = sum(t.count_value for t in full_set())  # 35
TRICKS_PER_HAND = 7
TOTAL_HAND_POINTS = TOTAL_COUNT_POINTS + TRICKS_PER_HAND  # 42


@dataclass
class HandScoreTracker:
    contract: Contract
    completed_tricks: List[Trick] = field(default_factory=list)

    @property
    def bidding_team(self) -> int:
        return team_of(self.contract.bidder)

    def _team_points(self, team: int) -> int:
        points = 0
        for trick in self.completed_tricks:
            if team_of(trick.winner) == team:  # type: ignore[arg-type]
                points += 1 + trick.count_value
        return points

    @property
    def bidding_team_points(self) -> int:
        return self._team_points(self.bidding_team)

    @property
    def defending_team_points(self) -> int:
        return self._team_points(1 - self.bidding_team)

    @property
    def points_claimed_total(self) -> int:
        return self.bidding_team_points + self.defending_team_points

    @property
    def is_locked_set(self) -> bool:
        """True once it's mathematically impossible for the bidding team to make
        their contract, regardless of how remaining tricks go."""
        if self.contract.requires_sweep:
            # Any lost trick already breaks a sweep-required contract.
            return any(
                team_of(t.winner) != self.bidding_team  # type: ignore[arg-type]
                for t in self.completed_tricks
            )
        remaining_points = TOTAL_HAND_POINTS - self.points_claimed_total
        shortfall = self.contract.points_needed - self.bidding_team_points
        return shortfall > remaining_points

    def record_trick(self, trick: Trick) -> None:
        if not trick.is_complete:
            raise ValueError("cannot record an incomplete trick")
        self.completed_tricks.append(trick)

    def final_result(self) -> "HandResult":
        made = (
            not self.is_locked_set
            and len(self.completed_tricks) == TRICKS_PER_HAND
            and self.bidding_team_points >= self.contract.points_needed
        )
        marks = self.contract.marks_at_stake
        return HandResult(
            bidding_team=self.bidding_team,
            made=made,
            marks_awarded_to=self.bidding_team if made else (1 - self.bidding_team),
            marks=marks,
            bidding_team_points=self.bidding_team_points,
        )


@dataclass(frozen=True)
class HandResult:
    bidding_team: int
    made: bool
    marks_awarded_to: int
    marks: int
    bidding_team_points: int
