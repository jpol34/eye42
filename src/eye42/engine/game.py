"""Game-level state: dealer rotation and marks-to-7 across hands.

A redeal does not advance the dealer or the hand index; a completed hand
(including a conceded one) does. Game is won at 7 marks (confirmed scoring mode —
splash/plunge only make sense in a marks game).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .hand import HandOutcomeKind, HandState

MARKS_TO_WIN = 7


@dataclass
class GameState:
    starting_dealer: int = 0
    marks: List[int] = field(default_factory=lambda: [0, 0])
    hands_played: List[HandState] = field(default_factory=list, init=False)
    _dealer: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._dealer = self.starting_dealer

    @property
    def dealer(self) -> int:
        return self._dealer

    @property
    def winner(self) -> Optional[int]:
        for team, m in enumerate(self.marks):
            if m >= MARKS_TO_WIN:
                return team
        return None

    def new_hand(self) -> HandState:
        return HandState(dealer=self._dealer)

    def record_hand(self, hand: HandState) -> None:
        self.hands_played.append(hand)
        if hand.outcome_kind == HandOutcomeKind.REDEAL:
            return  # same dealer deals again, hand index does not advance
        result = hand.final_result()
        if result.marks_awarded_to in (0, 1):
            self.marks[result.marks_awarded_to] += result.marks
        self._dealer = (self._dealer + 1) % 4
