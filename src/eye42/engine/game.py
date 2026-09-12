"""Game-level state: dealer rotation and marks-to-7 across hands.

A redeal does not advance the dealer or the hand index; a completed hand
(including a conceded one) does. Game is won at 7 marks (confirmed scoring mode —
splash/plunge only make sense in a marks game).

``observe_next_dealer`` implements the plan's own documented-but-previously-
missing retrospective cross-check: once the *next* hand's dealer is actually
observed, it either corroborates the previous hand's redeal/concession
classification or corrects it -- never raises, since an unexpected dealer just
means the table is ground truth and the engine's prior guess was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .hand import HandOutcomeKind, HandState
from .repair import Irregularity, IrregularityKind, RepairLog

MARKS_TO_WIN = 7


@dataclass
class HandLedgerEntry:
    """Snapshot of what was actually awarded for a hand at classification time,
    so a later correction can roll marks back exactly rather than recompute
    them from a hand whose ``outcome_kind`` has since changed."""

    hand: HandState
    dealer_at_deal: int
    outcome_kind: Optional[HandOutcomeKind]
    marks_awarded_to: Optional[int]
    marks: int


@dataclass
class GameState:
    starting_dealer: int = 0
    marks: List[int] = field(default_factory=lambda: [0, 0])
    hands_played: List[HandState] = field(default_factory=list, init=False)
    repairs: RepairLog = field(default_factory=RepairLog, init=False)
    _ledger: List[HandLedgerEntry] = field(default_factory=list, init=False)
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
        dealer_at_deal = self._dealer
        self.hands_played.append(hand)

        if hand.outcome_kind == HandOutcomeKind.REDEAL:
            self._ledger.append(HandLedgerEntry(
                hand=hand, dealer_at_deal=dealer_at_deal, outcome_kind=hand.outcome_kind,
                marks_awarded_to=None, marks=0,
            ))
            return  # same dealer deals again, hand index does not advance

        if hand.scorer is None:
            # Bidding never resolved to a contract -- shouldn't normally happen,
            # but game-level bookkeeping must not crash on a messy hand.
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.PLAY_BEFORE_CONTRACT,
                reason="hand ended with no resolved contract",
                needs_confirmation=True,
            ))
            self._ledger.append(HandLedgerEntry(
                hand=hand, dealer_at_deal=dealer_at_deal, outcome_kind=hand.outcome_kind,
                marks_awarded_to=None, marks=0,
            ))
            self._dealer = (self._dealer + 1) % 4
            return

        result = hand.final_result()
        awarded_to = result.marks_awarded_to if result.marks_awarded_to in (0, 1) else None
        if awarded_to is not None:
            self.marks[awarded_to] += result.marks
        self._ledger.append(HandLedgerEntry(
            hand=hand, dealer_at_deal=dealer_at_deal, outcome_kind=hand.outcome_kind,
            marks_awarded_to=awarded_to, marks=result.marks,
        ))
        self._dealer = (self._dealer + 1) % 4

    def observe_next_dealer(self, observed_dealer: int) -> Optional[str]:
        """Cross-check the previous hand's classification against who actually
        deals next. Returns a short tag describing what happened, or ``None`` if
        nothing needed correcting (including "no hand recorded yet")."""
        if not self._ledger:
            return None
        entry = self._ledger[-1]

        if observed_dealer == self._dealer:
            return None  # corroborated, nothing to correct

        expected_if_redeal = entry.dealer_at_deal
        expected_if_advanced = (entry.dealer_at_deal + 1) % 4

        if entry.outcome_kind == HandOutcomeKind.REDEAL and observed_dealer == expected_if_advanced:
            # Dealer actually rotated -- this was really a completed/conceded hand.
            entry.hand.outcome_kind = HandOutcomeKind.CONCESSION
            if entry.hand.scorer is None:
                # Bidding never resolved either -- nothing to score, but still
                # correct the classification and rotation rather than crash.
                entry.outcome_kind = HandOutcomeKind.CONCESSION
                entry.marks_awarded_to = None
                entry.marks = 0
            else:
                result = entry.hand.final_result()
                awarded_to = result.marks_awarded_to if result.marks_awarded_to in (0, 1) else None
                if awarded_to is not None:
                    self.marks[awarded_to] += result.marks
                entry.outcome_kind = HandOutcomeKind.CONCESSION
                entry.marks_awarded_to = awarded_to
                entry.marks = result.marks
            self._dealer = observed_dealer
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.DEALER_ROTATION_MISMATCH,
                reason="dealer rotated after a hand classified REDEAL -- reclassified CONCESSION",
                needs_confirmation=True,
            ))
            return "reclassified_concession"

        if entry.outcome_kind != HandOutcomeKind.REDEAL and observed_dealer == expected_if_redeal:
            # Dealer did NOT rotate -- this was really a redeal.
            if entry.marks_awarded_to is not None:
                self.marks[entry.marks_awarded_to] -= entry.marks
            entry.hand.outcome_kind = HandOutcomeKind.REDEAL
            entry.outcome_kind = HandOutcomeKind.REDEAL
            entry.marks_awarded_to = None
            entry.marks = 0
            self._dealer = expected_if_redeal
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.DEALER_ROTATION_MISMATCH,
                reason="dealer did not rotate after a hand classified CONCESSION -- reclassified REDEAL",
                needs_confirmation=True,
            ))
            return "reclassified_redeal"

        # Matches neither expectation -- an entire hand may have gone
        # unobserved. The table is ground truth; adopt it and flag for review.
        self._dealer = observed_dealer
        self.repairs.log_irregularity(Irregularity(
            kind=IrregularityKind.DEALER_ROTATION_MISMATCH,
            reason=f"observed dealer {observed_dealer} matches neither expected rotation",
            needs_confirmation=True,
        ))
        return "adopted_unexpected_dealer"

    def open_questions(self) -> List[Irregularity]:
        """Aggregated feed for the Phase 4 post-hoc confirmation UI -- every
        unresolved irregularity across the game and every hand played so far."""
        questions = list(self.repairs.open_questions)
        for hand in self.hands_played:
            questions.extend(hand.repairs.open_questions)
        return questions
