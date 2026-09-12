"""Orchestrates one hand: bidding -> trump resolution -> trick play -> scoring,
including the set/redeal/concession edge cases.

Tile identity, trump, and trick-winner/turn-order are treated as jointly resolved
per the plan: ``TrumpHypothesisTracker`` narrows trump using observed (trusted)
play, and full-hand knowledge (when available, e.g. in tests or once perception
can read a deal) lets ``Trick.play`` enforce follow-suit legality directly. Repairs
to noisy tile reads are logged with provenance rather than applied in place, so a
later-proven-wrong correction can be rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from .bidding import BiddingRound, BidKind, Contract, team_of
from .events import BidMade, IrregularEndSignal, Passed, TilePlayed, TrumpCueHeard
from .repair import RepairLog
from .scoring import HandResult, HandScoreTracker
from .tiles import Tile, effective_suit
from .trick import Trick
from .trump_inference import TrumpHypothesisTracker


class HandOutcomeKind(Enum):
    NORMAL = auto()  # played to completion (or locked-set) with no irregular end
    CONCESSION = auto()  # ended early, players agreed the outcome was determined
    REDEAL = auto()  # thrown out (misdeal / all-pass handled separately as forced-30)


class HandError(ValueError):
    pass


@dataclass
class HandState:
    dealer: int
    hands: Optional[Dict[int, List[Tile]]] = None  # full deal, when known (e.g. tests)
    bidding: BiddingRound = field(init=False)
    trump_tracker: TrumpHypothesisTracker = field(default_factory=TrumpHypothesisTracker, init=False)
    repairs: RepairLog = field(default_factory=RepairLog, init=False)
    tricks: List[Trick] = field(default_factory=list, init=False)
    _current_trick: Optional[Trick] = field(default=None, init=False)
    scorer: Optional[HandScoreTracker] = field(default=None, init=False)
    outcome_kind: Optional[HandOutcomeKind] = field(default=None, init=False)
    _seen_tiles: Dict[Tile, int] = field(default_factory=dict, init=False)  # tile -> player
    _pending_cue: Optional[str] = field(default=None, init=False)
    voids: Dict[int, set] = field(default_factory=lambda: {p: set() for p in range(4)}, init=False)

    def __post_init__(self) -> None:
        self.bidding = BiddingRound(dealer=self.dealer)

    # ---- bidding -----------------------------------------------------

    def bid(self, player: int, amount: int = 0, marks: int = 0) -> None:
        self.bidding.record_bid(BidMade(player=player, amount=amount, marks=marks))
        if self.bidding.is_done and self.bidding.contract.kind == BidKind.MARKS:  # type: ignore[union-attr]
            self._maybe_upgrade_marks_bid()
        if self.bidding.is_done:
            self._start_score_tracker()

    def bid_pass(self, player: int) -> None:
        self.bidding.record_pass(Passed(player=player))
        if self.bidding.is_done:
            self._start_score_tracker()

    def _maybe_upgrade_marks_bid(self) -> None:
        """A plain marks bid is reclassified as splash/plunge once the bidder's
        doubles are known (only possible when we have the full deal)."""
        assert self.bidding.contract is not None
        bidder = self.bidding.contract.bidder
        if self.hands is None or bidder not in self.hands:
            return
        doubles = sum(1 for t in self.hands[bidder] if t.is_double)
        if doubles >= 4:
            self.bidding.upgrade_to_splash_or_plunge(BidKind.PLUNGE)
        elif doubles >= 3:
            self.bidding.upgrade_to_splash_or_plunge(BidKind.SPLASH)

    def _start_score_tracker(self) -> None:
        assert self.bidding.contract is not None
        self.scorer = HandScoreTracker(contract=self.bidding.contract)

    @property
    def contract(self) -> Optional[Contract]:
        return self.bidding.contract

    # ---- trump ---------------------------------------------------------

    def call_trump(self, caller: int, trump: int) -> None:
        assert self.contract is not None
        if caller != self.contract.trump_caller:
            raise HandError(
                f"seat {caller} is not the trump caller for this contract "
                f"(expected seat {self.contract.trump_caller})"
            )
        self.trump_tracker.confirm(trump)

    def hear_trump_cue(self, cue: TrumpCueHeard) -> None:
        # Weak cues only resolve anything once combined with the led tile; stash
        # it and let the first tile-play of the hand consume it.
        self._pending_cue = cue.cue

    # ---- trick play ------------------------------------------------------

    def _ensure_trick_started(self) -> Trick:
        assert self.contract is not None
        if self._current_trick is None:
            leader = self.contract.trump_caller if not self.tricks else self.tricks[-1].winner
            self._current_trick = Trick(leader=leader, trump=self.trump_tracker.best_guess)
        return self._current_trick

    def play_tile(self, event: TilePlayed) -> None:
        if self.contract is None:
            raise HandError("cannot play before bidding is resolved")

        trick = self._ensure_trick_started()

        if trick.next_player == self.contract.trump_caller and not self.tricks and not trick.plays:
            # First tile of the hand: resolve trump from lead if not already known.
            cue = self._pending_cue
            if not self.trump_tracker.is_confirmed:
                self.trump_tracker.observe_cue_on_lead(event.tile, cue)
                trick.trump = self.trump_tracker.best_guess
            self._pending_cue = None

        if event.tile in self._seen_tiles:
            self.repairs.log_conflict(
                event=event, reason=f"tile {event.tile} already played by seat {self._seen_tiles[event.tile]}"
            )
        self._seen_tiles[event.tile] = event.player

        hand = self.hands.get(event.player) if self.hands else None

        if trick.led_suit is not None and hand is not None and not self.trump_tracker.is_confirmed:
            trick.trump = self._reconcile_trump_for_play(trick, event.player, event.tile, hand)

        trick.play(event.player, event.tile, hand=hand)
        if hand is not None:
            hand.remove(event.tile)

        if trick.is_complete:
            self._close_trick(trick)

    def _reconcile_trump_for_play(self, trick: Trick, player: int, tile: Tile, hand: List[Tile]) -> int:
        """A play that looks illegal under the current best-guess trump, when we
        know the player's hand, means the *guess* is wrong (real players can't
        illegally revoke) — not that the play is invalid. Find a candidate trump
        consistent with this play instead of raising, and soft-eliminate the old
        guess as a contradiction.
        """
        assert trick.led_suit is not None
        current = trick.trump

        def consistent(trump: int) -> bool:
            if effective_suit(tile, trump, trick.led_suit) == trick.led_suit:
                return True
            # Off-suit is only legal if the player holds no follower under `trump`.
            return not any(
                effective_suit(t, trump, trick.led_suit) == trick.led_suit for t in hand
            )

        if consistent(current):
            return current

        for candidate in sorted(self.trump_tracker.weights, key=lambda n: -self.trump_tracker.weights[n]):
            if candidate != current and consistent(candidate):
                self.trump_tracker.observe_contradiction({current}, confidence=1.0)
                return candidate

        # No candidate reconciles this play — flag for later review rather than
        # silently picking something inconsistent with observed play.
        self.repairs.log_conflict(
            event=TilePlayed(player=player, tile=tile),
            reason=f"no trump candidate is consistent with this play under led suit {trick.led_suit}",
        )
        return current

    def _close_trick(self, trick: Trick) -> None:
        assert self.scorer is not None
        self._check_trump_contradiction(trick)
        self._record_voids(trick)
        self.scorer.record_trick(trick)
        self.tricks.append(trick)
        self._current_trick = None

    def _record_voids(self, trick: Trick) -> None:
        """A player who didn't follow the led suit is known to hold none of it —
        only trustworthy once trump is confirmed (an ambiguous guess would record
        false voids). This feeds probability estimation (engine.probability),
        not legality, which is already enforced at play time.
        """
        if not self.trump_tracker.is_confirmed:
            return
        assert trick.led_suit is not None
        trump = self.trump_tracker.confirmed
        for player, tile in trick.plays:
            if effective_suit(tile, trump, trick.led_suit) != trick.led_suit:
                self.voids[player].add(trick.led_suit)

    def _check_trump_contradiction(self, trick: Trick) -> None:
        """If a player didn't follow the led suit under the current best-guess
        trump, but we later learn (via full-hand knowledge) they held a follower,
        that's a contradiction — soft-eliminate that trump guess."""
        if self.trump_tracker.is_confirmed or self.hands is None:
            return
        assert trick.led_suit is not None
        for player, tile in trick.plays:
            if effective_suit(tile, trick.trump, trick.led_suit) != trick.led_suit:
                # This player didn't follow. If we know their remaining hand and
                # they in fact hold no follower under trick.trump, no contradiction.
                # (Hand already had this tile removed by play_tile.)
                remaining = self.hands.get(player, [])
                held_follower = any(
                    effective_suit(t, trick.trump, trick.led_suit) == trick.led_suit
                    for t in remaining
                )
                if held_follower:
                    self.trump_tracker.observe_contradiction({trick.trump}, confidence=1.0)

    # ---- irregular end (concession / redeal) -----------------------------

    def classify_irregular_end(self, signal: IrregularEndSignal) -> HandOutcomeKind:
        """Route a perception-observed irregular-motion signal to a classification.

        Discriminators (see plan): tricks-played count distinguishes a misdeal
        (at/before trick 0) from a concession (mid-hand); dealer-rotation is the
        other discriminator but is only knowable once the *next* hand starts, so
        it's applied by the caller (engine.game), not here.
        """
        if signal.tricks_played_so_far == 0:
            self.outcome_kind = HandOutcomeKind.REDEAL
        else:
            self.outcome_kind = HandOutcomeKind.CONCESSION
        return self.outcome_kind

    # ---- state exposed for probability estimation -----------------------

    @property
    def played_tiles(self) -> set:
        return set(self._seen_tiles.keys())

    def remaining_hand_size(self, player: int) -> int:
        played = sum(1 for p in self._seen_tiles.values() if p == player)
        return 7 - played

    # ---- result ------------------------------------------------------

    def final_result(self) -> HandResult:
        if self.scorer is None:
            raise HandError("bidding never resolved to a contract")
        if self.outcome_kind == HandOutcomeKind.REDEAL:
            return HandResult(
                bidding_team=team_of(self.contract.bidder),  # type: ignore[union-attr]
                made=False,
                marks_awarded_to=-1,
                marks=0,
                bidding_team_points=self.scorer.bidding_team_points,
            )
        return self.scorer.final_result()
