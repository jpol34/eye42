"""Orchestrates one hand: bidding -> trump resolution -> trick play -> scoring,
including the set/redeal/concession edge cases.

Tile identity, trump, and trick-winner/turn-order are treated as jointly resolved
per the plan: ``TrumpHypothesisTracker`` narrows trump using observed (trusted)
play, and full-hand knowledge (when available, e.g. in tests or once perception
can read a deal) lets ``Trick.play`` enforce follow-suit legality directly. Repairs
to noisy tile reads are logged with provenance rather than applied in place, so a
later-proven-wrong correction can be rolled back.

A second, distinct class of problem lives here too: real human rule-breaks (an
out-of-turn play, a genuine revoke, a wrong-seat trump call, a bad deal). Per the
project's hard constraint, none of these may ever raise and halt processing of a
hand -- a physically-observed event is ground truth, and every irregularity is
logged (``RepairLog``/``Irregularity``) and reconciled as gracefully as today's
information allows, with anything unresolved surfaced only through the post-hoc
confirmation channel (never a live interruption).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from .bidding import BiddingRound, BidKind, Contract, team_of
from .events import BidMade, IrregularEndSignal, Passed, TilePlayed, TilesDealt, TrumpCueHeard
from .repair import Irregularity, IrregularityKind, RepairLog
from .scoring import HandResult, HandScoreTracker
from .tiles import Tile, effective_suit
from .trick import PlayViolation, Trick
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
    deal: Optional[TilesDealt] = field(default=None, init=False)
    misdeal_suspected: bool = field(default=False, init=False)
    _voids_trump: Optional[int] = field(default=None, init=False)  # trump the voids were recorded under
    disputed: bool = field(default=False, init=False)
    _orphan_plays: List[TilePlayed] = field(default_factory=list, init=False)
    _plays_by_seat: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.bidding = BiddingRound(dealer=self.dealer)
        if self.hands is not None:
            self._check_constructed_deal()

    def _check_constructed_deal(self) -> None:
        counts = {p: len(tiles) for p, tiles in self.hands.items()}  # type: ignore[union-attr]
        all_tiles = [t for tiles in self.hands.values() for t in tiles]  # type: ignore[union-attr]
        bad = sum(counts.values()) != 28 or any(c != 7 for c in counts.values()) or len(all_tiles) != len(set(all_tiles))
        if bad:
            self.misdeal_suspected = True
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.MISDEAL_TILE_COUNT,
                reason=f"constructed hand counts {counts} do not form a valid 28-tile deal",
                needs_confirmation=True,
            ))

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
        if self._orphan_plays:
            pending, self._orphan_plays = self._orphan_plays, []
            for event in pending:
                self._play_tile_now(event)

    @property
    def contract(self) -> Optional[Contract]:
        return self.bidding.contract

    # ---- trump ---------------------------------------------------------

    def call_trump(self, caller: int, trump: int) -> None:
        if trump not in range(7):
            # Garbage input / programming bug, not a real table event -- the
            # one remaining hard raise in this path.
            raise HandError(f"invalid trump value: {trump}")

        if self.contract is None:
            # Reachable in practice: e.g. speech parses a trump call before all
            # four bids have been parsed from overlapping audio.
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.PLAY_BEFORE_CONTRACT,
                reason=f"seat {caller} called trump {trump} before bidding resolved",
                player=caller,
                needs_confirmation=True,
            ))
            return

        if caller != self.contract.trump_caller:
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.TRUMP_CALLED_BY_WRONG_SEAT,
                reason=(
                    f"seat {caller} called trump {trump} "
                    f"(expected seat {self.contract.trump_caller})"
                ),
                player=caller,
                needs_confirmation=True,
                confidence=0.4,
            ))
            if self.trump_tracker.is_confirmed:
                return  # never overwrite a real confirmation with a suspect one

        self.trump_tracker.confirm(trump)

    def hear_trump_cue(self, cue: TrumpCueHeard) -> None:
        # Weak cues only resolve anything once combined with the led tile; stash
        # it and let the first tile-play of the hand consume it.
        self._pending_cue = cue.cue

    # ---- deal ----------------------------------------------------------

    def record_deal(self, event: TilesDealt) -> None:
        """The missing deal-event representation: buildable now (validation,
        consequences, routing into ``classify_irregular_end``), even though the
        perception signal that would populate ``counts`` from real video is
        Phase 2 and doesn't exist yet."""
        self.deal = event
        total = sum(event.counts.values())
        bad = total != 28 or any(c != 7 for c in event.counts.values()) or bool(event.exposed_tiles)
        if bad:
            self.misdeal_suspected = True
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.MISDEAL_TILE_COUNT,
                reason=f"deal counts {event.counts} (total {total}), exposed={event.exposed_tiles}",
                needs_confirmation=True,
            ))

    # ---- trick play ------------------------------------------------------

    def _ensure_trick_started(self) -> Trick:
        assert self.contract is not None
        if self._current_trick is None:
            leader = self.contract.trump_caller if not self.tricks else self.tricks[-1].winner
            self._current_trick = Trick(leader=leader, trump=self.trump_tracker.best_guess)
        return self._current_trick

    def play_tile(self, event: TilePlayed) -> None:
        if self.contract is None:
            # Perception segmented a play before it segmented the bids -- buffer
            # rather than crash; drained once bidding resolves (or, if it never
            # does, surfaces as an open question at hand end).
            self._orphan_plays.append(event)
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.PLAY_BEFORE_CONTRACT,
                reason=f"seat {event.player} played {event.tile} before bidding resolved",
                player=event.player,
                needs_confirmation=True,
            ))
            return
        self._play_tile_now(event)

    def _play_tile_now(self, event: TilePlayed) -> None:
        trick = self._ensure_trick_started()

        if not self.tricks and not trick.plays:
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

        self._plays_by_seat[event.player] = self._plays_by_seat.get(event.player, 0) + 1
        if self._plays_by_seat[event.player] > 7:
            self.misdeal_suspected = True
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.SEAT_TILE_COUNT_ANOMALY,
                reason=f"seat {event.player} has now played more than 7 tiles this hand",
                player=event.player,
                needs_confirmation=True,
            ))

        violations = trick.play(event.player, event.tile, hand=hand, strict=False)
        self._log_play_violations(event, trick, violations)

        if hand is not None:
            if event.tile in hand:
                hand.remove(event.tile)
            else:
                # A revoke, a duplicate/misread tile, or a misdeal can all land
                # here -- the play is still ground truth, so record it and move
                # on rather than crashing on list.remove(x not in list).
                self.disputed = True
                self.repairs.log_irregularity(Irregularity(
                    kind=IrregularityKind.TILE_NOT_IN_HAND,
                    reason=f"seat {event.player} played {event.tile}, not in their known remaining hand",
                    player=event.player,
                    needs_confirmation=True,
                ))

        if trick.is_complete:
            if trick.force_closed:
                self.repairs.log_irregularity(Irregularity(
                    kind=IrregularityKind.TRICK_FORCE_CLOSED,
                    reason="trick force-closed without ever reaching 4 distinct seats",
                    needs_confirmation=True,
                ))
            self._close_trick(trick)

    def _log_play_violations(self, event: TilePlayed, trick: Trick, violations: set) -> None:
        if PlayViolation.OUT_OF_TURN in violations:
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.OUT_OF_TURN,
                reason=f"seat {event.player} played out of turn",
                player=event.player,
            ))
        if PlayViolation.REVOKE in violations and self.trump_tracker.is_confirmed:
            self.disputed = True
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.REVOKE,
                reason=(
                    f"seat {event.player} played {event.tile} off suit {trick.led_suit} "
                    "while holding a follower"
                ),
                player=event.player,
                needs_confirmation=True,
            ))
        if PlayViolation.TRICK_FULL in violations or PlayViolation.SEAT_ALREADY_PLAYED in violations:
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.SEAT_TILE_COUNT_ANOMALY,
                reason=f"seat {event.player} played {event.tile} but had already played this trick",
                player=event.player,
                needs_confirmation=True,
            ))

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

        # No candidate reconciles this play under any trump hypothesis -- either
        # a tile misread, or a genuine revoke that happens to look inconsistent
        # under every candidate. Log both possibilities; the play is still
        # recorded (ground truth) via the caller's strict=False trick.play, never
        # raised on here.
        self.repairs.log_conflict(
            event=TilePlayed(player=player, tile=tile),
            reason=f"no trump candidate is consistent with this play under led suit {trick.led_suit}",
        )
        self.repairs.log_irregularity(Irregularity(
            kind=IrregularityKind.REVOKE,
            reason=(
                f"seat {player} played {tile}, not reconcilable with any trump candidate "
                f"under led suit {trick.led_suit} -- may be a real revoke"
            ),
            player=player,
            needs_confirmation=True,
            confidence=0.5,
        ))
        self.disputed = True
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
        false voids), and never for a seat already flagged with a real revoke on
        this trick (they held a follower and played off-suit anyway -- recording
        that as a void would be a *false* constraint, which is exactly what makes
        engine.probability's deal sampler unsatisfiable). This feeds probability
        estimation (engine.probability), not legality, which is already enforced
        at play time.
        """
        if not self.trump_tracker.is_confirmed:
            return
        assert trick.led_suit is not None
        trump = self.trump_tracker.confirmed
        if self._voids_trump != trump:
            # Trump changed value since the voids were recorded -- e.g. a soft
            # (inferred) confirmation was reopened by a later contradiction and
            # then re-confirmed elsewhere. Everything recorded under the
            # rejected hypothesis is a *false* constraint for the deal sampler,
            # so drop it rather than carry it forward. Placed after the
            # is_confirmed gate above on purpose: before it, this would fire on
            # every trick of an unconfirmed hand and clear voids constantly.
            self.voids = {p: set() for p in range(4)}
            self._voids_trump = trump
        for player, tile in trick.plays:
            if PlayViolation.REVOKE in trick.violations.get(player, set()):
                continue
            if effective_suit(tile, trump, trick.led_suit) != trick.led_suit:
                self.voids[player].add(trick.led_suit)

    def _check_trump_contradiction(self, trick: Trick) -> None:
        """If a player didn't follow the led suit under the current best-guess
        trump, but we later learn (via full-hand knowledge) they held a follower,
        that's a contradiction — soft-eliminate that trump guess. Skipped for a
        seat already flagged with a real revoke on this trick: that's not
        evidence the trump guess is wrong, it's evidence the human made a
        mistake, and firing a contradiction on it would poison the tracker.

        Gated on *hard* confirmation, not confirmation in general: a soft
        (inferred) confirmation is a strong guess, and this is exactly the
        evidence that should be allowed to overturn it. Without this, the
        tracker's reopening path would be unreachable dead code. The per-play
        reconciliation call site (``_reconcile_trump_for_play``) keeps its
        original blanket is_confirmed gate — see the plan's residual-risk note.

        REACHABILITY, stated plainly so a later reader isn't misled about the
        coverage this has: ``self.hands is None`` in every production hand today
        — a HandState only carries ``hands`` when it was *constructed* with a
        known deal (tests, or a full-knowledge scenario), and the perception
        that would populate it from real video is Phase 2 and does not exist
        yet. This method therefore early-returns always in production, which
        makes it the sole feed for ``TrumpHypothesisTracker``'s reopening path
        and ``_voids_trump``'s invalidation, and makes both of those test-only
        for now. That is expected at this phase, not a defect, but it does mean
        the soft-confirmation machinery is exercised by tests rather than
        proven by live play. A second, independent constraint compounds it: even
        *with* known hands, reaching a soft confirmation needs a candidate above
        CONFIRMED_THRESHOLD, and a search of 84,000 randomized legal playouts
        (300 deals x 7 possible trumps x 40 play-choice restarts, choosing
        contradiction-maximizing plays) never got a candidate above 0.75 — see
        ``TrumpHypothesisTracker.is_soft_confirmed`` for why the weight
        arithmetic makes that hard.

        Called exactly once per trick, from ``_close_trick``, on a Trick object
        that is never closed twice — so a given contradiction is evidence that
        gets counted once, and the count is bounded by the number of tricks.
        """
        if self.trump_tracker.hard_confirmed or self.hands is None:
            return
        assert trick.led_suit is not None
        for player, tile in trick.plays:
            if PlayViolation.REVOKE in trick.violations.get(player, set()):
                continue
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
        it's applied by the caller (engine.game), not here. A suspected misdeal
        already logged this hand (bad deal counts, a seat over-playing) overrides
        the trick-count heuristic -- if something already looked wrong with the
        deal itself, that's REDEAL regardless of how many tricks were played.
        """
        if self.misdeal_suspected:
            self.outcome_kind = HandOutcomeKind.REDEAL
        elif signal.tricks_played_so_far == 0:
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
        return max(0, 7 - played)

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
