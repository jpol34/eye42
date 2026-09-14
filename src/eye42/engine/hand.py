"""Orchestrates one hand: bidding -> trump resolution -> trick play -> scoring,
including the set/redeal/concession edge cases.

No player's concealed hand is ever known during live play -- perception only ever
observes a tile once it is played, with no hole-camera equivalent. Trump is
therefore resolved from observed (trusted) play alone: an explicit call, a cue
resolved against the led tile, or ``TrumpHypothesisTracker``'s weighted narrowing
over the led-tile/void evidence a real camera+mic system can actually produce.
Repairs to noisy tile reads are logged with provenance rather than applied in
place, so a later-proven-wrong correction can be rolled back.

A second, distinct class of problem lives here too: real human rule-breaks (an
out-of-turn play, a wrong-seat trump call, a bad deal). Per the
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

from ..telemetry import EventStore, repair_sink
from .bidding import (
    BidKind,
    BiddingError,
    BiddingRound,
    Contract,
    SPLASH_MIN_MARKS,
    partner_of,
    team_of,
)
from .events import (
    BidMade,
    IrregularEndSignal,
    Passed,
    TilePlayed,
    TilesDealt,
    TrumpCalled,
    TrumpCueHeard,
)
from .repair import Irregularity, IrregularityKind, RepairLog
from .scoring import TRICKS_PER_HAND, HandResult, HandScoreTracker
from .tiles import Tile, effective_suit, follows_suit
from .trick import PlayViolation, Trick
from .trump_inference import TrumpHypothesisTracker

LOW_ATTRIBUTION_CONFIDENCE = 0.6  # matches trump_inference.LOW_CONFIDENCE_MARGIN's bar for "too shaky to trust outright"


class HandOutcomeKind(Enum):
    NORMAL = auto()  # played to completion (or locked-set) with no irregular end
    CONCESSION = auto()  # ended early, players agreed the outcome was determined
    REDEAL = auto()  # thrown out (misdeal / all-pass handled separately as forced-30)


class HandError(ValueError):
    pass


@dataclass
class HandState:
    dealer: int
    store: Optional[EventStore] = None
    session_id: str = "default"
    hand_index: Optional[int] = None
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
    disputed: bool = field(default=False, init=False)  # suppresses the viewer-only probability estimate
    scoring_disputed: bool = field(default=False, init=False)  # the real score/marks can't be trusted either
    _orphan_plays: List[TilePlayed] = field(default_factory=list, init=False)
    _plays_by_seat: Dict[int, int] = field(default_factory=dict, init=False)
    _revokes_checked: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.bidding = BiddingRound(dealer=self.dealer)
        if self.store is not None:
            # The default-factory-built RepairLog above has no sink; rebuild
            # it rather than mutate, since the sink can't be threaded through
            # a bare `field(default_factory=RepairLog)`.
            self.repairs = RepairLog(sink=repair_sink(self.store, self.session_id, lambda: self.hand_index))

    # ---- event ingestion (the logging chokepoint) -------------------------

    def ingest(self, event: object, *, frame_path: Optional[str] = None) -> None:
        """Single entry point for driving a hand from real event objects
        (``events.py`` dataclasses) -- used by the live-session harness and,
        later, real perception/speech output. Logs the raw event (if a store
        is attached) before dispatching, so every event type reaching this
        method is captured with no per-type logging code to remember. Direct
        calls to ``bid``/``play_tile``/etc. (as the unit tests do) bypass this
        and are not logged -- this is the boundary that makes logging
        automatic, not the individual methods.

        ``frame_path`` lets a camera-backed caller correlate this exact event
        with the frame that was on the table when it was observed -- the
        engine itself has no camera knowledge; it just carries the path
        through to the stored row."""
        if self.store is not None:
            self.store.log_event(
                event, session_id=self.session_id, hand_index=self.hand_index, frame_path=frame_path
            )

        if isinstance(event, BidMade):
            self.bid(event.player, event.amount, event.marks)
        elif isinstance(event, Passed):
            self.bid_pass(event.player)
        elif isinstance(event, TrumpCalled):
            self.call_trump(event.caller, event.trump)
        elif isinstance(event, TrumpCueHeard):
            self.hear_trump_cue(event)
        elif isinstance(event, TilePlayed):
            self.play_tile(event)
        elif isinstance(event, TilesDealt):
            self.record_deal(event)
        elif isinstance(event, IrregularEndSignal):
            self.classify_irregular_end(event)
        else:
            raise HandError(f"ingest: unrecognized event type {type(event).__name__}")

    # ---- bidding -----------------------------------------------------

    def bid(self, player: int, amount: int = 0, marks: int = 0) -> None:
        """Record a bid. Non-strict by design (the same contract every other
        ``HandState`` event method honours): ``BiddingRound`` is the strict pure-
        rules layer, but a bid arriving here came from perception/speech and may
        be out of turn, under the current high bid, or after bidding closed. Such
        an event is logged and dropped, never raised on -- a Phase-3 speech front
        end producing noisy bids must not be able to halt a hand."""
        try:
            self.bidding.record_bid(BidMade(player=player, amount=amount, marks=marks))
        except BiddingError as exc:
            self._log_rejected_bid(player, f"bid (amount={amount}, marks={marks})", exc)
            return
        self._on_bidding_closed()

    def bid_pass(self, player: int) -> None:
        try:
            self.bidding.record_pass(Passed(player=player))
        except BiddingError as exc:
            self._log_rejected_bid(player, "pass", exc)
            return
        self._on_bidding_closed()

    def _log_rejected_bid(self, player: int, what: str, exc: BiddingError) -> None:
        self.repairs.log_irregularity(Irregularity(
            kind=IrregularityKind.BID_NOT_LEGAL,
            reason=f"seat {player} {what} rejected by the bidding rules: {exc}",
            player=player,
            needs_confirmation=True,
        ))

    def _on_bidding_closed(self) -> None:
        """The one place that reacts to bidding having just resolved, shared by
        ``bid`` and ``bid_pass``."""
        if not self.bidding.is_done:
            return
        self._start_score_tracker()

    def _start_score_tracker(self) -> None:
        if self.bidding.contract is None:
            return
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

        if not self.tricks and (self._current_trick is None or not self._current_trick.plays):
            # Only evidence of splash/plunge before the hand's first tile is
            # played -- a stray/misheard call_trump arriving mid-hand (e.g.
            # after a real trump call and lead already happened normally)
            # must not retroactively reclassify an already-correct contract
            # and silently overwrite an already-confirmed trump.
            self._maybe_infer_splash_or_plunge(caller)

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

    def _maybe_infer_splash_or_plunge(self, actor: int) -> bool:
        """A plain marks contract's trump-caller role landing on the bidder's
        PARTNER (not the bidder) -- whether by the partner leading the first
        trick, or the partner calling trump before any tile is played -- is
        exactly what splash/plunge looks like from observed play alone.
        ``Contract.trump_caller`` already resolves correctly for SPLASH/
        PLUNGE, but a plain ``BidKind.MARKS`` contract still points
        ``trump_caller`` at the bidder, so without this the partner's call or
        lead would just get logged as a false wrong-seat/out-of-turn
        irregularity. Called from both ``call_trump`` and the first-lead
        handling in ``_play_tile_now``, whichever happens first in a given
        hand; a no-op once the contract is no longer a plain ``MARKS`` bid
        (already upgraded, or never was one).

        Always reclassifies to SPLASH, never PLUNGE: nothing observable live
        (no hole-camera) distinguishes them, and every behavior that matters
        here -- ``trump_caller``, ``marks_at_stake``, ``points_needed``,
        ``requires_sweep`` -- is identical between the two, so the label is a
        cosmetic guess only. Requiring ``SPLASH_MIN_MARKS`` (the lower of the
        two thresholds) rather than ``PLUNGE_MIN_MARKS`` means a real 4-mark
        plunge is never rejected by this check.

        Below ``SPLASH_MIN_MARKS`` marks, no valid splash/plunge bid exists,
        so the partner acting as trump-caller is left as a genuine
        irregularity instead (unchanged existing handling). Returns whether
        it upgraded the contract, so a caller that also needs to fix up
        other state (e.g. a trick's already-assigned leader) knows to do so.
        """
        contract = self.contract
        if contract is None or contract.kind != BidKind.MARKS:
            return False
        if self.trump_tracker.is_confirmed:
            # Trump is already settled -- by the bidder (ruling out splash/
            # plunge outright, since the bidder never calls trump under a
            # real one) or by an earlier, legitimate run of this same
            # inference. Either way, a later partner-as-trump-caller signal
            # (a stray/duplicate call, or an out-of-turn/misattributed lead)
            # is no longer evidence of anything -- it's an irregularity to
            # flag through the ordinary wrong-seat/out-of-turn paths, not a
            # license to retroactively reclassify an already-settled contract.
            return False
        if actor != partner_of(contract.bidder) or contract.amount < SPLASH_MIN_MARKS:
            return False
        self.bidding.upgrade_to_splash_or_plunge(BidKind.SPLASH)
        self.repairs.log_irregularity(Irregularity(
            kind=IrregularityKind.SPLASH_PLUNGE_INFERRED,
            reason=(
                f"seat {actor} (bidder {contract.bidder}'s partner) acted as trump-caller "
                f"under a {contract.amount}-mark contract -- reclassified as splash/plunge"
            ),
            player=actor,
            needs_confirmation=True,
        ))
        return True

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

    def _refuse_extra_trick(self, event: TilePlayed) -> None:
        """A play that would have to open an 8th trick. Seven is the whole hand;
        anything past it is a spurious/duplicated read, and accepting it silently
        corrupts the score (a hand capped at 42 computing 65 points, a team that
        swept every trick reported as ``made=False``).

        Marked ``disputed`` rather than ``misdeal_suspected`` on purpose: a
        misdeal forces ``classify_irregular_end`` to REDEAL and zeroes the marks
        for the whole hand off one stray event, which is wildly disproportionate.
        ``disputed`` suppresses the (viewer-only) probability display and leaves
        the real score alone.
        """
        self.disputed = True
        self.repairs.log_irregularity(Irregularity(
            kind=IrregularityKind.TRICK_COUNT_EXCEEDED,
            reason=(
                f"seat {event.player} played {event.tile} after all "
                f"{TRICKS_PER_HAND} tricks were already complete"
            ),
            player=event.player,
            needs_confirmation=True,
        ))
        self.repairs.log_conflict(
            event=event, reason=f"play arrives after {TRICKS_PER_HAND} completed tricks"
        )

    def _starts_a_new_trick(self, trick: Trick, event: TilePlayed) -> bool:
        """Does this play from a seat that already played this trick actually
        mean the *next* trick started while we missed a tile?

        Without this, one occluded play cascades: the seat-already-played case
        was unconditionally treated as superseded, so an entire following trick
        (4 real tiles) could vanish into ``superseded_plays``, flipping the
        hand's score. With it, both discriminators are required, because the
        naive "≥3 seats in" test alone breaks an ordinary double-read:

        (a) an *identical* tile from that seat is a duplicate CV read of the
            play we already have, never a new trick; and
        (b) the incoming seat must be the one entitled to lead next -- i.e. the
            current trick's winner -- or this is just another stray read.
        """
        if trick.is_complete or event.player not in trick.seats_played:
            return False
        if any(p == event.player and t == event.tile for p, t in trick.plays):
            return False  # duplicate read of the same physical tile
        if len(trick.seats_played) < 3:
            return False
        return trick.winner == event.player

    def _play_tile_now(self, event: TilePlayed) -> None:
        if event.player_confidence < LOW_ATTRIBUTION_CONFIDENCE:
            # A shaky guess at *who* played this is recorded as ground truth like
            # any other observed play (the table is ground truth, never rejected),
            # but it's flagged here rather than trusted silently -- an attribution
            # error that's still turn-legal would otherwise feed voids/trump
            # inference under the wrong seat with nothing to catch it.
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.AMBIGUOUS_ATTRIBUTION,
                reason=f"seat {event.player} attributed at confidence {event.player_confidence:.2f}",
                player=event.player,
                needs_confirmation=True,
                confidence=event.player_confidence,
            ))

        if self._current_trick is None and len(self.tricks) >= TRICKS_PER_HAND:
            self._refuse_extra_trick(event)
            return

        trick = self._ensure_trick_started()

        if self._starts_a_new_trick(trick, event):
            trick.closed_early_for_new_trick = True
            # This trick is genuinely short a tile -- the missed seat's play
            # never arrived -- so its count value is silently lost from the
            # real score unless this is logged. Not suppressed from the
            # probability estimator (see _has_occluded_trick); that's a
            # separate, deliberate trade-off. This is just the audit trail.
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.TRICK_CLOSED_EARLY,
                reason="a seat's play started the next trick before this one reached 4 plays",
                player=event.player,
                needs_confirmation=True,
            ))
            self._close_trick(trick)
            if len(self.tricks) >= TRICKS_PER_HAND:
                # The early close was a real trick boundary, so it counts toward
                # the seven -- and if it was the seventh, this play has no trick
                # left to belong to.
                self._refuse_extra_trick(event)
                return
            trick = self._ensure_trick_started()

        if not self.tricks and not trick.plays:
            if self._maybe_infer_splash_or_plunge(event.player):
                trick.leader = event.player
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
        # First writer wins. A duplicate/misread tile must not silently erase the
        # earlier seat's attribution -- that feeds the deal sampler, which would
        # then happily re-deal a tile lying face-up on the table. The conflict is
        # already logged one line above; that log is the resolution path, not a
        # silent overwrite. (Deliberately *not* switching remaining_hand_size /
        # played_tiles over to _plays_by_seat: that regresses every duplicate-read
        # hand to sum(sizes) != len(unseen), which blanks both estimators where
        # they currently still work.)
        self._seen_tiles.setdefault(event.tile, event.player)

        self._plays_by_seat[event.player] = self._plays_by_seat.get(event.player, 0) + 1
        if self._plays_by_seat[event.player] > 7:
            self.misdeal_suspected = True
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.SEAT_TILE_COUNT_ANOMALY,
                reason=f"seat {event.player} has now played more than 7 tiles this hand",
                player=event.player,
                needs_confirmation=True,
            ))

        violations = trick.play(event.player, event.tile, strict=False)
        if PlayViolation.TRICK_FULL not in violations and PlayViolation.SEAT_ALREADY_PLAYED not in violations:
            # Only a play actually recorded into trick.plays (not bounced to
            # superseded_plays) should ever set this -- a rejected duplicate's
            # confidence must never clobber the real play's.
            trick.play_confidence[event.player] = (event.confidence, event.player_confidence)
        self._log_play_violations(event, trick, violations)

        if trick.is_complete:
            if trick.force_closed:
                self.repairs.log_irregularity(Irregularity(
                    kind=IrregularityKind.TRICK_FORCE_CLOSED,
                    reason="trick force-closed without ever reaching 4 distinct seats",
                    needs_confirmation=True,
                ))
                # A trick only force-closes while fewer than 4 seats have played
                # (see Trick.force_closed), so some seats' tiles were never
                # observed here -- the count value this trick contributes to the
                # running score reflects only the tiles perception actually
                # caught, not zero, and whichever team won it may be missing
                # count they were rightfully owed. HandScoreTracker's arithmetic
                # (is_locked_set, bidding_team_points) has no way to tell
                # "accounted for" from "missing," so the real score -- not just
                # the viewer-only estimate -- can no longer be trusted for this
                # hand; final_result() checks scoring_disputed and withholds
                # marks rather than award them off corrupted arithmetic.
                self.disputed = True
                self.scoring_disputed = True
            self._close_trick(trick)

    def _log_play_violations(self, event: TilePlayed, trick: Trick, violations: set) -> None:
        if PlayViolation.OUT_OF_TURN in violations:
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.OUT_OF_TURN,
                reason=f"seat {event.player} played out of turn",
                player=event.player,
            ))
        if PlayViolation.TRICK_FULL in violations or PlayViolation.SEAT_ALREADY_PLAYED in violations:
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.SEAT_TILE_COUNT_ANOMALY,
                reason=f"seat {event.player} played {event.tile} but had already played this trick",
                player=event.player,
                needs_confirmation=True,
            ))

    def _close_trick(self, trick: Trick) -> None:
        self._record_voids(trick)
        self._narrow_trump_from_voids(trick)
        if self.scorer is None:
            # Explicit guard, not an assert: `python -O` strips asserts, and a
            # trick can complete with the contract set but the scorer never
            # built. Log and skip the scoring rather than halting a hand that
            # is otherwise fine.
            self.repairs.log_irregularity(Irregularity(
                kind=IrregularityKind.PLAY_BEFORE_CONTRACT,
                reason="trick completed before a score tracker existed; not scored",
                needs_confirmation=True,
            ))
        else:
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

        ACCEPTED RESIDUAL RISK (matching this project's existing pattern for
        gaps of this shape): the voids-half of the late-confirmation problem is
        fixed below, but ``trick.trump``/``trick.winner``/``scorer.record_trick``
        still adjudicate such a trick under the stale *guessed* trump. Fixing
        that half needs the retroactive-reinterpretation machinery that has
        already been deferred twice; it is not built here.
        """
        if not self.trump_tracker.is_confirmed:
            return
        if trick.led_suit is None or not trick.plays:
            return
        trump = self.trump_tracker.confirmed
        if self._voids_trump != trump:
            # Trump changed value since the voids were recorded -- e.g. the
            # trump-caller called trump a second time with a different value.
            # Everything recorded under the rejected hypothesis is a *false*
            # constraint for the deal sampler,
            # so drop it rather than carry it forward. Placed after the
            # is_confirmed gate above on purpose: before it, this would fire on
            # every trick of an unconfirmed hand and clear voids constantly.
            self.voids = {p: set() for p in range(4)}
            self._voids_trump = trump

        # ``trick.led_suit`` was frozen when the trick was led, under whatever
        # trump was the best guess *then*. If trump was confirmed one tile later
        # (speech lagging video -- the realistic case for this project), the led
        # tile may count as a different suit entirely under the confirmed trump,
        # and every "didn't follow" judgement below would be measured against the
        # wrong suit. The existing `_voids_trump` staleness check cannot catch
        # this: it compares against the *current* trump, which is exactly the one
        # being applied incorrectly. Skip this trick's voids rather than record
        # false ones -- but still stamp `_voids_trump`, or the next trick sees a
        # mismatch and wipes the (good) voids for no reason.
        if not self._led_suit_is_trustworthy(trick, trump):
            self._voids_trump = trump
            return

        for player, tile in trick.plays:
            if PlayViolation.REVOKE in trick.violations.get(player, set()):
                continue
            if effective_suit(tile, trump, trick.led_suit) != trick.led_suit:
                self.voids[player].add(trick.led_suit)

    def _narrow_trump_from_voids(self, trick: Trick) -> None:
        """Mid-hand trump narrowing from observed void contradictions, live
        only while trump is still unconfirmed -- once confirmed,
        ``_record_voids`` is the authoritative single-trump mechanism and
        this stops being consulted.

        For every candidate trump ``n``, a player who doesn't follow this
        trick's led suit *under n* is recorded void in that suit under n
        (skipping a seat already flagged with a real ``REVOKE`` this trick,
        same as ``_record_voids`` -- a genuine revoke means they still hold
        it, so recording a void would be a false constraint). A later play
        (this trick or a future one) that computes as a previously-voided
        suit under n is a hard contradiction: they can't be void in that
        suit under n and also hold/play it under n, so n must be wrong.
        Reuses ``effective_suit``/``follows_suit`` (already parameterized by
        an arbitrary trump) and ``TrumpHypothesisTracker.observe_contradiction``
        unchanged -- no folklore weighting, just the same publicly-observed
        play stream. Multiple independent contradictions against the same
        candidate within one trick collapse into a single
        ``observe_contradiction`` call on the worst confidence among them,
        honoring that method's own at-most-once-per-trick contract.
        """
        if self.trump_tracker.is_confirmed:
            return
        if trick.led_suit is None or not trick.plays:
            return
        _, led_tile = trick.plays[0]

        for n in range(7):
            voids_n = self.trump_tracker.voids[n]
            led_suit_under_n = effective_suit(led_tile, n, None)
            for player, tile in trick.plays[1:]:
                if PlayViolation.REVOKE in trick.violations.get(player, set()):
                    continue  # a real revoke means they still hold it -- not a void
                if not follows_suit(tile, n, led_suit_under_n):
                    tile_confidence, player_confidence = trick.play_confidence.get(player, (1.0, 1.0))
                    voids_n[player][led_suit_under_n] = min(tile_confidence, player_confidence)

            # At most one observe_contradiction call per candidate per trick
            # (its own contract, trump_inference.py) -- collect every match
            # this trick found for `n` and fire once on the worst confidence
            # among them, rather than compounding the penalty per match.
            contradiction_confidence: Optional[float] = None
            for player, tile in trick.plays:
                tile_confidence, player_confidence = trick.play_confidence.get(player, (1.0, 1.0))
                reveal_confidence = min(tile_confidence, player_confidence)
                for suit, void_confidence in voids_n[player].items():
                    if follows_suit(tile, n, suit):
                        match_confidence = min(void_confidence, reveal_confidence)
                        if contradiction_confidence is None or match_confidence < contradiction_confidence:
                            contradiction_confidence = match_confidence
            if contradiction_confidence is not None:
                self.trump_tracker.observe_contradiction({n}, confidence=contradiction_confidence)

    def _led_suit_is_trustworthy(self, trick: Trick, trump: int) -> bool:
        """A trick's ``led_suit`` was frozen when it was led, under whatever
        trump was the best guess *then*. If trump was confirmed later (speech
        lagging video -- the realistic case for this project) and reinterprets
        the led tile into a different suit, every "didn't follow" judgement
        against ``trick.led_suit`` would be measured against the wrong suit.
        Shared by ``_record_voids`` and ``_detect_revokes``, both of which
        need to skip a trick this has happened to rather than judge it wrong.
        """
        if trick.led_suit is None or not trick.plays:
            return False
        led_tile = trick.plays[0][1]
        return effective_suit(led_tile, trump, None) == trick.led_suit

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

    # ---- live state (the "game layer" projection) -------------------------

    def live_state(self) -> dict:
        """Read-only snapshot of what's happening right now, computed from the
        live objects (never re-derived from the log) -- the shape the live-
        session viewer polls."""
        contract = self.contract
        current_trick = None
        if self._current_trick is not None:
            current_trick = {
                "leader": self._current_trick.leader,
                "plays": [[p, str(t)] for p, t in self._current_trick.plays],
            }
        return {
            "dealer": self.dealer,
            "current_bidder": self.bidding.current_bidder,
            "contract": None if contract is None else {
                "bidder": contract.bidder,
                "kind": contract.kind.name,
                "amount": contract.amount,
                "trump_caller": contract.trump_caller,
            },
            "trump": {
                "confirmed": self.trump_tracker.confirmed,
                "best_guess": self.trump_tracker.best_guess,
            },
            "current_trick": current_trick,
            "tricks": [
                {"winner": t.winner, "plays": [[p, str(tile)] for p, tile in t.plays]}
                for t in self.tricks
            ],
            "bidding_team_points": None if self.scorer is None else self.scorer.bidding_team_points,
            "outcome_kind": None if self.outcome_kind is None else self.outcome_kind.name,
            "open_questions": [
                {"kind": q.kind.name, "reason": q.reason, "player": q.player}
                for q in self.repairs.open_questions
            ],
        }

    def _detect_revokes(self) -> None:
        """Retroactive revoke detection.

        ``PlayViolation.REVOKE`` (trick.py) only fires when a player's
        concealed hand is passed into ``Trick.check_play``/``play`` -- but no
        player's hand is ever observed live (no hole-camera), so ``hand.py``'s
        live path always calls ``trick.play(..., strict=False)`` with no
        ``hand`` argument, and that violation never fires there in practice.

        By a hand's end, though, every seat's holding at any past moment is
        reconstructable without any oracle knowledge: a player's remaining
        hand at trick *i* is exactly the tiles they go on to play in trick
        *i* and every trick after it (a played tile leaves the hand and is
        never seen again, and all 28 tiles are accounted for by the end of a
        cleanly-completed hand). So if a player didn't follow the led suit in
        trick *i* but later plays a tile that would have followed it, that's
        a hard contradiction: they held a follower and chose not to play it.

        Only runs when the whole hand's data is trustworthy enough for that
        reconstruction to actually hold -- skips entirely (never guesses)
        otherwise, same conservative posture as ``_record_voids``.
        """
        if self._revokes_checked:
            return
        self._revokes_checked = True

        if not self.trump_tracker.is_confirmed:
            return
        if self.disputed or self.scoring_disputed or self.misdeal_suspected:
            return
        if self.repairs.has_conflicts:
            # A misread/duplicate tile attributed to the wrong seat would
            # corrupt exactly the union-of-plays reconstruction this relies
            # on -- see the conflict log for provenance.
            return
        if any(ir.kind == IrregularityKind.AMBIGUOUS_ATTRIBUTION for ir in self.repairs.irregularities):
            # A shaky guess at *who* played a tile breaks the same
            # reconstruction assumption -- the union of plays is only
            # trustworthy if every play's seat attribution is trustworthy too.
            return
        if len(self.tricks) != TRICKS_PER_HAND:
            return
        for trick in self.tricks:
            if trick.force_closed or trick.closed_early_for_new_trick or trick.superseded_plays:
                return

        trump = self.trump_tracker.confirmed
        for i, trick in enumerate(self.tricks):
            led_suit = trick.led_suit
            if not self._led_suit_is_trustworthy(trick, trump):
                continue
            for player, tile in trick.plays:
                if PlayViolation.REVOKE in trick.violations.get(player, set()):
                    continue
                if follows_suit(tile, trump, led_suit):
                    continue
                held_a_follower_later = any(
                    follows_suit(later_tile, trump, led_suit)
                    for later_trick in self.tricks[i + 1:]
                    for later_player, later_tile in later_trick.plays
                    if later_player == player
                )
                if held_a_follower_later:
                    # `_record_voids` ran live, before this trick's status as
                    # a genuine revoke (rather than a real void) was knowable,
                    # and recorded `led_suit` as a void for `player` -- a
                    # false constraint per its own docstring. Undo it now that
                    # better information exists, or a stale false void keeps
                    # corrupting engine.probability's deal sampler for the
                    # rest of the hand's life.
                    self.voids[player].discard(led_suit)
                    self.repairs.log_irregularity(Irregularity(
                        kind=IrregularityKind.REVOKE,
                        reason=(
                            f"seat {player} did not follow suit {led_suit} in trick "
                            f"{i + 1} but later played a tile of that suit"
                        ),
                        player=player,
                        needs_confirmation=True,
                    ))

    # ---- result ------------------------------------------------------

    def final_result(self) -> HandResult:
        if self.scorer is None:
            raise HandError("bidding never resolved to a contract")
        self._detect_revokes()
        if self.outcome_kind == HandOutcomeKind.REDEAL:
            return HandResult(
                bidding_team=team_of(self.contract.bidder),  # type: ignore[union-attr]
                made=False,
                marks_awarded_to=-1,
                marks=0,
                bidding_team_points=self.scorer.bidding_team_points,
            )
        if self.scoring_disputed:
            # is_locked_set/bidding_team_points can't distinguish a trick's
            # count as "zero" from "never observed" once one force-closed
            # short, so neither made-or-not nor the point total can be trusted
            # -- withhold marks (same -1 sentinel as REDEAL, which game.py
            # already treats as "award nothing") rather than resolve a made/
            # missed question from corrupted arithmetic.
            return HandResult(
                bidding_team=team_of(self.contract.bidder),  # type: ignore[union-attr]
                made=False,
                marks_awarded_to=-1,
                marks=0,
                bidding_team_points=self.scorer.bidding_team_points,
            )
        return self.scorer.final_result()
