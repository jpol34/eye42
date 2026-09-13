from __future__ import annotations

from eye42.engine.bidding import BidKind
from eye42.engine.events import IrregularEndSignal, TilePlayed, TilesDealt
from eye42.engine.hand import HandOutcomeKind, HandState
from eye42.engine.probability import (
    _has_occluded_trick,
    estimate_bid_probability,
    tile_hold_probability,
)
from eye42.engine.repair import IrregularityKind
from eye42.engine.tiles import Tile


def _kinds(hand: HandState) -> set:
    return {i.kind for i in hand.repairs.irregularities}


# ---------------------------------------------------------------------------
# out-of-turn play
# ---------------------------------------------------------------------------

def test_out_of_turn_lead_is_recorded_not_raised():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Seat 2 leads instead of the entitled seat 0.
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))

    assert len(hand.tricks) == 1
    assert hand.tricks[0].leader == 2
    assert hand.tricks[0].entitled_leader == 0
    assert IrregularityKind.OUT_OF_TURN in _kinds(hand)


def test_skipped_seat_recovers_turn_order():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # 0 leads, then 2 plays out of turn (skipping 1), then 1 and 3 catch up.
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))

    assert len(hand.tricks) == 1
    assert {p for p, _ in hand.tricks[0].plays} == {0, 1, 2, 3}
    assert IrregularityKind.OUT_OF_TURN in _kinds(hand)


def test_duplicate_seat_play_goes_to_superseded_and_scores_nothing():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    # Seat 1 plays again (e.g. a CV double-read) before seats 2/3 have played.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))

    assert len(hand.tricks) == 1
    trick = hand.tricks[0]
    assert {p for p, _ in trick.plays} == {0, 1, 2, 3}
    assert (1, Tile.of(5, 5)) in trick.superseded_plays
    assert trick.count_value == 0  # the superseded 5-5 (10-count) never scores


def test_trick_force_closes_instead_of_hanging():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    # Seat 2 never appears at all (occluded during a sweep, say); seats 1 and 3
    # each get double-read a couple of times while the trick waits for it.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 4)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(2, 2)))

    assert len(hand.tricks) == 1  # force-closed rather than left open forever
    assert hand.tricks[0].force_closed is True
    assert IrregularityKind.TRICK_FORCE_CLOSED in _kinds(hand)

    # The hand keeps going afterward -- a later play starts a fresh trick.
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(4, 4)))
    assert hand._current_trick is not None


# ---------------------------------------------------------------------------
# call_trump edge cases
# ---------------------------------------------------------------------------

def test_call_trump_before_contract_resolved_does_not_assert():
    hand = HandState(dealer=3)
    hand.call_trump(0, trump=6)  # bidding hasn't even started
    assert hand.contract is None
    assert IrregularityKind.PLAY_BEFORE_CONTRACT in _kinds(hand)


def test_wrong_seat_trump_call_is_logged_and_does_not_overwrite_confirmation():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    hand.call_trump(0, trump=6)  # correct seat confirms first
    hand.call_trump(1, trump=3)  # wrong seat tries to call trump afterward

    assert hand.trump_tracker.confirmed == 6  # unchanged
    assert IrregularityKind.TRUMP_CALLED_BY_WRONG_SEAT in _kinds(hand)


def test_invalid_trump_value_still_raises():
    import pytest
    from eye42.engine.hand import HandError

    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    with pytest.raises(HandError):
        hand.call_trump(0, trump=9)


# ---------------------------------------------------------------------------
# deal / misdeal
# ---------------------------------------------------------------------------

def test_misdeal_tile_count_is_logged_not_raised():
    hand = HandState(dealer=3)
    hand.record_deal(TilesDealt(dealer=3, counts={0: 8, 1: 7, 2: 7, 3: 6}))
    assert hand.misdeal_suspected is True
    assert IrregularityKind.MISDEAL_TILE_COUNT in _kinds(hand)


def test_misdeal_routes_to_redeal_regardless_of_trick_count():
    hand = HandState(dealer=3)
    hand.record_deal(TilesDealt(dealer=3, counts={0: 8, 1: 7, 2: 7, 3: 6}))
    outcome = hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=3))
    assert outcome == HandOutcomeKind.REDEAL


def test_seat_playing_an_eighth_tile_is_logged_and_hand_size_clamps():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    trump_leads = [Tile.of(6, n) for n in (6, 5, 4, 3, 2, 1, 0)]
    fillers = [
        [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1)],
        [Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2)],
        [Tile.of(3, 0), Tile.of(3, 1), Tile.of(3, 3)],
        [Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2)],
        [Tile.of(4, 3), Tile.of(4, 4), Tile.of(5, 1)],
        [Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3)],
        [Tile.of(5, 4), Tile.of(5, 5), Tile.of(5, 0)],
    ]
    for lead, (f1, f2, f3) in zip(trump_leads, fillers):
        hand.play_tile(TilePlayed(player=0, tile=lead))
        hand.play_tile(TilePlayed(player=1, tile=f1))
        hand.play_tile(TilePlayed(player=2, tile=f2))
        hand.play_tile(TilePlayed(player=3, tile=f3))

    assert len(hand.tricks) == 7

    # Seat 0 "plays" an 8th tile -- e.g. a stray misread after the hand is done.
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 2)))

    # Refused before it can open an 8th trick, and marked `disputed` rather than
    # `misdeal_suspected` on purpose: a misdeal forces the whole hand to REDEAL
    # (zero marks) off one stray read, which is disproportionate. `disputed` only
    # suppresses the viewer-facing probability estimate.
    assert len(hand.tricks) == 7
    assert hand.disputed is True
    assert hand.misdeal_suspected is False
    assert IrregularityKind.TRICK_COUNT_EXCEEDED in _kinds(hand)
    assert hand.repairs.has_conflicts
    assert hand.remaining_hand_size(0) == 0  # clamped, never negative


def test_play_before_contract_is_buffered_and_replayed():
    hand = HandState(dealer=3)
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    assert IrregularityKind.PLAY_BEFORE_CONTRACT in _kinds(hand)
    assert len(hand.tricks) == 0

    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)  # bidding resolves here -- draining happens immediately

    assert hand._orphan_plays == []
    assert hand._current_trick is not None
    assert (0, Tile.of(6, 6)) in hand._current_trick.plays


# ---------------------------------------------------------------------------
# bug-hunt fix pass: occlusion cascade (F3), trick bound (F4), duplicate
# attribution (F5), and estimator suppression (B3)
# ---------------------------------------------------------------------------

def _contracted_hand() -> HandState:
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)
    return hand


def test_identical_replay_is_still_superseded_not_a_new_trick():
    """F3, discriminator (a). The sequence 0, 1, 2, seat-1-replays-the-*same*-
    tile, 3 is an ordinary CV double-read and must still complete as one 4-seat
    trick. A naive "≥3 seats are in, so this must be the next trick" rule breaks
    exactly here -- seat 1 is not even the entitled next leader, and the tile is
    one already on the table."""
    hand = _contracted_hand()
    for player, tile in [(0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))  # duplicate read
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))

    assert len(hand.tricks) == 1
    trick = hand.tricks[0]
    assert trick.seats_played == {0, 1, 2, 3}
    assert (1, Tile.of(3, 0)) in trick.superseded_plays
    assert trick.closed_early_for_new_trick is False


def test_a_replay_from_a_seat_that_is_not_the_entitled_leader_is_superseded():
    """F3, discriminator (b). A different tile from a seat that already played,
    when that seat is not the trick's winner, is a stray read -- not the next
    trick's lead."""
    hand = _contracted_hand()
    for player, tile in [(0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    assert hand._current_trick.winner == 0  # seat 1 is not entitled to lead next
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))

    assert len(hand.tricks) == 1
    assert hand.tricks[0].seats_played == {0, 1, 2, 3}
    assert (1, Tile.of(5, 5)) in hand.tricks[0].superseded_plays


def test_one_occluded_play_does_not_swallow_the_following_trick():
    """F3. Seat 3's tile is never read. Seat 0 -- the current trick's winner, so
    the seat entitled to lead next -- then leads. Previously every play from a
    seat that had already played this trick was absorbed into
    ``superseded_plays``, so the whole next trick (4 real tiles) vanished; one
    dropped tile mid-hand flipped bidding-team points from 0 to 13."""
    hand = _contracted_hand()
    for player, tile in [(0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 5)))  # the next trick's lead

    assert len(hand.tricks) == 1
    closed = hand.tricks[0]
    assert len(closed.plays) == 3
    assert closed.closed_early_for_new_trick is True
    assert closed.force_closed is False  # a resolved boundary, not an occlusion bail
    assert closed.superseded_plays == []

    assert hand._current_trick is not None
    assert hand._current_trick.leader == 0
    assert hand._current_trick.plays == [(0, Tile.of(6, 5))]

    # The trick is genuinely short a tile -- the missed seat's play never
    # arrived -- so its count value is silently lost from the real score
    # unless this is logged. Not suppressing the probability estimate for it
    # (see the next test) is a deliberate, separate call; the audit trail is
    # not optional.
    diverged = [i for i in hand.repairs.irregularities if i.kind == IrregularityKind.TRICK_CLOSED_EARLY]
    assert len(diverged) == 1
    assert diverged[0].needs_confirmation is True
    assert diverged[0] in hand.repairs.open_questions


def test_an_f3_early_close_does_not_suppress_the_estimate():
    """B3. ``closed_early_for_new_trick`` is a *resolved* state with no missing
    tiles, so the equity display must keep working -- this is why it needs its
    own marker rather than reusing ``force_closed``."""
    hand = _contracted_hand()
    for player, tile in [(0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 5)))

    assert _has_occluded_trick(hand) is False
    assert estimate_bid_probability(hand, samples=40) is not None
    # The other kind of early close is fully resolved -- nothing is missing --
    # so it must not be mistaken for the disputed, tiles-genuinely-unread case.
    assert hand.disputed is False


def test_a_force_closed_short_trick_does_suppress_the_estimate():
    """B3. A trick that force-closed with fewer than four real plays means tiles
    genuinely went unread; the simulation then strands the missed seats' surplus
    tiles and under-totals the hand's 42-point budget with nothing surfacing it.
    Suppress the viewer-only estimate instead."""
    hand = _contracted_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    # Seat 2 never appears; 1 and 3 get double-read until the trick force-closes.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 4)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(2, 2)))

    assert hand.tricks[0].force_closed is True
    assert len(hand.tricks[0].plays) < 4
    assert _has_occluded_trick(hand) is True
    assert estimate_bid_probability(hand, samples=40) is None
    # The hand's remaining-points arithmetic can no longer tell "accounted for"
    # from "missing" once a trick force-closed short, so the hand is disputed
    # on top of (separately from) the estimator's own occlusion check.
    assert hand.disputed is True


def test_a_force_closed_short_trick_still_records_its_winner_and_count():
    """The disputed flag marks the hand's remaining-points bookkeeping as
    untrustworthy; it must not erase the trick that actually happened. The
    known tiles' identities and count are still real observed facts, kept in
    both ``hand.tricks`` and the scorer so a future reconciliation mechanism
    (not built here) would still have them to work with."""
    hand = _contracted_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))  # trump, wins the trick
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))  # 10-count
    # Seat 2 never appears; 1 and 3 get double-read until the trick force-closes.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 4)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(2, 2)))

    assert hand.disputed is True
    assert len(hand.tricks) == 1
    trick = hand.tricks[0]
    assert trick.winner == 0
    assert trick.count_value == 10  # the 5-5 seen before the trick force-closed
    assert trick in hand.scorer.completed_tricks
    assert hand.scorer.bidding_team_points == 1 + 10  # seat 0 bid; trick + count


def test_scoring_disputed_suppresses_even_the_exact_answer_short_circuits():
    """estimate_bid_probability answers is_locked_set/points-already-met
    analytically, without sampling, before its general disputed/occlusion
    guard runs -- so a force-closed-short trick (which corrupts that same
    arithmetic, not just the sampler) must be caught earlier, via
    scoring_disputed, or these exact short-circuits would confidently answer
    from the very numbers that can no longer be trusted."""
    hand = HandState(dealer=3)
    hand.bid(0, 42)  # needs every point in the hand; losing any trick locks it
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(0, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(6, 6)))  # trump, wins for the defense
    # Seat 2 never appears; 0 and 3 get double-read until the trick force-closes.
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 2)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(3, 3)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(4, 4)))

    assert hand.scoring_disputed is True
    assert hand.scorer.is_locked_set is True  # would trigger the exact 0.0 short-circuit
    assert estimate_bid_probability(hand, samples=40) is None


def test_scoring_disputed_hand_withholds_marks_at_final_result():
    """The real score/marks determination -- not just the viewer estimate --
    must not be computed from is_locked_set/bidding_team_points once a trick
    force-closed short. final_result() withholds marks (the same -1 sentinel
    used for a REDEAL, which game.py already treats as "award nothing") rather
    than resolve a made/missed question from corrupted arithmetic."""
    hand = _contracted_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    # Seat 2 never appears; 1 and 3 get double-read until the trick force-closes.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 4)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(2, 2)))

    assert hand.scoring_disputed is True
    result = hand.final_result()
    assert result.marks_awarded_to not in (0, 1)
    assert result.marks == 0


def test_an_f3_early_close_is_not_mistaken_for_an_eighth_trick():
    """The F3/F4 interaction. An early close is a *real* trick boundary, so it
    counts toward the seven -- but a hand with one occluded tile still has
    exactly seven tricks and must not trip the 8th-trick refusal or look like a
    misdeal."""
    hand = _contracted_hand()
    trump_leads = [Tile.of(6, n) for n in (6, 5, 4, 3, 2, 1, 0)]
    fillers = [
        [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1)],
        [Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2)],
        [Tile.of(3, 0), Tile.of(3, 1), Tile.of(3, 3)],
        [Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2)],
        [Tile.of(4, 3), Tile.of(4, 4), Tile.of(5, 1)],
        [Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3)],
        [Tile.of(5, 4), Tile.of(5, 5), Tile.of(5, 0)],
    ]
    for i, (lead, (f1, f2, f3)) in enumerate(zip(trump_leads, fillers)):
        hand.play_tile(TilePlayed(player=0, tile=lead))
        hand.play_tile(TilePlayed(player=1, tile=f1))
        hand.play_tile(TilePlayed(player=2, tile=f2))
        if i == 0:
            continue  # seat 3's very first tile is occluded
        hand.play_tile(TilePlayed(player=3, tile=f3))

    assert len(hand.tricks) == 7
    assert hand.tricks[0].closed_early_for_new_trick is True
    assert hand.misdeal_suspected is False
    assert IrregularityKind.TRICK_COUNT_EXCEEDED not in _kinds(hand)
    # Seat 0 still swept every trick; the score is not corrupted by the boundary.
    assert all(t.winner == 0 for t in hand.tricks)
    assert hand.final_result().made is True


def test_a_whole_extra_trick_after_seven_is_refused_not_scored():
    """F4. Without a bound, an 8th trick was accepted outright: a hand capped at
    42 computed 65 points, and a team that swept every trick and won by 35 came
    back ``made=False``."""
    hand = _contracted_hand()
    trump_leads = [Tile.of(6, n) for n in (6, 5, 4, 3, 2, 1, 0)]
    fillers = [
        [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1)],
        [Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2)],
        [Tile.of(3, 0), Tile.of(3, 1), Tile.of(3, 3)],
        [Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2)],
        [Tile.of(4, 3), Tile.of(4, 4), Tile.of(5, 1)],
        [Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3)],
        [Tile.of(5, 4), Tile.of(5, 5), Tile.of(5, 0)],
    ]
    for lead, (f1, f2, f3) in zip(trump_leads, fillers):
        hand.play_tile(TilePlayed(player=0, tile=lead))
        hand.play_tile(TilePlayed(player=1, tile=f1))
        hand.play_tile(TilePlayed(player=2, tile=f2))
        hand.play_tile(TilePlayed(player=3, tile=f3))
    assert len(hand.tricks) == 7

    # A whole spurious extra trick's worth of reads arrives afterwards.
    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(5, 5)), (2, Tile.of(4, 4)), (3, Tile.of(3, 3))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert len(hand.tricks) == 7
    assert hand.scorer.points_claimed_total == 42  # not 65
    result = hand.final_result()
    assert result.made is True
    assert result.bidding_team_points == 42
    assert IrregularityKind.TRICK_COUNT_EXCEEDED in _kinds(hand)


def test_a_duplicate_tile_read_does_not_erase_the_first_attribution():
    """F5. ``_seen_tiles`` is keyed by tile identity, so a later misread of a
    tile another seat already played silently reassigned it -- feeding the deal
    sampler a tile that is physically face-up on the table."""
    hand = _contracted_hand()
    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0)), (3, Tile.of(1, 0))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 5)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(3, 0)))  # misread duplicate

    assert hand._seen_tiles[Tile.of(3, 0)] == 1  # first writer wins
    assert tile_hold_probability(hand, Tile.of(3, 0), 1) == 1.0
    assert tile_hold_probability(hand, Tile.of(3, 0), 2) == 0.0
    assert hand.repairs.has_conflicts  # the conflict is logged, not silent


def test_a_duplicate_read_hand_still_returns_a_probability_estimate():
    """F5, the regression guard the plan-critic proved is needed: deriving
    ``remaining_hand_size`` from ``_plays_by_seat`` instead would make
    ``sum(sizes) != len(unseen)`` on every duplicate-read hand and blank both
    estimators where they currently still work."""
    hand = _contracted_hand()
    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(3, 0)), (2, Tile.of(2, 0)), (3, Tile.of(1, 0))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 5)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(3, 0)))

    sizes = {p: hand.remaining_hand_size(p) for p in range(4)}
    assert sum(sizes.values()) == 28 - len(hand.played_tiles)
    assert estimate_bid_probability(hand, samples=40) is not None


# ---------------------------------------------------------------------------
# splash/plunge inference from the first trick's actual leader
# ---------------------------------------------------------------------------

def test_partner_leading_a_marks_2_contract_infers_splash():
    hand = HandState(dealer=3)
    hand.bid(0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    # Seat 2 (seat 0's partner) leads instead of seat 0 -- the splash/plunge
    # shape, not an out-of-turn play.
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))

    assert hand.contract.kind == BidKind.SPLASH
    assert hand.contract.trump_caller == 2
    assert hand.tricks[0].leader == 2
    assert IrregularityKind.OUT_OF_TURN not in _kinds(hand)
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED in _kinds(hand)


def test_partner_leading_a_marks_1_contract_stays_out_of_turn():
    """Below SPLASH_MIN_MARKS no valid splash/plunge bid exists, so this is a
    genuine out-of-turn play, not evidence of anything else."""
    hand = HandState(dealer=3)
    hand.bid(0, marks=1)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))

    assert hand.contract.kind == BidKind.MARKS
    assert IrregularityKind.OUT_OF_TURN in _kinds(hand)
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED not in _kinds(hand)


def test_bidder_leading_a_marks_contract_is_unaffected():
    hand = HandState(dealer=3)
    hand.bid(0, marks=4)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))

    assert hand.contract.kind == BidKind.MARKS
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED not in _kinds(hand)


def test_an_unrelated_seat_leading_a_marks_2_contract_stays_out_of_turn():
    """Only the bidder's partner leading first is splash/plunge-shaped -- any
    other seat is still a genuine out-of-turn play."""
    hand = HandState(dealer=3)
    hand.bid(0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    # Seat 1 (unrelated to bidder 0 or partner 2) leads first.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(3, 0)))

    assert hand.contract.kind == BidKind.MARKS
    assert IrregularityKind.OUT_OF_TURN in _kinds(hand)
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED not in _kinds(hand)


# ---------------------------------------------------------------------------
# retroactive revoke detection
# ---------------------------------------------------------------------------

def _play_trick(hand: HandState, plays) -> None:
    for player, tile in plays:
        hand.play_tile(TilePlayed(player=player, tile=tile))


def test_retroactive_revoke_is_detected_at_hand_end():
    """Seat 1 fails to follow trick 1's led suit (3) despite holding several
    suit-3 tiles, which it reveals across later tricks -- a hard contradiction
    only checkable once the whole hand's plays are known."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    _play_trick(hand, [(0, Tile.of(3, 1)), (1, Tile.of(2, 0)), (2, Tile.of(3, 0)), (3, Tile.of(4, 0))])
    _play_trick(hand, [(0, Tile.of(6, 6)), (1, Tile.of(3, 3)), (2, Tile.of(6, 0)), (3, Tile.of(0, 0))])
    _play_trick(hand, [(0, Tile.of(6, 5)), (1, Tile.of(3, 2)), (2, Tile.of(4, 2)), (3, Tile.of(2, 1))])
    _play_trick(hand, [(0, Tile.of(6, 4)), (1, Tile.of(4, 3)), (2, Tile.of(4, 4)), (3, Tile.of(2, 2))])
    _play_trick(hand, [(0, Tile.of(6, 3)), (1, Tile.of(5, 3)), (2, Tile.of(5, 1)), (3, Tile.of(4, 1))])
    _play_trick(hand, [(0, Tile.of(6, 2)), (1, Tile.of(1, 0)), (2, Tile.of(5, 2)), (3, Tile.of(5, 0))])
    _play_trick(hand, [(0, Tile.of(6, 1)), (1, Tile.of(1, 1)), (2, Tile.of(5, 5)), (3, Tile.of(5, 4))])

    assert len(hand.tricks) == 7
    hand.final_result()  # triggers retroactive revoke detection

    revokes = [i for i in hand.repairs.irregularities if i.kind == IrregularityKind.REVOKE]
    assert len(revokes) == 1
    assert revokes[0].player == 1
    assert "trick 1" in revokes[0].reason


def test_a_genuine_void_is_not_flagged_as_a_revoke():
    """The control for the test above: seat 1 doesn't follow trick 1's led
    suit, but never holds a suit-3 tile at all -- a real void, not a revoke."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Seat 1 never holds a suit-3 tile at any point -- its off-suit play in
    # trick 1 is a real void, not a revoke. Seat 2 (which did follow suit 3
    # in trick 1) absorbs the deal's other suit-3 tiles instead, which is
    # harmless: only a non-follower is ever checked.
    _play_trick(hand, [(0, Tile.of(3, 1)), (1, Tile.of(2, 0)), (2, Tile.of(3, 0)), (3, Tile.of(4, 0))])
    _play_trick(hand, [(0, Tile.of(6, 6)), (1, Tile.of(1, 0)), (2, Tile.of(6, 0)), (3, Tile.of(0, 0))])
    _play_trick(hand, [(0, Tile.of(6, 5)), (1, Tile.of(1, 1)), (2, Tile.of(3, 3)), (3, Tile.of(2, 1))])
    _play_trick(hand, [(0, Tile.of(6, 4)), (1, Tile.of(4, 2)), (2, Tile.of(3, 2)), (3, Tile.of(2, 2))])
    _play_trick(hand, [(0, Tile.of(6, 3)), (1, Tile.of(4, 4)), (2, Tile.of(4, 3)), (3, Tile.of(4, 1))])
    _play_trick(hand, [(0, Tile.of(6, 2)), (1, Tile.of(5, 1)), (2, Tile.of(5, 3)), (3, Tile.of(5, 0))])
    _play_trick(hand, [(0, Tile.of(6, 1)), (1, Tile.of(5, 2)), (2, Tile.of(5, 5)), (3, Tile.of(5, 4))])

    hand.final_result()
    revokes = [i for i in hand.repairs.irregularities if i.kind == IrregularityKind.REVOKE]
    assert revokes == []


def test_revoke_detection_skipped_when_trump_never_confirmed():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    # No call_trump(): trump is only ever a best-guess, never confirmed.

    _play_trick(hand, [(0, Tile.of(3, 1)), (1, Tile.of(2, 0)), (2, Tile.of(3, 0)), (3, Tile.of(4, 0))])
    _play_trick(hand, [(0, Tile.of(6, 6)), (1, Tile.of(3, 3)), (2, Tile.of(6, 0)), (3, Tile.of(0, 0))])
    _play_trick(hand, [(0, Tile.of(6, 5)), (1, Tile.of(3, 2)), (2, Tile.of(4, 2)), (3, Tile.of(2, 1))])
    _play_trick(hand, [(0, Tile.of(6, 4)), (1, Tile.of(4, 3)), (2, Tile.of(4, 4)), (3, Tile.of(2, 2))])
    _play_trick(hand, [(0, Tile.of(6, 3)), (1, Tile.of(5, 3)), (2, Tile.of(5, 1)), (3, Tile.of(4, 1))])
    _play_trick(hand, [(0, Tile.of(6, 2)), (1, Tile.of(1, 0)), (2, Tile.of(5, 2)), (3, Tile.of(5, 0))])
    _play_trick(hand, [(0, Tile.of(6, 1)), (1, Tile.of(1, 1)), (2, Tile.of(5, 5)), (3, Tile.of(5, 4))])

    hand.final_result()
    assert IrregularityKind.REVOKE not in _kinds(hand)


def test_revoke_detection_skipped_for_a_force_closed_hand():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Same trick-1 revoke shape as the positive test above...
    _play_trick(hand, [(0, Tile.of(3, 1)), (1, Tile.of(2, 0)), (2, Tile.of(3, 0)), (3, Tile.of(4, 0))])
    _play_trick(hand, [(0, Tile.of(6, 6)), (1, Tile.of(3, 3)), (2, Tile.of(6, 0)), (3, Tile.of(0, 0))])
    _play_trick(hand, [(0, Tile.of(6, 5)), (1, Tile.of(3, 2)), (2, Tile.of(4, 2)), (3, Tile.of(2, 1))])
    _play_trick(hand, [(0, Tile.of(6, 4)), (1, Tile.of(4, 3)), (2, Tile.of(4, 4)), (3, Tile.of(2, 2))])
    _play_trick(hand, [(0, Tile.of(6, 3)), (1, Tile.of(5, 3)), (2, Tile.of(5, 1)), (3, Tile.of(4, 1))])
    # ...but the 6th trick force-closes short (seat 2 never appears).
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 2)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(1, 1)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(5, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(5, 4)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(4, 1)))  # push past FORCE_CLOSE_AFTER

    assert hand.tricks[-1].force_closed is True
    # A 7th trick still opens (seat 2's remaining tiles, seat 0's last trump);
    # feed it so the hand actually reaches 7 tricks and final_result() doesn't
    # raise for an incomplete contract.
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 1)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(5, 2)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(5, 5)))

    hand.final_result()
    assert IrregularityKind.REVOKE not in _kinds(hand)


def test_partner_calling_trump_before_any_play_infers_splash():
    """The realistic table order: trump is called right after bidding
    closes, before the first tile is played. The lead-based inference alone
    (test above) doesn't cover this -- call_trump() must also recognize the
    partner as evidence, or it logs a false TRUMP_CALLED_BY_WRONG_SEAT."""
    hand = HandState(dealer=3)
    hand.bid(0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    hand.call_trump(2, trump=6)  # seat 0's partner calls trump

    assert hand.contract.kind == BidKind.SPLASH
    assert hand.contract.trump_caller == 2
    assert hand.trump_tracker.confirmed == 6
    assert IrregularityKind.TRUMP_CALLED_BY_WRONG_SEAT not in _kinds(hand)
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED in _kinds(hand)

    # The first trick led by that same partner is then unaffected too.
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))
    assert hand.tricks[0].leader == 2
    assert IrregularityKind.OUT_OF_TURN not in _kinds(hand)


def test_retroactive_revoke_corrects_the_false_void_it_supersedes():
    """`_record_voids` runs live and, not yet knowing this was a revoke,
    records seat 1 as void in suit 3 -- a false constraint per its own
    docstring. Once `_detect_revokes` proves otherwise, that false void must
    be removed, or engine.probability's deal sampler keeps using it."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    _play_trick(hand, [(0, Tile.of(3, 1)), (1, Tile.of(2, 0)), (2, Tile.of(3, 0)), (3, Tile.of(4, 0))])
    assert 3 in hand.voids[1]  # recorded live, before it's known to be a revoke

    _play_trick(hand, [(0, Tile.of(6, 6)), (1, Tile.of(3, 3)), (2, Tile.of(6, 0)), (3, Tile.of(0, 0))])
    _play_trick(hand, [(0, Tile.of(6, 5)), (1, Tile.of(3, 2)), (2, Tile.of(4, 2)), (3, Tile.of(2, 1))])
    _play_trick(hand, [(0, Tile.of(6, 4)), (1, Tile.of(4, 3)), (2, Tile.of(4, 4)), (3, Tile.of(2, 2))])
    _play_trick(hand, [(0, Tile.of(6, 3)), (1, Tile.of(5, 3)), (2, Tile.of(5, 1)), (3, Tile.of(4, 1))])
    _play_trick(hand, [(0, Tile.of(6, 2)), (1, Tile.of(1, 0)), (2, Tile.of(5, 2)), (3, Tile.of(5, 0))])
    _play_trick(hand, [(0, Tile.of(6, 1)), (1, Tile.of(1, 1)), (2, Tile.of(5, 5)), (3, Tile.of(5, 4))])

    hand.final_result()

    assert 3 not in hand.voids[1]  # the false void is corrected once the revoke is known
    revokes = [i for i in hand.repairs.irregularities if i.kind == IrregularityKind.REVOKE]
    assert len(revokes) == 1 and revokes[0].player == 1


def test_stray_mid_hand_trump_call_does_not_retroactively_infer_splash():
    """A correctly-attributed plain marks contract, played normally, must not
    be reclassified by a stray/misheard call_trump arriving mid-hand -- only
    a call before the hand's first tile is played is splash/plunge evidence."""
    hand = HandState(dealer=3)
    hand.bid(0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)  # correct bidder calls trump normally

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))

    # A stray/misheard call, mid-trick, attributed to the bidder's partner.
    hand.call_trump(2, trump=3)

    assert hand.contract.kind == BidKind.MARKS  # unchanged, not reclassified
    assert hand.contract.trump_caller == 0
    assert hand.trump_tracker.confirmed == 6  # NOT overwritten by the stray call
    assert IrregularityKind.TRUMP_CALLED_BY_WRONG_SEAT in _kinds(hand)
    assert IrregularityKind.SPLASH_PLUNGE_INFERRED not in _kinds(hand)
