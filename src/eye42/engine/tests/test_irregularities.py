from __future__ import annotations

from eye42.engine.events import IrregularEndSignal, TilePlayed, TilesDealt
from eye42.engine.hand import HandOutcomeKind, HandState
from eye42.engine.probability import estimate_bid_probability, tile_hold_probability
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
# revoke (confirmed trump)
# ---------------------------------------------------------------------------

def _confirmed_trump_hand():
    """A full, valid 28-tile deal where trump (6) is spread across all four
    seats, so any seat other than the leader can genuinely hold and withhold
    a trump follower -- unlike a hand where one seat holds the entire trump
    suit, which makes a real revoke by anyone else impossible to construct."""
    hand = HandState(dealer=3, hands={
        0: [Tile.of(6, 6), Tile.of(4, 4), Tile.of(3, 3), Tile.of(2, 2), Tile.of(1, 1), Tile.of(0, 0), Tile.of(5, 5)],
        1: [Tile.of(6, 1), Tile.of(4, 3), Tile.of(4, 2), Tile.of(4, 1), Tile.of(4, 0), Tile.of(3, 2), Tile.of(3, 1)],
        2: [Tile.of(6, 2), Tile.of(3, 0), Tile.of(2, 1), Tile.of(2, 0), Tile.of(1, 0), Tile.of(5, 0), Tile.of(5, 1)],
        3: [Tile.of(6, 3), Tile.of(5, 2), Tile.of(5, 3), Tile.of(5, 4), Tile.of(6, 4), Tile.of(6, 5), Tile.of(6, 0)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)
    return hand


def test_confirmed_trump_revoke_is_logged_and_play_continues():
    hand = _confirmed_trump_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    # Seat 1 holds a trump (6-1) but plays off-suit instead -- a real revoke.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 3)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 2)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(6, 3)))

    assert len(hand.tricks) == 1
    assert hand.disputed is True
    assert IrregularityKind.REVOKE in _kinds(hand)


def test_revoke_does_not_record_a_false_void():
    hand = _confirmed_trump_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 3)))  # revoke: holds 6-1
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 2)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(6, 3)))

    assert IrregularityKind.REVOKE in _kinds(hand)
    assert 6 not in hand.voids[1]  # no false void recorded for the revoking seat


def test_revoke_does_not_crash_probability_sampler():
    hand = _confirmed_trump_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(4, 3)))  # revoke: holds 6-1 trump
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 2)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(6, 3)))

    assert hand.disputed is True
    result = estimate_bid_probability(hand)
    assert result is None  # suppressed, not a crash

    assert tile_hold_probability(hand, Tile.of(5, 5), player=0) is None


# ---------------------------------------------------------------------------
# tile not in hand
# ---------------------------------------------------------------------------

def test_played_tile_not_in_known_hand_does_not_crash():
    hand = HandState(dealer=3, hands={
        0: [Tile.of(6, 6)],
        1: [Tile.of(1, 0)],
        2: [Tile.of(2, 0)],
        3: [Tile.of(3, 0)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    # Seat 1 "plays" a tile it isn't recorded as holding.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 5)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(3, 0)))

    assert len(hand.tricks) == 1
    assert hand.disputed is True
    assert IrregularityKind.TILE_NOT_IN_HAND in _kinds(hand)


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

    assert hand.misdeal_suspected is True
    assert IrregularityKind.SEAT_TILE_COUNT_ANOMALY in _kinds(hand)
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
