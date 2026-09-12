from __future__ import annotations

import pytest

from eye42.engine.bidding import BidKind, BiddingError, BiddingRound, Contract, partner_of
from eye42.engine.events import IrregularEndSignal, TilePlayed
from eye42.engine.game import GameState
from eye42.engine.hand import HandOutcomeKind, HandState
from eye42.engine.scoring import HandScoreTracker
from eye42.engine.tiles import Tile, effective_suit, is_trump, suits_of
from eye42.engine.trick import Trick
from eye42.engine.trump_inference import TrumpHypothesisTracker


# ---------------------------------------------------------------------------
# tile / suit primitives
# ---------------------------------------------------------------------------

def test_suits_of_and_trump():
    assert suits_of(Tile.of(4, 3)) == {4, 3}
    assert suits_of(Tile.of(5, 5)) == {5}
    assert is_trump(Tile.of(4, 3), trump=3) is True
    assert is_trump(Tile.of(4, 3), trump=6) is False


def test_effective_suit_prefers_trump_over_led():
    # 6-4 with trump=6 is a trump tile, even if 4 was led.
    assert effective_suit(Tile.of(6, 4), trump=6, led_suit=4) == 6
    # 4-3 with trump=6, suit 4 led -> follows on the 4 end.
    assert effective_suit(Tile.of(4, 3), trump=6, led_suit=4) == 4
    # 4-3 with trump=6, suit 3 led -> follows on the 3 end.
    assert effective_suit(Tile.of(4, 3), trump=6, led_suit=3) == 3


def test_trick_winner_double_trump_beats_all():
    trick = Trick(leader=0, trump=6)
    trick.play(0, Tile.of(6, 6))
    trick.play(1, Tile.of(5, 0))
    trick.play(2, Tile.of(6, 2))
    trick.play(3, Tile.of(4, 1))
    assert trick.winner == 0
    assert trick.count_value == 10  # 5-0 (5) + 4-1 (5)


# ---------------------------------------------------------------------------
# bidding
# ---------------------------------------------------------------------------

def test_all_pass_forces_dealer_to_bid_thirty():
    round_ = BiddingRound(dealer=2)
    for seat in (3, 0, 1, 2):
        round_.record_pass(_pass(seat))
    assert round_.forced_thirty is True
    assert round_.contract == Contract(bidder=2, kind=BidKind.POINTS, amount=30)


def test_bid_must_exceed_current_high_bid():
    round_ = BiddingRound(dealer=3)
    round_.record_bid(_bid(0, 30))
    with pytest.raises(BiddingError):
        round_.record_bid(_bid(1, 30))


def test_splash_upgrade_requires_minimum_marks():
    round_ = BiddingRound(dealer=3)
    round_.record_bid(_bid(0, 0, marks=1))
    round_.record_pass(_pass(1))
    round_.record_pass(_pass(2))
    round_.record_pass(_pass(3))
    with pytest.raises(BiddingError):
        round_.upgrade_to_splash_or_plunge(BidKind.SPLASH)  # only 1 mark bid, needs 2


def test_splash_contract_partner_calls_trump_and_leads():
    contract = Contract(bidder=1, kind=BidKind.SPLASH, amount=2)
    assert contract.trump_caller == partner_of(1) == 3
    assert contract.requires_sweep is True
    assert contract.points_needed == 42


def _bid(player: int, amount: int, marks: int = 0):
    from eye42.engine.events import BidMade

    return BidMade(player=player, amount=amount, marks=marks)


def _pass(player: int):
    from eye42.engine.events import Passed

    return Passed(player=player)


# ---------------------------------------------------------------------------
# scoring / set detection
# ---------------------------------------------------------------------------

def test_locked_set_fires_before_hand_completes_on_points_bid():
    contract = Contract(bidder=0, kind=BidKind.POINTS, amount=35)
    tracker = HandScoreTracker(contract=contract)

    trick_a = Trick(leader=0, trump=6)
    trick_a.play(0, Tile.of(1, 0))
    trick_a.play(1, Tile.of(6, 4))  # trump + 10-count, seat1 (team1) wins
    trick_a.play(2, Tile.of(2, 0))
    trick_a.play(3, Tile.of(3, 0))
    assert trick_a.winner == 1
    tracker.record_trick(trick_a)

    trick_b = Trick(leader=1, trump=6)
    trick_b.play(1, Tile.of(5, 5))  # double, 10-count, seat1 wins again
    trick_b.play(2, Tile.of(5, 0))  # also a 5-count tile, goes to the trick winner
    trick_b.play(3, Tile.of(5, 1))
    trick_b.play(0, Tile.of(5, 2))
    assert trick_b.winner == 1
    tracker.record_trick(trick_b)

    assert tracker.defending_team_points == 27  # (1+10) + (1+10+5)
    assert tracker.bidding_team_points == 0
    # Only 2 of 7 tricks played, but 35 - 0 = 35 > (42 - 27) = 15 remaining.
    assert tracker.is_locked_set is True


def test_sweep_contract_locks_immediately_on_any_lost_trick():
    contract = Contract(bidder=1, kind=BidKind.SPLASH, amount=2)
    tracker = HandScoreTracker(contract=contract)

    trick = Trick(leader=3, trump=6)
    trick.play(3, Tile.of(1, 0))
    trick.play(0, Tile.of(6, 6))  # seat0 (team0, defenders) trumps in and wins
    trick.play(1, Tile.of(2, 0))
    trick.play(2, Tile.of(3, 0))
    assert trick.winner == 0
    tracker.record_trick(trick)

    # Only 1 of 7 tricks played, but a sweep contract dies the instant one is lost.
    assert tracker.is_locked_set is True


# ---------------------------------------------------------------------------
# full hand via HandState
# ---------------------------------------------------------------------------

def test_full_hand_made_when_bidding_team_sweeps():
    hand = HandState(dealer=3)  # rotation starts at seat 0
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    assert hand.contract == Contract(bidder=0, kind=BidKind.POINTS, amount=30)

    hand.call_trump(0, trump=6)

    # Seat 0 holds and leads all 7 trumps, winning every trick outright; the
    # other three seats' tiles (all 21 non-trump tiles, including all 5 count
    # tiles) are irrelevant to who wins but must still sum to a real 28-tile hand.
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
    assert all(t.winner == 0 for t in hand.tricks)
    result = hand.final_result()
    assert result.made is True
    assert result.bidding_team_points == 42
    assert result.marks_awarded_to == 0
    assert result.marks == 1


# ---------------------------------------------------------------------------
# redeal / concession classification
# ---------------------------------------------------------------------------

def test_redeal_does_not_advance_dealer():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=0))
    assert hand.outcome_kind == HandOutcomeKind.REDEAL

    game.record_hand(hand)
    assert game.dealer == 1  # unchanged
    assert game.marks == [0, 0]


def test_concession_mid_hand_advances_dealer_and_scores():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.call_trump(2, trump=6)

    trick = Trick(leader=2, trump=6)
    trick.play(2, Tile.of(6, 6))
    trick.play(3, Tile.of(1, 0))
    trick.play(0, Tile.of(2, 0))
    trick.play(1, Tile.of(3, 0))
    hand.scorer.record_trick(trick)
    hand.tricks.append(trick)

    outcome = hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=1))
    assert outcome == HandOutcomeKind.CONCESSION

    game.record_hand(hand)
    assert game.dealer == 2  # advanced past the old dealer (seat 1 -> seat 2)


# ---------------------------------------------------------------------------
# trump hypothesis tracker
# ---------------------------------------------------------------------------

def test_explicit_trump_call_confirms_immediately():
    tracker = TrumpHypothesisTracker()
    tracker.confirm(5)
    assert tracker.is_confirmed is True
    assert tracker.best_guess == 5
    assert tracker.is_ambiguous is False


def test_unstated_trump_on_double_lead_is_strong_but_unconfirmed():
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(5, 5), cue=None)
    assert tracker.is_confirmed is False
    assert tracker.best_guess == 5
    assert tracker.weights[5] > 0.8


def test_weak_cue_on_nondouble_lead_hard_resolves():
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(4, 5), cue="low")
    assert tracker.is_confirmed is True
    assert tracker.best_guess == 4


def test_ambiguous_nondouble_lead_narrows_on_high_confidence_contradiction():
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(4, 2), cue=None)
    assert tracker.is_ambiguous is True
    assert set(n for n, w in tracker.weights.items() if w > 0.2) == {2, 4}

    # A confident contradiction against 2 should shift the best guess to 4,
    # even without hard-confirming it (a single contradiction isn't decisive).
    tracker.observe_contradiction({2}, confidence=1.0)
    assert tracker.best_guess == 4
    assert tracker.is_confirmed is False


def test_low_confidence_contradiction_only_nudges_weight():
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(4, 2), cue=None)
    before = tracker.weights[2]
    tracker.observe_contradiction({2}, confidence=0.3)  # a shaky tile read
    after = tracker.weights[2]
    assert after < before
    assert after > before * 0.4  # much gentler than the high-confidence case


def test_hand_reconciles_apparent_illegal_play_instead_of_crashing():
    """When we know players' hands and a play looks illegal under the current
    best-guess trump, the engine must treat that as evidence the guess is wrong
    (real players can't illegally revoke) rather than raising."""
    hand = HandState(dealer=3, hands={
        0: [Tile.of(4, 2), Tile.of(6, 6)],
        1: [Tile.of(1, 0), Tile.of(3, 2)],
        2: [Tile.of(0, 0), Tile.of(5, 1)],
        3: [Tile.of(6, 1), Tile.of(5, 3)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    # No explicit trump call: trump must be inferred from the lead alone.

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(4, 2)))
    # seat 1 has no tile matching the *actual* led suit under some guesses and
    # will look like an illegal revoke under at least one candidate; this must
    # not raise IllegalPlayError.
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=2, tile=Tile.of(0, 0)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(6, 1)))

    assert len(hand.tricks) == 1  # completed without raising
    assert hand.trump_tracker.weights[2] < 1 / 7  # original ambiguous share
