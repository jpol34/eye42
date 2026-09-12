from __future__ import annotations

import pytest

from eye42.engine.bidding import BidKind, BiddingError, BiddingRound, Contract, partner_of
from eye42.engine.events import IrregularEndSignal, TilePlayed
from eye42.engine.game import GameState
from eye42.engine.hand import HandOutcomeKind, HandState
from eye42.engine.probability import estimate_bid_probability, tile_hold_probability
from eye42.engine.repair import IrregularityKind
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


# ---------------------------------------------------------------------------
# trump-hypothesis reversibility (hard vs. soft confirmation)
# ---------------------------------------------------------------------------

def _soft_confirmed_tracker() -> TrumpHypothesisTracker:
    """Drive the tracker to an *inferred* confirmation: a led double biases
    heavily toward one candidate, then a confident contradiction against every
    other candidate pushes it past CONFIRMED_THRESHOLD on its own."""
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(5, 5), cue=None)
    tracker.observe_contradiction({0, 1, 2, 3, 4, 6}, confidence=1.0)
    assert tracker.confirmed == 5
    assert tracker.is_soft_confirmed is True
    return tracker


def test_soft_confirmation_leaves_the_weight_distribution_intact():
    """Unlike a hard confirm, a soft one must not collapse to one-hot -- the
    distribution is what tracking resumes from if it gets reopened, so there is
    nothing to save and restore."""
    tracker = _soft_confirmed_tracker()
    assert tracker.hard_confirmed is False
    assert any(0.0 < w < 1.0 for w in tracker.weights.values())


def test_hard_confirmation_is_permanent():
    tracker = TrumpHypothesisTracker()
    tracker.confirm(5)
    assert tracker.hard_confirmed is True

    tracker.observe_contradiction({5}, confidence=1.0)
    assert tracker.confirmed == 5  # an explicit call is a fact, not a hypothesis
    assert tracker.hard_confirmed is True
    assert tracker.weights[5] == 1.0


def test_soft_confirmation_reopens_on_a_full_confidence_contradiction():
    tracker = _soft_confirmed_tracker()
    before = tracker.weights[5]
    tracker.observe_contradiction({5}, confidence=1.0)

    assert tracker.is_confirmed is False  # reopened, back to weighted tracking
    assert tracker.hard_confirmed is False
    assert tracker.ever_confirmed is True  # it *had* a trump; this isn't bidding
    assert tracker.is_ambiguous is True
    # The contradiction is applied on the way out, but a single one is still not
    # decisive (the project's standing rule) -- 5 stays the leading candidate.
    assert tracker.weights[5] < before


def test_low_confidence_contradiction_does_not_reopen_a_soft_confirmation():
    tracker = _soft_confirmed_tracker()
    tracker.observe_contradiction({5}, confidence=0.3)
    assert tracker.confirmed == 5  # a shaky read can't undo accumulated evidence


def test_contradiction_against_another_candidate_does_not_reopen():
    tracker = _soft_confirmed_tracker()
    tracker.observe_contradiction({2}, confidence=1.0)
    assert tracker.confirmed == 5  # only the confirmed candidate itself reopens


def _hand_that_soft_confirms_then_contradicts_itself() -> HandState:
    """End-to-end fixture for the reopening path, built entirely from real
    plays -- the point being that it is reachable from ``HandState``, not only
    from the tracker's own API.

    Trick 1 is ordinary and leaves trump unconfirmed. Weights are then pinned to
    a near-tie between 5 and 4 (standing in for whatever evidence the hand
    accumulated). In trick 2, seat 0 leads suit 3; seat 1's off-suit play is
    perfectly legal under the leading guess of 5; seat 2's is not, so
    ``_reconcile_trump_for_play`` switches the trick to 4 and that contradiction
    pushes 4 past the confirmation threshold -- a *soft* confirmation. At trick
    close, seat 1's play (made under the old guess, so never revoke-flagged) is
    re-evaluated under 4 and now contradicts it, which reopens the soft
    confirmation instead of being silently ignored forever.
    """
    hand = HandState(dealer=3, hands={
        0: [Tile.of(6, 6), Tile.of(3, 3), Tile.of(6, 5)],
        1: [Tile.of(0, 0), Tile.of(2, 0), Tile.of(5, 3)],
        2: [Tile.of(1, 0), Tile.of(2, 2), Tile.of(4, 3)],
        3: [Tile.of(2, 1), Tile.of(1, 1), Tile.of(0, 1)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    # No explicit trump call: everything below is inference.

    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(0, 0)), (2, Tile.of(1, 0)), (3, Tile.of(2, 1))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    assert hand.trump_tracker.is_confirmed is False

    hand.trump_tracker.weights = {n: 0.0 for n in range(7)}
    hand.trump_tracker.weights[5] = 0.51
    hand.trump_tracker.weights[4] = 0.49

    for player, tile in [
        (0, Tile.of(3, 3)), (1, Tile.of(2, 0)), (2, Tile.of(2, 2)), (3, Tile.of(1, 1))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    return hand


def test_soft_confirmation_is_reopened_by_real_play_at_trick_close():
    hand = _hand_that_soft_confirms_then_contradicts_itself()

    assert len(hand.tricks) == 2
    assert hand.tricks[1].trump == 4  # reconciled mid-trick, then soft-confirmed
    assert hand.trump_tracker.is_confirmed is False  # ...and reopened at close
    assert hand.trump_tracker.hard_confirmed is False
    assert hand.trump_tracker.ever_confirmed is True


def test_reopened_trump_suppresses_the_estimate_instead_of_raising():
    """A reopened confirmation is a real, valid, in-progress hand state -- the
    estimator must report "unavailable", never blow up. The ValueError is kept
    only for the genuine API-misuse case of asking before trump ever existed."""
    hand = _hand_that_soft_confirms_then_contradicts_itself()
    assert hand.trump_tracker.is_confirmed is False

    assert estimate_bid_probability(hand) is None
    # An exact answer that needs no trump at all still comes back normally.
    assert tile_hold_probability(hand, Tile.of(6, 5), player=0) == 1.0


def test_estimate_still_raises_when_trump_was_never_confirmed():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    with pytest.raises(ValueError):
        estimate_bid_probability(hand)


def test_voids_are_invalidated_when_the_trump_confirmation_changes():
    """Voids recorded under a rejected hypothesis are *false* constraints for
    the deal sampler. When the confirmed trump changes value, drop them."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(0, 0)), (2, Tile.of(1, 0)), (3, Tile.of(2, 1))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    assert hand.voids[1] == {6}

    # Stand in for "a soft confirmation was reopened and later re-confirmed
    # elsewhere": the tracker's confirmed value is now a different suit.
    hand.trump_tracker._soft_confirm(3)

    for player, tile in [
        (0, Tile.of(3, 3)), (1, Tile.of(3, 0)), (2, Tile.of(3, 1)), (3, Tile.of(3, 2))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert 6 not in hand.voids[1]  # the suit-6 voids were recorded under the old trump
    assert 6 not in hand.voids[2] and 6 not in hand.voids[3]


def test_voids_are_not_cleared_while_trump_is_merely_unconfirmed():
    """The invalidation hook sits after the is_confirmed gate on purpose -- put
    before it, it would wipe voids on every trick of an unconfirmed hand."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(0, 0)), (2, Tile.of(1, 0)), (3, Tile.of(2, 1))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    for player, tile in [
        (0, Tile.of(6, 5)), (1, Tile.of(1, 1)), (2, Tile.of(2, 0)), (3, Tile.of(2, 2))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert hand.voids[1] == {6}  # unchanged across a second trick at the same trump


# ---------------------------------------------------------------------------
# the reopen path end-to-end, and an honest account of how reachable it is
# ---------------------------------------------------------------------------

_REOPEN_HANDS = {
    0: [Tile.of(6, 6), Tile.of(3, 3), Tile.of(4, 2), Tile.of(6, 2)],
    1: [Tile.of(0, 0), Tile.of(2, 0), Tile.of(5, 3), Tile.of(5, 1)],
    2: [Tile.of(1, 0), Tile.of(2, 2), Tile.of(6, 4), Tile.of(4, 3)],
    3: [Tile.of(2, 1), Tile.of(1, 1), Tile.of(6, 0), Tile.of(5, 0)],
}
_REOPEN_TRICK_1 = [(0, Tile.of(6, 6)), (1, Tile.of(0, 0)), (2, Tile.of(1, 0)), (3, Tile.of(2, 1))]
_REOPEN_TRICK_2 = [(0, Tile.of(3, 3)), (1, Tile.of(2, 0)), (2, Tile.of(2, 2)), (3, Tile.of(1, 1))]
_REOPEN_TRICK_3 = [(0, Tile.of(6, 2)), (1, Tile.of(5, 3)), (2, Tile.of(6, 4)), (3, Tile.of(5, 0))]


def _play(hand: HandState, plays) -> None:
    for player, tile in plays:
        hand.play_tile(TilePlayed(player=player, tile=tile))


def test_full_reopen_cycle_runs_through_sequential_engine_calls():
    """soft-confirm -> contradicting real play at trick close -> reopen ->
    re-normalized weights -> re-confirm, driven entirely by ``play_tile``.

    Every state transition below is produced by the engine reacting to tiles
    hitting the table. The one thing handed to it is the *starting* weight
    distribution: a near-tie between 5 and 4, standing in for evidence the hand
    would have accumulated. That seeding is not cosmetic laziness -- see
    ``TrumpHypothesisTracker.is_soft_confirmed``: the tracker's cue priors cap
    at 0.85, contradictions only ever push the *leading* candidate down, and a
    search of 84,000 randomized legal playouts never got any candidate above
    0.75. A fully unseeded route to a soft confirmation does not currently
    exist, and this test should not pretend otherwise.
    """
    hand = HandState(dealer=3, hands={p: list(v) for p, v in _REOPEN_HANDS.items()})
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    # No explicit trump call anywhere: everything below is inference.

    _play(hand, _REOPEN_TRICK_1)
    assert hand.trump_tracker.is_confirmed is False
    assert hand.trump_tracker.ever_confirmed is False

    hand.trump_tracker.weights = {n: 0.0 for n in range(7)}
    hand.trump_tracker.weights[5] = 0.51
    hand.trump_tracker.weights[4] = 0.49

    # Trick 2. Seat 2's play is illegal under the leading guess of 5 but fine
    # under 4, so _reconcile_trump_for_play switches the trick and contradicts
    # 5 -- which alone carries 4 over CONFIRMED_THRESHOLD: a soft confirmation.
    _play(hand, _REOPEN_TRICK_2[:3])
    assert hand.trump_tracker.confirmed == 4
    assert hand.trump_tracker.is_soft_confirmed is True
    assert hand.trump_tracker.weights[4] == pytest.approx(0.9505, abs=1e-3)

    # ...and at the close of that same trick, seat 1's earlier play (made under
    # the old guess, so never revoke-flagged) is re-evaluated under 4 and now
    # contradicts it. That reopens the soft confirmation rather than being
    # silently ignored forever.
    _play(hand, _REOPEN_TRICK_2[3:])
    assert len(hand.tricks) == 2
    assert hand.tricks[1].trump == 4
    assert hand.trump_tracker.is_confirmed is False  # reopened
    assert hand.trump_tracker.hard_confirmed is False
    assert hand.trump_tracker.ever_confirmed is True  # it *had* one; not bidding
    # Reopening resumes weighted tracking from a properly re-normalized
    # distribution -- not a collapsed one-hot, and not an unnormalized remnant.
    assert sum(hand.trump_tracker.weights.values()) == pytest.approx(1.0)
    assert hand.trump_tracker.weights[5] == pytest.approx(0.51, abs=1e-3)
    assert hand.trump_tracker.weights[4] == pytest.approx(0.49, abs=1e-3)
    assert estimate_bid_probability(hand) is None  # unavailable while reopened

    # Trick 3 brings fresh contradicting play against 5, and 4 -- challenged,
    # penalized, and still the better hypothesis -- is confirmed again.
    _play(hand, _REOPEN_TRICK_3)
    assert len(hand.tricks) == 3
    assert hand.trump_tracker.confirmed == 4
    assert hand.trump_tracker.is_soft_confirmed is True
    assert hand.trump_tracker.weights[4] == pytest.approx(0.9505, abs=1e-3)
    # Voids recorded at that close are stamped with the trump actually in force.
    assert hand._voids_trump == 4
    assert hand.voids[1] == {6}


def test_the_reopen_path_is_unreachable_without_known_hands():
    """Honest scope marker, not an aspiration: ``_check_trump_contradiction``
    is the only thing that can reopen a soft confirmation or invalidate voids,
    and it early-returns whenever ``hand.hands is None`` -- which is every
    production hand until Phase 2 perception can read a real deal. The exact
    same tile sequence therefore does nothing at all here.

    If a future change makes perception populate ``hands``, this test failing is
    the intended signal that the feature just became live and its behavior now
    needs real-play validation rather than test-only coverage.
    """
    hand = HandState(dealer=3)  # no known hands: the production shape today
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    _play(hand, _REOPEN_TRICK_1)
    hand.trump_tracker.weights = {n: 0.0 for n in range(7)}
    hand.trump_tracker.weights[5] = 0.51
    hand.trump_tracker.weights[4] = 0.49
    _play(hand, _REOPEN_TRICK_2)
    _play(hand, _REOPEN_TRICK_3)

    assert len(hand.tricks) == 3
    assert hand.trump_tracker.ever_confirmed is False  # never even soft-confirmed
    assert hand.trump_tracker.weights[5] == pytest.approx(0.51)
    assert hand.trump_tracker.weights[4] == pytest.approx(0.49)


# ---------------------------------------------------------------------------
# reopen -> immediate re-confirmation: the boundary, pinned
# ---------------------------------------------------------------------------

def _tracker_soft_confirmed_at(weight: float, trump: int = 5) -> TrumpHypothesisTracker:
    tracker = TrumpHypothesisTracker()
    rest = (1.0 - weight) / 6
    tracker.weights = {n: (weight if n == trump else rest) for n in range(7)}
    tracker._soft_confirm(trump)
    return tracker


def test_a_reopen_below_the_boundary_actually_takes_effect():
    """The ordinary case: a full-confidence contradiction reopens and the
    post-penalty distribution is nowhere near peaked enough to re-confirm, so
    the reopen is real rather than a silent no-op."""
    tracker = _tracker_soft_confirmed_at(0.99)
    tracker.observe_contradiction({5}, confidence=1.0)
    assert tracker.is_confirmed is False
    assert tracker.weights[5] == pytest.approx(0.8319, abs=1e-3)


def test_a_very_peaked_reopen_may_re_confirm_the_same_value_and_that_is_correct():
    """The boundary the review flagged. With a singleton contradiction -- the
    only shape ``engine.hand`` ever passes -- the confirmed candidate's
    post-penalty share is 0.05w / (0.05w + 1 - w), which only clears
    CONFIRMED_THRESHOLD above w ~= 0.9945. Re-confirming there is *not* a
    swallowed reopen: the hypothesis was challenged, took its full penalty, and
    still leads by a wide margin, which is what surviving contrary evidence
    looks like. It is also self-limiting -- the value comes back materially
    lower, so a second contradiction drops it straight out of confirmation.
    """
    tracker = _tracker_soft_confirmed_at(0.995)
    tracker.observe_contradiction({5}, confidence=1.0)
    assert tracker.confirmed == 5  # re-confirmed, deliberately
    assert tracker.is_soft_confirmed is True
    assert tracker.weights[5] == pytest.approx(0.9087, abs=1e-3)
    assert tracker.weights[5] < 0.995  # the evidence still cost it real weight

    tracker.observe_contradiction({5}, confidence=1.0)
    assert tracker.is_confirmed is False  # and the next one is decisive


def test_repeated_contradictions_do_not_oscillate_forever():
    """Whatever the starting peak, repeated identical contradictions converge
    downward -- there is no fixed point that keeps re-firing a reopen."""
    tracker = _tracker_soft_confirmed_at(0.999)
    for _ in range(6):
        tracker.observe_contradiction({5}, confidence=1.0)
    assert tracker.is_confirmed is False
    assert tracker.weights[5] < 0.01


def test_each_trick_is_evaluated_for_contradictions_exactly_once():
    """Why the boundary above is bounded in practice rather than a loop risk:
    ``_check_trump_contradiction`` runs from ``_close_trick``, a trick is
    closed exactly once, and tricks are finite -- so no piece of contradiction
    evidence is ever counted twice."""
    seen = []
    original = HandState._check_trump_contradiction

    def spy(self, trick):
        seen.append(id(trick))
        return original(self, trick)

    HandState._check_trump_contradiction = spy
    try:
        hand = HandState(dealer=3, hands={p: list(v) for p, v in _REOPEN_HANDS.items()})
        hand.bid(0, 30)
        hand.bid_pass(1)
        hand.bid_pass(2)
        hand.bid_pass(3)
        _play(hand, _REOPEN_TRICK_1 + _REOPEN_TRICK_2 + _REOPEN_TRICK_3)
    finally:
        HandState._check_trump_contradiction = original

    assert len(hand.tricks) == 3
    assert len(seen) == 3  # once per trick...
    assert len(set(seen)) == 3  # ...and never the same trick twice


# ---------------------------------------------------------------------------
# bug-hunt fix pass: marks-bid upgrade (F1 / F2) and non-strict bidding (F6)
# ---------------------------------------------------------------------------

def _marks_bid_hand(bidder_tiles, *, dealer: int = 3, bidder: int = 0):
    """A HandState whose seat ``bidder`` holds ``bidder_tiles``. Only the
    bidder's doubles matter to the upgrade, so the other three seats are filled
    with whatever is left over (the deal is still a real 28 tiles)."""
    from eye42.engine.tiles import full_set

    rest = sorted(full_set() - set(bidder_tiles), key=lambda t: (t.high, t.low))
    hands = {bidder: list(bidder_tiles)}
    others = [p for p in range(4) if p != bidder]
    for i, seat in enumerate(others):
        hands[seat] = rest[i * 7:(i + 1) * 7]
    return HandState(dealer=dealer, hands=hands)


_THREE_DOUBLES = [
    Tile.of(6, 6), Tile.of(5, 5), Tile.of(4, 4),
    Tile.of(6, 5), Tile.of(6, 4), Tile.of(6, 3), Tile.of(6, 2),
]
_FOUR_DOUBLES = [
    Tile.of(6, 6), Tile.of(5, 5), Tile.of(4, 4), Tile.of(3, 3),
    Tile.of(6, 5), Tile.of(6, 4), Tile.of(6, 3),
]


def test_one_mark_bid_by_a_three_doubles_hand_does_not_raise():
    """F1. Three doubles is the SPLASH doubles threshold, but SPLASH also needs
    two marks -- so a 1-mark bid has no upgrade available. The old code checked
    only the doubles count and let ``upgrade_to_splash_or_plunge`` raise
    ``BiddingError`` mid-``bid()``: after the contract was set, before the score
    tracker was built, leaving a contract with no scorer for the next trick to
    crash on."""
    hand = _marks_bid_hand(_THREE_DOUBLES)
    hand.bid(0, 0, marks=1)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    assert hand.contract == Contract(bidder=0, kind=BidKind.MARKS, amount=1)
    assert hand.scorer is not None  # the crash was a contract with no scorer
    assert hand.scorer.contract is hand.contract


def test_four_doubles_at_two_marks_is_a_splash_not_a_failed_plunge():
    """F1. Four doubles clears the PLUNGE doubles bar but 2 marks is below the
    PLUNGE marks floor -- this is a perfectly legal SPLASH. The fix must pick the
    highest *satisfiable* kind, not gate on the minimum and not raise."""
    hand = _marks_bid_hand(_FOUR_DOUBLES)
    hand.bid(0, 0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    assert hand.contract.kind == BidKind.SPLASH
    assert hand.contract.amount == 2
    assert hand.scorer.contract.kind == BidKind.SPLASH


def test_four_doubles_at_four_marks_is_a_plunge():
    hand = _marks_bid_hand(_FOUR_DOUBLES)
    hand.bid(0, 0, marks=4)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    assert hand.contract.kind == BidKind.PLUNGE
    assert hand.contract.trump_caller == partner_of(0)


def test_splash_upgrade_fires_when_three_passes_close_the_bidding():
    """F2. The canonical splash close is "marks bid, then three passes", which
    resolves the contract inside ``bid_pass``. The upgrade used to be wired only
    into ``bid()``, so this -- the normal case -- left a plain MARKS contract and
    handed trump/the first lead to the wrong seat."""
    hand = _marks_bid_hand(_THREE_DOUBLES)
    hand.bid(0, 0, marks=2)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)  # bidding closes here, inside bid_pass

    assert hand.contract.kind == BidKind.SPLASH
    assert hand.contract.trump_caller == partner_of(0) == 2
    # Contract is frozen and the upgrade replaces the object, so the tracker has
    # to be built from the upgraded one -- order is load-bearing here.
    assert hand.scorer.contract is hand.contract
    assert hand.scorer.contract.requires_sweep is True


def test_hand_bid_out_of_turn_is_logged_and_dropped_not_raised():
    """F6. ``BiddingRound`` stays strict; ``HandState.bid``/``bid_pass`` are the
    non-strict wrapper every other HandState event method already is."""
    hand = HandState(dealer=3)  # rotation is 0, 1, 2, 3
    hand.bid(2, 32)  # seat 2 bids out of turn

    assert hand.contract is None
    assert hand.bidding.current_bidder == 0  # the round never saw it
    kinds = {i.kind for i in hand.repairs.irregularities}
    assert IrregularityKind.BID_NOT_LEGAL in kinds

    # ...and the hand carries on normally from there.
    hand.bid(0, 30)
    hand.bid(1, 30)  # not above the high bid -- also dropped, not raised
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    assert hand.contract == Contract(bidder=0, kind=BidKind.POINTS, amount=30)


def test_bidding_after_the_contract_closed_is_dropped_not_raised():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    before = hand.scorer

    hand.bid(2, 41)
    hand.bid_pass(2)

    assert hand.contract == Contract(bidder=0, kind=BidKind.POINTS, amount=30)
    assert hand.scorer is before  # not rebuilt, so no recorded tricks are lost
    assert IrregularityKind.BID_NOT_LEGAL in {i.kind for i in hand.repairs.irregularities}


def test_bidding_round_itself_stays_strict():
    """The strictness F6 wraps is deliberately still there at the pure-rules
    layer -- other callers may want it."""
    round_ = BiddingRound(dealer=3)
    with pytest.raises(BiddingError):
        round_.record_bid(_bid(2, 32))  # seat 0 is expected


# ---------------------------------------------------------------------------
# bug-hunt fix pass: trump cue normalization (B5) and tracker divergence (B4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cue", ["the low end", "", "sixes", "  ", "trump is low"])
def test_unrecognized_trump_cue_does_not_hard_confirm_the_high_end(cue: str):
    """B5. A bare ``cue == "low"`` meant every other string -- an empty
    transcription, "the low end", "sixes" -- silently hard-confirmed the tile's
    *high* end, permanently. Speech transcription fills this field, so
    unnormalized input is the expected case, and a phrase that plainly says
    "low" must not resolve to high; it falls through to weighted tracking."""
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(5, 3), cue=cue)
    assert tracker.is_confirmed is False  # falls through to weighted tracking
    assert tracker.confirmed != 5  # the high end is not hard-confirmed by a garbled cue
    other_candidates = [n for n in range(7) if n not in (5, 3)]
    assert tracker.weights[5] == tracker.weights[3]
    assert all(tracker.weights[5] > tracker.weights[n] for n in other_candidates)


@pytest.mark.parametrize("cue,expected", [
    ("low", 3), ("high", 5), ("LOW", 3), (" High ", 5), ("Low\n", 3),
])
def test_recognized_trump_cue_still_hard_confirms_after_normalization(cue: str, expected: int):
    tracker = TrumpHypothesisTracker()
    tracker.observe_cue_on_lead(Tile.of(5, 3), cue=cue)
    assert tracker.confirmed == expected
    assert tracker.hard_confirmed is True


def test_reconcile_logs_when_the_trick_and_the_tracker_diverge():
    """B4 (reshaped). ``_reconcile_trump_for_play`` can switch the trick onto a
    candidate that is not the tracker's best guess, and the two then disagree
    with nothing logged. Only the logging half is fixed: rewarding the new
    candidate changes the exact post-reconcile weights other behaviour is pinned
    to, and activates the still-deferred ``_bias_toward`` overwrite hazard."""
    hand = HandState(dealer=3, hands={
        0: [Tile.of(6, 6), Tile.of(3, 3), Tile.of(6, 5)],
        1: [Tile.of(0, 0), Tile.of(5, 0), Tile.of(4, 3)],
        2: [Tile.of(1, 0), Tile.of(2, 2), Tile.of(1, 1)],
        3: [Tile.of(2, 1), Tile.of(5, 2), Tile.of(5, 1)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    for player, tile in [
        (0, Tile.of(6, 6)), (1, Tile.of(0, 0)), (2, Tile.of(1, 0)), (3, Tile.of(2, 1))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    # 6 leads, but 5 (the next candidate) is *not* consistent with what seat 1 is
    # about to play, so the reconcile has to reach past it to 4.
    hand.trump_tracker.weights = {n: 0.0 for n in range(7)}
    hand.trump_tracker.weights[6] = 0.4
    hand.trump_tracker.weights[5] = 0.35
    hand.trump_tracker.weights[4] = 0.25

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(3, 3)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(5, 0)))

    assert hand._current_trick.trump == 4
    assert hand.trump_tracker.best_guess == 5  # the two genuinely disagree
    assert IrregularityKind.TRUMP_HYPOTHESIS_DIVERGED in {
        i.kind for i in hand.repairs.irregularities
    }


# ---------------------------------------------------------------------------
# bug-hunt fix pass: voids under a trump confirmed after the lead (B1)
# ---------------------------------------------------------------------------

def _hand_with_trump_confirmed_one_tile_late() -> HandState:
    """Speech lagging video by a single tile: seat 0 leads, the trick's
    ``led_suit`` freezes under the *guessed* trump, and only then does the
    trump call land -- on a different number, under which the led tile counts
    as a different suit entirely."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(5, 3)))  # no cue: 5/3 both live
    assert hand._current_trick.led_suit == 3
    hand.call_trump(0, trump=6)  # ...and the call arrives one tile late
    for player, tile in [(1, Tile.of(1, 0)), (2, Tile.of(2, 0)), (3, Tile.of(4, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    return hand


def test_voids_are_skipped_for_a_trick_led_under_a_different_trump():
    """B1. Under trump 6 the led 5-3 counts as a *5*, not the 3 the trick froze.
    Every "didn't follow suit 3" judgement would be measured against the wrong
    suit and produce confidently-wrong void constraints (recorded on 66% of 400
    randomized hands where the call lands one tile late)."""
    hand = _hand_with_trump_confirmed_one_tile_late()

    assert len(hand.tricks) == 1
    assert all(not v for v in hand.voids.values())
    # Still stamped, or the next trick sees a mismatch and wipes good voids.
    assert hand._voids_trump == 6


def test_voids_are_still_recorded_for_a_trick_whose_lead_is_consistent():
    """The control for the test above: an ordinary trick, confirmed before the
    lead, must still record its voids exactly as before."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))  # trump led
    for player, tile in [(1, Tile.of(1, 0)), (2, Tile.of(2, 0)), (3, Tile.of(4, 0))]:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert hand.voids[1] == {6}
    assert hand.voids[2] == {6}
    assert hand.voids[3] == {6}
