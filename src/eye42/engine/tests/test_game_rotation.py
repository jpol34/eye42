from __future__ import annotations

from eye42.engine.events import IrregularEndSignal, TilePlayed
from eye42.engine.game import GameState
from eye42.engine.hand import HandOutcomeKind
from eye42.engine.repair import IrregularityKind
from eye42.engine.tiles import Tile


def test_dealer_did_not_rotate_reclassifies_concession_as_redeal():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.call_trump(2, trump=6)

    hand.play_tile(TilePlayed(player=2, tile=Tile.of(6, 6)))
    hand.play_tile(TilePlayed(player=3, tile=Tile.of(1, 0)))
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(2, 0)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(3, 0)))

    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=1))
    game.record_hand(hand)
    assert game.dealer == 2  # advanced, provisionally classified CONCESSION
    assert game.marks == [0, 1]  # defenders awarded a mark for the (only 1-trick) hand

    # The next hand is observed to be dealt by seat 1 again -- the dealer never
    # actually rotated, so this was really a redeal.
    result = game.observe_next_dealer(1)

    assert result == "reclassified_redeal"
    assert hand.outcome_kind == HandOutcomeKind.REDEAL
    assert game.dealer == 1
    assert game.marks == [0, 0]  # rolled back exactly
    assert IrregularityKind.DEALER_ROTATION_MISMATCH in {i.kind for i in game.repairs.irregularities}


def test_dealer_rotated_reclassifies_redeal_as_concession():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=0))
    game.record_hand(hand)
    assert game.dealer == 1  # unchanged, provisionally classified REDEAL

    # The next hand is observed to be dealt by seat 2 -- the dealer DID rotate,
    # so this was really a completed/conceded hand, not a redeal.
    result = game.observe_next_dealer(2)

    assert result == "reclassified_concession"
    assert hand.outcome_kind == HandOutcomeKind.CONCESSION
    assert game.dealer == 2


def test_matching_dealer_confirms_a_provisional_classification():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=0))
    game.record_hand(hand)

    result = game.observe_next_dealer(1)  # matches expectation for a REDEAL
    assert result is None
    assert game.dealer == 1


def test_unexpected_dealer_adopts_observed_value_and_flags():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=0))
    game.record_hand(hand)

    result = game.observe_next_dealer(3)  # matches neither expectation
    assert result == "adopted_unexpected_dealer"
    assert game.dealer == 3
    assert IrregularityKind.DEALER_ROTATION_MISMATCH in {i.kind for i in game.repairs.irregularities}


def test_observe_next_dealer_on_empty_ledger_is_a_noop():
    game = GameState(starting_dealer=0)
    assert game.observe_next_dealer(2) is None


def test_record_hand_survives_a_hand_with_no_contract():
    game = GameState(starting_dealer=0)
    hand = game.new_hand()  # no bids ever recorded -- scorer is None

    game.record_hand(hand)  # must not raise

    assert game.dealer == 1  # still rotates
    assert game.marks == [0, 0]


def test_reclassifying_redeal_to_concession_survives_a_hand_with_no_contract():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()  # bidding never even starts -- scorer is None
    hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=0))
    game.record_hand(hand)
    assert game.dealer == 1  # unchanged (REDEAL)

    result = game.observe_next_dealer(2)  # dealer rotated -- should have been a concession

    assert result == "reclassified_concession"  # must not raise HandError internally
    assert hand.outcome_kind == HandOutcomeKind.CONCESSION
    assert game.dealer == 2
    assert game.marks == [0, 0]


def test_open_questions_aggregates_hand_and_game_irregularities():
    game = GameState(starting_dealer=1)
    hand = game.new_hand()
    hand.bid(2, 30)
    hand.bid_pass(3)
    hand.bid_pass(0)
    hand.bid_pass(1)
    hand.call_trump(0, trump=6)  # wrong seat -- logs an open question on the hand
    game.record_hand(hand)
    game.observe_next_dealer(3)  # logs an open question on the game

    questions = game.open_questions()
    assert any(q.kind == IrregularityKind.TRUMP_CALLED_BY_WRONG_SEAT for q in questions)
    assert any(q.kind == IrregularityKind.DEALER_ROTATION_MISMATCH for q in questions)
