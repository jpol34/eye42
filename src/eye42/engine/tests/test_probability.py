from __future__ import annotations

import random

from eye42.engine.events import TilePlayed
from eye42.engine.hand import HandState
from eye42.engine.probability import estimate_bid_probability, tile_hold_probability
from eye42.engine.tiles import Tile


def test_locked_set_hand_has_zero_bid_probability():
    hand = HandState(dealer=3)
    hand.bid(0, 35)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Defenders sweep the first two tricks worth 27 points (mirrors the locked-set
    # scoring test) -- 35 - 0 > 42 - 27, mathematically impossible to recover.
    trick_a = [(0, Tile.of(1, 0)), (1, Tile.of(6, 4)), (2, Tile.of(2, 0)), (3, Tile.of(3, 0))]
    trick_b = [(1, Tile.of(5, 5)), (2, Tile.of(5, 0)), (3, Tile.of(5, 1)), (0, Tile.of(5, 2))]
    for player, tile in trick_a + trick_b:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert hand.scorer.is_locked_set is True
    assert estimate_bid_probability(hand) == 0.0


def test_already_guaranteed_hand_has_full_bid_probability():
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Three trump-led tricks that sweep two 5-count tiles and two 10-count
    # tiles between them: 3 tricks (3) + 4-1(5) + 6-4(10) + 3-2(5) + 5-5(10)
    # = 3 + 30 = 33 points for seat 0's team, clearing the bid of 30 with 4
    # tricks still unplayed.
    plays = [
        (0, Tile.of(6, 6)), (1, Tile.of(4, 0)), (2, Tile.of(4, 1)), (3, Tile.of(4, 2)),
        (0, Tile.of(6, 4)), (1, Tile.of(3, 0)), (2, Tile.of(3, 1)), (3, Tile.of(0, 0)),
        (0, Tile.of(6, 5)), (1, Tile.of(1, 1)), (2, Tile.of(3, 2)), (3, Tile.of(5, 5)),
    ]
    for player, tile in plays:
        hand.play_tile(TilePlayed(player=player, tile=tile))

    assert hand.scorer.bidding_team_points == 33
    assert len(hand.tricks) == 3  # 4 tricks still to come, but already guaranteed
    assert estimate_bid_probability(hand) == 1.0


def test_fully_known_hands_gives_deterministic_probability():
    hand = HandState(dealer=3, hands={
        0: [Tile.of(6, 6), Tile.of(6, 5), Tile.of(6, 4), Tile.of(6, 3), Tile.of(6, 2), Tile.of(6, 1), Tile.of(6, 0)],
        1: [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1), Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2), Tile.of(3, 0)],
        2: [Tile.of(3, 1), Tile.of(3, 3), Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2), Tile.of(4, 3), Tile.of(4, 4)],
        3: [Tile.of(5, 1), Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3), Tile.of(5, 4), Tile.of(5, 5), Tile.of(5, 0)],
    })
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    # Seat 0 holds every trump and will sweep all 7 tricks under the estimator's
    # policy -- with fully known hands the result must be exactly 1.0, not a
    # sampled approximation.
    assert estimate_bid_probability(hand) == 1.0


def test_tile_hold_probability_with_known_hands_is_exact():
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

    assert tile_hold_probability(hand, Tile.of(6, 6), player=0) == 1.0
    assert tile_hold_probability(hand, Tile.of(6, 6), player=1) == 0.0


def test_tile_hold_probability_uses_voids_to_deduce_holder():
    """Every trick below is trump-led, so seats 1-3 all fail to follow suit 6
    from trick 1 onward and become void in trump -- the last unseen trump tile
    can then only belong to seat 0, deterministically, purely from voids."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    used = [Tile.of(6, n) for n in (6, 5, 4, 3, 2, 1)]
    fillers = [
        [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1)],
        [Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2)],
        [Tile.of(3, 0), Tile.of(3, 1), Tile.of(3, 3)],
        [Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2)],
        [Tile.of(4, 3), Tile.of(4, 4), Tile.of(5, 1)],
        [Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3)],
    ]
    for lead, (f1, f2, f3) in zip(used, fillers):
        hand.play_tile(TilePlayed(player=0, tile=lead))
        hand.play_tile(TilePlayed(player=1, tile=f1))
        hand.play_tile(TilePlayed(player=2, tile=f2))
        hand.play_tile(TilePlayed(player=3, tile=f3))

    assert hand.voids[1] == {6} and hand.voids[2] == {6} and hand.voids[3] == {6}

    remaining_trump = Tile.of(6, 0)
    rng = random.Random(42)
    assert tile_hold_probability(hand, remaining_trump, player=0, samples=200, rng=rng) == 1.0
    assert tile_hold_probability(hand, remaining_trump, player=1, samples=200, rng=rng) == 0.0
