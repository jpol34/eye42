"""The hard constraint, enforced mechanically: "the program should always
handle this in a professional way without blowing up. hands go on, stuff can
get messy, the main thing is to not just completely blow up."

This drives thousands of randomized, plausibly-messy event sequences through
HandState/GameState and asserts zero exceptions of any kind -- including
AssertionError, since a bare `assert` on a path reachable from a real event is
exactly the kind of crash this project can't afford.

Caveat, stated deliberately rather than relied on as a safety net: `python -O`
strips `assert` statements, so this test cannot prove there is no reachable
`assert` under an optimized run -- it can only prove none fire under normal
execution. That's precisely why reachable asserts were converted to explicit
checks in hand.py/game.py rather than left as the only guard.
"""

from __future__ import annotations

import random

import pytest

from eye42.engine.events import IrregularEndSignal, TilePlayed, TilesDealt
from eye42.engine.game import GameState
from eye42.engine.hand import HandError, HandState
from eye42.engine.probability import estimate_bid_probability, tile_hold_probability
from eye42.engine.tiles import full_set

SEEDS = [0, 1, 2, 3, 7, 42, 1337, 99991]
ALL_TILES = list(full_set())


def _random_event_sequence(rng: random.Random, length: int):
    """A grab-bag of plausible-and-implausible actions: legal bids/passes,
    trump calls (including from the wrong seat or before bidding resolves),
    tile plays (including repeats, out-of-turn, and tiles from outside the
    seat's real hand), and a deal event with possibly-bad counts."""
    events = []
    for _ in range(length):
        kind = rng.choice(["bid", "pass", "trump", "play", "deal"])
        player = rng.randrange(4)
        if kind == "bid":
            if rng.random() < 0.3:
                # Marks bids (splash/plunge territory). Without these the whole
                # splash/plunge upgrade path -- including the marks-upgrade crash
                # this suite exists to catch -- was invisible to the fuzzer.
                events.append(("bid", player, 0, rng.randrange(1, 5)))
            else:
                events.append(("bid", player, rng.randrange(29, 43), 0))
        elif kind == "pass":
            events.append(("pass", player))
        elif kind == "trump":
            events.append(("trump", player, rng.randrange(-1, 8)))
        elif kind == "play":
            tile = rng.choice(ALL_TILES)
            events.append(("play", player, tile))
        else:
            counts = {p: rng.randrange(0, 10) for p in range(4)}
            events.append(("deal", player, counts))
    return events


def _run_sequence(hand: HandState, events) -> None:
    for event in events:
        kind = event[0]
        if kind == "bid":
            _, player, amount, marks = event
            # No try/except any more: HandState.bid is now the non-strict wrapper
            # every other HandState event method already is -- an out-of-turn,
            # under-the-high-bid or after-close bid is logged and dropped, not
            # raised. BiddingRound itself stays strict; that contract is tested at
            # the BiddingRound level in test_state_machine.py.
            hand.bid(player, amount, marks)
        elif kind == "pass":
            _, player = event
            hand.bid_pass(player)
        elif kind == "trump":
            _, player, trump = event
            if trump not in range(7):
                with pytest.raises(HandError):
                    hand.call_trump(player, trump)
            else:
                hand.call_trump(player, trump)
        elif kind == "play":
            _, player, tile = event
            hand.play_tile(TilePlayed(player=player, tile=tile))
        else:
            _, _player, counts = event
            hand.record_deal(TilesDealt(dealer=hand.dealer, counts=counts))


def _exercise_probability(hand: HandState, rng: random.Random) -> None:
    """Drag the probability engine over whatever mess the fuzz left behind.

    Gated the same way this file already handles BiddingRound's strict contract:
    the estimators deliberately raise if asked before a contract/trump ever
    existed (API misuse, not a table event), so skip those. Everything past that
    gate must return a value or ``None`` -- never raise. Correctness is not
    asserted here; only that this crash surface stays closed.
    """
    if hand.contract is None or hand.scorer is None:
        return
    if not hand.trump_tracker.ever_confirmed:
        return
    estimate_bid_probability(hand, samples=5)
    tile_hold_probability(hand, rng.choice(ALL_TILES), player=rng.randrange(4), samples=5)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_hand_event_sequence_raises(seed: int):
    rng = random.Random(seed)
    hand = HandState(dealer=rng.randrange(4))
    events = _random_event_sequence(rng, length=60)
    _run_sequence(hand, events)  # the assertion is simply that this returns
    _exercise_probability(hand, rng)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_hand_event_sequence_raises_with_known_hands(seed: int):
    """Same fuzz, but with hands known -- this is the branch that exercises
    _reconcile_trump_for_play, revoke detection, and the hand.remove guard."""
    rng = random.Random(seed)
    shuffled = list(ALL_TILES)
    rng.shuffle(shuffled)
    hands = {p: shuffled[p * 7:(p + 1) * 7] for p in range(4)}
    hand = HandState(dealer=rng.randrange(4), hands=hands)
    events = _random_event_sequence(rng, length=60)
    _run_sequence(hand, events)
    _exercise_probability(hand, rng)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_game_sequence_raises(seed: int):
    rng = random.Random(seed)
    game = GameState(starting_dealer=rng.randrange(4))

    for _ in range(10):
        hand = game.new_hand()
        events = _random_event_sequence(rng, length=20)
        _run_sequence(hand, events)
        _exercise_probability(hand, rng)
        hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=len(hand.tricks)))
        game.record_hand(hand)
        game.observe_next_dealer(rng.randrange(4))
