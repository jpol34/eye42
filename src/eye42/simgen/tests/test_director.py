from __future__ import annotations

import random

from eye42.engine.tiles import full_set
from eye42.simgen.director import DealIntent, PlayIntent, SweepIntent, script_one_hand


def test_deal_gives_each_seat_exactly_seven_tiles():
    intents = script_one_hand(random.Random(0))

    deal = intents[0]
    assert isinstance(deal, DealIntent)
    assert all(len(hand) == 7 for hand in deal.hands)


def test_every_tile_in_the_full_set_is_played_exactly_once():
    intents = script_one_hand(random.Random(0))

    played = [i.tile for i in intents if isinstance(i, PlayIntent)]
    assert len(played) == 28
    assert set(played) == full_set()


def test_seven_tricks_of_four_plays_each_with_a_sweep_after_every_trick():
    intents = script_one_hand(random.Random(0))

    plays = [i for i in intents if isinstance(i, PlayIntent)]
    sweeps = [i for i in intents if isinstance(i, SweepIntent)]
    assert len(sweeps) == 7
    for trick_index in range(7):
        trick_plays = [p for p in plays if p.trick_index == trick_index]
        assert len(trick_plays) == 4
        assert {p.seat for p in trick_plays} == {0, 1, 2, 3}


def test_sweep_winner_leads_the_next_trick():
    intents = script_one_hand(random.Random(0))

    sweeps = {s.trick_index: s.winner for s in intents if isinstance(s, SweepIntent)}
    plays = [i for i in intents if isinstance(i, PlayIntent)]
    for trick_index in range(1, 7):
        first_play_of_trick = next(p for p in plays if p.trick_index == trick_index)
        assert first_play_of_trick.seat == sweeps[trick_index - 1]


def test_scripting_a_hand_never_raises_an_illegal_play_error():
    """script_one_hand relies on Trick.play(..., strict=True) raising IllegalPlayError
    on any violation -- so simply completing without raising, across many seeds, is
    itself the legality proof (every play the engine's own tested legality check
    accepted)."""
    for seed in range(25):
        script_one_hand(random.Random(seed))  # raises IllegalPlayError on any violation


def test_same_seed_scripts_the_same_hand():
    intents_a = script_one_hand(random.Random(7))
    intents_b = script_one_hand(random.Random(7))

    plays_a = [(i.seat, i.tile) for i in intents_a if isinstance(i, PlayIntent)]
    plays_b = [(i.seat, i.tile) for i in intents_b if isinstance(i, PlayIntent)]
    assert plays_a == plays_b
