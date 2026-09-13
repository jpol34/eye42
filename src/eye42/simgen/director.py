"""Scripts a full 4-player Texas 42 hand's tile-motion events (deal, plays, sweeps) for
simgen's physics+rendering pipeline, reusing the engine's own tested legality and
playout-policy code rather than reimplementing Texas 42 rules.

Bidding is deliberately out of scope: it's a purely verbal game phase with no tile
motion, so it produces nothing for a physics/rendering pipeline to depict. Trump is
picked directly from a simple heuristic instead of scripting a full BiddingRound.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from eye42.engine.probability import _choose_play, _lead_choice
# Reused: eye42's own tested (if underscore-private, viewer-heuristic) playout policy --
# see the plan's accepted-risk note. probability.py has no test contract obligating this
# private shape to stay stable; if it changes, this module's own legality assertion
# (Trick.play(..., strict=True) raising) will surface the break immediately, even though
# probability.py's own suite won't.
from eye42.engine.tiles import Tile, full_set, suits_of
from eye42.engine.trick import Trick


@dataclass(frozen=True)
class DealIntent:
    hands: Tuple[Tuple[Tile, ...], Tuple[Tile, ...], Tuple[Tile, ...], Tuple[Tile, ...]]  # by seat


@dataclass(frozen=True)
class PlayIntent:
    seat: int
    tile: Tile
    trick_index: int


@dataclass(frozen=True)
class SweepIntent:
    """A completed trick's 4 tiles being collected by its winner."""

    trick_index: int
    winner: int


Intent = Union[DealIntent, PlayIntent, SweepIntent]


def _dominant_suit(hand: Sequence[Tile]) -> int:
    """A simple trump heuristic (the pip value appearing most often across the hand's
    tiles) -- not real Texas 42 bidding strategy, just enough to pick a plausible trump
    for scripted gameplay now that bidding itself is out of scope."""
    counts: Counter[int] = Counter()
    for tile in hand:
        for suit in suits_of(tile):
            counts[suit] += 1
    return counts.most_common(1)[0][0]


def script_one_hand(rng: random.Random) -> List[Intent]:
    """Deals, then plays out, one complete 7-trick Texas 42 hand using the engine's own
    legality (``Trick.play(..., strict=True)``) and heuristic playout policy
    (``_lead_choice``/``_choose_play``) -- the same decision logic eye42 already trusts
    for live-viewer probability estimates, reused here to drive scripted gameplay
    instead of reimplementing play strategy from scratch."""
    tiles = list(full_set())
    rng.shuffle(tiles)
    hands: Dict[int, List[Tile]] = {seat: tiles[seat * 7 : (seat + 1) * 7] for seat in range(4)}

    intents: List[Intent] = [DealIntent(hands=tuple(tuple(hands[seat]) for seat in range(4)))]

    leader = 0
    trump = _dominant_suit(hands[0])
    for trick_index in range(7):
        trick = Trick(leader=leader, trump=trump)
        while not trick.is_complete:
            player = trick.next_player
            if player is None:
                break
            hand_tiles = hands[player]
            choice = (
                _lead_choice(hand_tiles, trump)
                if trick.led_suit is None
                else _choose_play(trick, player, hand_tiles, trump)
            )
            trick.play(player, choice, hand=hand_tiles, strict=True)
            hand_tiles.remove(choice)
            intents.append(PlayIntent(seat=player, tile=choice, trick_index=trick_index))

        winner = trick.winner
        assert winner is not None  # every trick above completes with exactly 4 plays
        intents.append(SweepIntent(trick_index=trick_index, winner=winner))
        leader = winner

    return intents
