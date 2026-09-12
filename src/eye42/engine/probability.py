"""Live probability estimation for the viewer-facing display (the "poker-broadcast
equity bar" feature).

Once trump is confirmed, played tiles are known and voids are tracked (see
HandState._record_voids), but the ~21 tiles not yet played are hidden among the
other players' hands. This module estimates:

- ``estimate_bid_probability``: P(the bidding team still makes their contract),
  via Monte Carlo — sample many tile deals consistent with known voids and hand
  sizes, play each out with a simple heuristic policy, and take the fraction
  that succeed. The deterministic 0%/100% edge cases (locked-set / already
  mathematically guaranteed) are short-circuited without sampling.
- ``tile_hold_probability``: P(a specific unseen tile is held by a specific
  player), via the *same* sampler (tallying how often the deal assigns it to
  that player) rather than a separate closed-form calculation — 42's hidden
  space is small enough (a few thousand samples) that one simple, well-tested
  mechanism beats two parallel ones for a hobby project.

This is a heuristic estimator, not a solver: the playout policy (play to win if
you can, else shed your lowest count-value tile) is a reasonable simple stand-in
for real play, not optimal double-dummy analysis. Good enough for a live "which
way is this trending" viewer display; not a claim of game-theoretic accuracy.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .hand import HandState
from .tiles import Tile, effective_suit, full_set, is_trump, trick_rank
from .trick import Trick

MAX_DEAL_ATTEMPTS = 200


def _eligible(tile: Tile, trump: int, voided_suits: set) -> bool:
    """A player void in suit S cannot hold any tile that would follow suit S."""
    return not any(effective_suit(tile, trump, s) == s for s in voided_suits)


def sample_consistent_deal(
    unseen: List[Tile],
    hand_sizes: Dict[int, int],
    voids: Dict[int, set],
    trump: int,
    rng: random.Random,
) -> Dict[int, List[Tile]]:
    """Randomly assign ``unseen`` tiles to players' remaining hand slots,
    respecting void constraints and hand-size constraints.

    Uses randomized retry rather than a formal backtracking solver — the
    hidden-tile space in 42 (at most 21 tiles across 3 hands) is small enough
    that this converges quickly in practice; it is not guaranteed to sample
    uniformly over all valid deals in pathological cases, which is an
    acceptable tradeoff for a live estimate over a real game-theoretic solver.
    """
    players = sorted(hand_sizes, key=lambda p: hand_sizes[p])  # most-constrained slot-count first
    for _ in range(MAX_DEAL_ATTEMPTS):
        pool = list(unseen)
        rng.shuffle(pool)
        deal: Dict[int, List[Tile]] = {}
        ok = True
        for player in players:
            eligible = [t for t in pool if _eligible(t, trump, voids.get(player, set()))]
            need = hand_sizes[player]
            if len(eligible) < need:
                ok = False
                break
            chosen = rng.sample(eligible, need)
            deal[player] = chosen
            for t in chosen:
                pool.remove(t)
        if ok and not pool:
            return deal
    raise RuntimeError("could not find a deal consistent with known voids after max attempts")


def _lead_choice(hand_tiles: List[Tile], trump: int) -> Tile:
    return max(hand_tiles, key=lambda t: (is_trump(t, trump), t.is_double, t.high))


def _follow_choice(legal: List[Tile], trump: int, led_suit: int) -> Tile:
    best = max(legal, key=lambda t: trick_rank(t, trump, led_suit))
    if trick_rank(best, trump, led_suit)[0] > 0:
        return best  # can trump or follow-and-possibly-win
    return min(legal, key=lambda t: t.count_value)  # can't win: shed lowest count-value tile


def _play_out_trick(leader: int, trump: int, hands: Dict[int, List[Tile]]) -> Trick:
    trick = Trick(leader=leader, trump=trump)
    player = leader
    for _ in range(4):
        hand_tiles = hands[player]
        if trick.led_suit is None:
            choice = _lead_choice(hand_tiles, trump)
        else:
            choice = _follow_choice(trick.legal_plays(hand_tiles), trump, trick.led_suit)
        trick.play(player, choice, hand=hand_tiles)
        hand_tiles.remove(choice)
        player = (player + 1) % 4
    return trick


def _simulate_from_deal(hand: HandState, deal: Dict[int, List[Tile]]) -> bool:
    """Play out the remainder of the hand from a sampled deal and return
    whether the bidding team's contract is made."""
    assert hand.contract is not None and hand.scorer is not None
    trump = hand.trump_tracker.confirmed
    assert trump is not None

    bidding_points = hand.scorer.bidding_team_points
    tricks_done = len(hand.tricks)
    leader = hand.tricks[-1].winner if hand.tricks else hand.contract.trump_caller

    hands = {p: list(deal.get(p, [])) for p in range(4)}
    swept = True  # only relevant for requires_sweep contracts
    for p, tiles in hand.hands.items() if hand.hands else []:
        hands[p] = list(tiles)  # prefer real known-hand data over sampled guesses

    for _ in range(7 - tricks_done):
        trick = _play_out_trick(leader, trump, hands)
        won_by_bidding_team = (trick.winner % 2) == (hand.contract.bidder % 2)  # type: ignore[operator]
        if not won_by_bidding_team:
            swept = False
        if won_by_bidding_team:
            bidding_points += 1 + trick.count_value
        leader = trick.winner  # type: ignore[assignment]

    if hand.contract.requires_sweep:
        return swept
    return bidding_points >= hand.contract.points_needed


def _hands_fully_known(hand: HandState) -> bool:
    return hand.hands is not None and all(p in hand.hands for p in range(4))


def estimate_bid_probability(
    hand: HandState, samples: int = 1000, rng: Optional[random.Random] = None
) -> Optional[float]:
    """P(the bidding team still makes their contract), 0.0-1.0, or ``None`` if
    the hand has an unresolved irregularity (a revoke, a tile that didn't match
    a known hand, ...). A disputed hand's voids/hand-size bookkeeping may no
    longer be internally consistent, which could otherwise make the deal
    sampler unsatisfiable -- suppress the (viewer-only) estimate rather than
    risk crashing on it."""
    if hand.contract is None or hand.scorer is None:
        raise ValueError("bidding must be resolved and scoring started first")
    if hand.trump_tracker.confirmed is None:
        raise ValueError("trump must be confirmed before estimating bid probability")
    if hand.disputed:
        return None

    if hand.scorer.is_locked_set:
        return 0.0
    if not hand.contract.requires_sweep and hand.scorer.bidding_team_points >= hand.contract.points_needed:
        return 1.0  # already mathematically guaranteed; more points can't unmake it
    if len(hand.tricks) == 7:
        return 1.0 if hand.scorer.bidding_team_points >= hand.contract.points_needed else 0.0
    if _hands_fully_known(hand):
        # No hidden information left to sample -- the outcome is deterministic
        # under the playout policy, so a single simulation is the exact answer
        # (not a probability under uncertainty, just this policy's result).
        return 1.0 if _simulate_from_deal(hand, {}) else 0.0

    rng = rng or random.Random()
    trump = hand.trump_tracker.confirmed
    unseen = list(full_set() - hand.played_tiles)
    hand_sizes = {p: hand.remaining_hand_size(p) for p in range(4)}

    made_count = 0
    for _ in range(samples):
        deal = sample_consistent_deal(unseen, hand_sizes, hand.voids, trump, rng)
        if _simulate_from_deal(hand, deal):
            made_count += 1
    return made_count / samples


def tile_hold_probability(
    hand: HandState, tile: Tile, player: int, samples: int = 1000, rng: Optional[random.Random] = None
) -> Optional[float]:
    """P(``player`` holds ``tile``), 0.0-1.0, among tiles not yet played, or
    ``None`` if the hand is disputed (see ``estimate_bid_probability``). A tile
    already observed played is answered directly regardless -- that's an
    observed fact, not something the sampler needs to be consistent for."""
    if tile in hand.played_tiles:
        return 1.0 if hand._seen_tiles.get(tile) == player else 0.0
    if hand.trump_tracker.confirmed is None:
        raise ValueError("trump must be confirmed before estimating hold probability")
    if hand.disputed:
        return None
    if _hands_fully_known(hand):
        return 1.0 if tile in hand.hands[player] else 0.0  # type: ignore[index]

    rng = rng or random.Random()
    trump = hand.trump_tracker.confirmed
    unseen = list(full_set() - hand.played_tiles)
    hand_sizes = {p: hand.remaining_hand_size(p) for p in range(4)}

    hits = 0
    for _ in range(samples):
        deal = sample_consistent_deal(unseen, hand_sizes, hand.voids, trump, rng)
        if tile in deal.get(player, []):
            hits += 1
    return hits / samples
