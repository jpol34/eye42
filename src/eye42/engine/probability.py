"""Live probability estimation for the viewer-facing display (the "poker-broadcast
equity bar" feature).

Once trump is confirmed, played tiles are known and voids are tracked (see
HandState._record_voids), but the ~21 tiles not yet played are hidden among the
other players' hands. This module estimates:

- ``estimate_bid_probability``: P(the bidding team still makes their contract),
  via Monte Carlo — sample many tile deals consistent with known voids and hand
  sizes, play each out with a simple heuristic policy, and take the fraction
  that succeed. The deterministic 0%/100% edge cases (locked-set / already
  mathematically guaranteed / hand fully played out) are short-circuited
  without sampling. It returns a ``BidOutcomeEstimate`` (probability + Wilson
  95% interval + samples actually used + expected final defending-team
  points), not a bare float: the defense's expected points fall out of the
  *same* sampling loop for free, and a live display wants the interval to know
  how much to trust the bar.
- ``tile_hold_probability``: P(a specific unseen tile is held by a specific
  player). Two degenerate cases are decided analytically from voids alone (the
  player can't hold the tile → 0.0; exactly one player still can → 1.0);
  anything genuinely ambiguous falls back to the *same* sampler as above
  (tallying how often the deal assigns it to that player) rather than a separate
  closed-form calculation — per-player void constraints make this not a simple
  hypergeometric ratio, so one well-tested mechanism beats two parallel ones.

This is a heuristic estimator, not a solver: the playout policy (win the trick
only if you can actually beat the current best play; shed your lowest
count-value tile otherwise, or your highest onto a trick your partner has
already locked up as the last player to act) is a reasonable simple stand-in for
real play, not optimal double-dummy analysis. Good enough for a live "which way
is this trending" viewer display; not a claim of game-theoretic accuracy.

Hard project constraint: nothing in here may ever raise on a real (even messy)
hand. Every unsatisfiable/inconsistent state returns ``None`` instead.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bidding import partner_of, team_of
from .hand import HandState
from .tiles import Tile, effective_suit, full_set, is_trump, trick_rank
from .trick import Trick

MAX_DEAL_ATTEMPTS = 200
TRICKS_PER_HAND = 7

# How many *consecutive* failed deal-sampling attempts abort a whole estimation
# call. Each individual attempt already costs up to ``MAX_DEAL_ATTEMPTS``
# shuffles, so without this an unsatisfiable void configuration burns
# ``samples * MAX_DEAL_ATTEMPTS`` shuffles (~14s at the default 1000 samples)
# before returning None -- a live-display hang, which this project forbids.
# Well below any realistic ``samples`` value, and far beyond what legitimate
# variance produces on a satisfiable state: the most-constrained-first ordering
# makes a satisfiable deal essentially deterministic per attempt, so 20 failures
# in a row means unsatisfiable, not unlucky.
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20

# The consecutive-failure guard above only fires on an unbroken run of failures,
# so a void shape that fails most attempts but succeeds occasionally resets it
# forever and never trips it -- measured at 9.7s for one default-``samples``
# call, worst case ~16s, which is exactly the live-display hang the constant
# above exists to prevent. Bound the *total* failures too, as a fraction of the
# requested sample count so it scales with the caller's budget.
MAX_FAILURE_FRACTION = 4  # bail once failures exceed samples // this


@dataclass(frozen=True)
class BidOutcomeEstimate:
    """The result of one estimation pass.

    ``confidence_interval`` is a Wilson score 95% interval around
    ``probability``. Deterministic short-circuits (locked set, already
    guaranteed, a completed hand) return a degenerate result: ``samples == 0``
    and the interval collapsed onto the point value.

    ``expected_defense_points`` is the defending team's *projected final*
    points for the completed hand -- an average over the sampled playouts. It
    is ``None`` when that value would require a simulation the code path did not
    perform: the ``is_locked_set`` and already-guaranteed short-circuits decide
    the *win/loss* question by pure arithmetic on the running score and never
    play out the remaining tricks, so no projected final point split exists.
    Reporting the running tally there would silently mean something different
    from every other path. (A hand that has actually finished all seven tricks
    is not such a case: its running tally *is* the final figure.)
    """

    probability: float
    confidence_interval: Tuple[float, float]
    samples: int
    expected_defense_points: Optional[float]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion — closed-form arithmetic,
    ``math`` only (no scipy). Unlike the normal approximation it stays inside
    [0, 1] and stays sane at 0 or 100% success, which is exactly where a
    probability bar spends a lot of its time.
    """
    if n <= 0:
        return (0.0, 1.0)
    # Clamped rather than trusted: a caller passing successes outside [0, n]
    # otherwise reaches math.sqrt of a negative and raises a ValueError out of a
    # public, directly-tested helper.
    successes = min(max(successes, 0), n)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _degenerate(probability: float, defense_points: Optional[float]) -> BidOutcomeEstimate:
    return BidOutcomeEstimate(
        probability=probability,
        confidence_interval=(probability, probability),
        samples=0,
        expected_defense_points=defense_points,
    )


# ---- deal sampling ---------------------------------------------------------


def _eligible(tile: Tile, trump: int, voided_suits: set) -> bool:
    """A player void in suit S cannot hold any tile that would follow suit S."""
    return not any(effective_suit(tile, trump, s) == s for s in voided_suits)


def _voids_for(hand: HandState, trump: int) -> Dict[int, set]:
    """``hand.voids``, but only when they were actually recorded under ``trump``.

    "Seat 2 didn't follow the led suit" is only a void *relative to a trump
    hypothesis* -- which end of a tile counts as the led suit depends on trump.
    ``HandState`` stamps the hypothesis the voids were recorded under onto
    ``_voids_trump`` and clears them when it changes, but that only happens at
    the *next* trick close. In the window between a confirmed trump changing
    (a soft confirmation reopened and re-confirmed elsewhere, mid-trick) and
    that close, ``hand.voids`` still holds constraints derived from a
    now-rejected hypothesis.

    Those are not merely stale, they are *false*: fed to the sampler they make
    real deals unsatisfiable, and fed to ``tile_hold_probability``'s analytic
    short-circuits they produce confident 1.0/0.0 answers that are simply wrong.
    An unknown void is the safe default, so treat them as empty -- every reader
    of ``hand.voids`` in this module must go through here.
    """
    if hand._voids_trump != trump:
        return {p: set() for p in range(4)}
    return hand.voids


def sample_consistent_deal(
    unseen: List[Tile],
    hand_sizes: Dict[int, int],
    voids: Dict[int, set],
    trump: int,
    rng: random.Random,
) -> Dict[int, List[Tile]]:
    """Randomly assign ``unseen`` tiles to players' remaining hand slots,
    respecting void constraints and hand-size constraints.

    Players are filled **most-constrained-first, measured by how many of the
    remaining tiles they are actually eligible to hold** — not by hand size.
    Hand size is nearly useless as a constraint ordering here (mid-hand every
    seat usually has the same number of tiles left) and gets it actively wrong
    in a completely ordinary shape: trick 1 led in trump with the other three
    seats all showing off. There the unconstrained seat sorts first, hoovers up
    tiles only it is eligible for, and leaves the void seats unsatisfiable —
    burning all the attempts and raising on a perfectly legal hand. Ordering by
    eligible-tile count assigns the void seats first and makes that case
    deterministic rather than a 1-in-134,596 lottery per attempt.

    Uses randomized retry rather than a formal backtracking solver — the
    hidden-tile space in 42 (at most 21 tiles across 3 hands) is small enough
    that this converges quickly in practice; it is not guaranteed to sample
    uniformly over all valid deals in pathological cases, which is an
    acceptable tradeoff for a live estimate over a real game-theoretic solver.
    """
    players = sorted(
        hand_sizes,
        key=lambda p: len([t for t in unseen if _eligible(t, trump, voids.get(p, set()))]),
    )
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


def _try_sample(
    unseen: List[Tile],
    hand_sizes: Dict[int, int],
    voids: Dict[int, set],
    trump: int,
    rng: random.Random,
) -> Optional[Dict[int, List[Tile]]]:
    """``sample_consistent_deal`` with the exhaustion case turned into ``None``.
    After the ordering fix above this should be rare, but a genuinely
    inconsistent state (a misdeal, an over-played seat) can still be
    unsatisfiable and must never propagate out of a viewer-only estimate."""
    try:
        return sample_consistent_deal(unseen, hand_sizes, voids, trump, rng)
    except RuntimeError:
        return None


# ---- playout policy --------------------------------------------------------


def _lead_choice(hand_tiles: List[Tile], trump: int) -> Tile:
    return max(hand_tiles, key=lambda t: (is_trump(t, trump), t.is_double, t.high))


def _choose_play(trick: Trick, player: int, hand_tiles: List[Tile], trump: int) -> Tile:
    """The heuristic policy for one seat's play.

    1. Leading: play your strongest tile.
    2. Following: play to win only if the candidate actually *beats* the current
       best play — "I can follow/trump" is not the same as "this wins," and
       burning a trump onto a trick you're already losing (or already winning)
       is a real strategic error, not just an approximation.
    3. Can't win, partner already winning, and you are the **last** seat to act
       (3 plays already on the table): dump your highest count-value tile onto
       partner's trick. The last-to-act condition matters — with an opponent
       still to play, partner "winning" is provisional and dumping a 10-count is
       actively bad.
    4. Otherwise: shed your lowest count-value tile.
    """
    if trick.led_suit is None:
        return _lead_choice(hand_tiles, trump)

    legal = trick.legal_plays(hand_tiles)
    led = trick.led_suit
    best_on_table = max(trick_rank(t, trump, led) for _, t in trick.plays)
    candidate = max(legal, key=lambda t: trick_rank(t, trump, led))
    if trick_rank(candidate, trump, led) > best_on_table:
        return candidate

    if trick.winner == partner_of(player) and len(trick.plays) == 3:
        return max(legal, key=lambda t: t.count_value)
    return min(legal, key=lambda t: t.count_value)


def _finish_trick(trick: Trick, trump: int, hands: Dict[int, List[Tile]]) -> bool:
    """Drive ``trick`` to completion from whatever state it's already in, using
    ``trick.next_player`` for turn order (which is what lets a trick that
    already had an out-of-turn/skipped seat recover correctly, instead of us
    reconstructing the rotation by hand). Returns False if a seat we're asked to
    play for has no tiles left — an inconsistent state we bail out of rather
    than crash on."""
    while not trick.is_complete:
        player = trick.next_player
        if player is None:
            break
        hand_tiles = hands.get(player) or []
        if not hand_tiles:
            return False
        choice = _choose_play(trick, player, hand_tiles, trump)
        trick.play(player, choice, hand=hand_tiles, strict=False)
        hand_tiles.remove(choice)
    return True


def _play_out_trick(leader: int, trump: int, hands: Dict[int, List[Tile]]) -> Optional[Trick]:
    trick = Trick(leader=leader, trump=trump)
    if not _finish_trick(trick, trump, hands):
        return None
    return trick


def _rebased_copy(current: Trick, trump: int) -> Optional[Trick]:
    """A deep copy of the in-progress trick, re-synced onto ``trump``.

    The copy's own ``.trump`` field is what ``Trick.legal_plays`` and
    ``Trick.winner`` use internally, and it is frozen at whatever the best guess
    was when the trick *started*. The playout policy is handed the trump the
    estimate is actually being computed under, so without this re-sync the same
    simulation could rank plays under one trump and decide legality and the
    trick winner under another -- the same reopen-mid-trick window that makes
    stale voids dangerous (see ``_voids_for``).

    The led suit is the part that cannot always be reconciled: it was derived
    from the physically-led tile under the *old* trump, and under the new one
    that tile may count as a different suit entirely. Rewriting it would
    retroactively change which of the already-played tiles followed suit, so
    that case bails to ``None`` (an unavailable estimate) rather than guessing.
    """
    trick = copy.deepcopy(current)
    if trick.trump == trump:
        return trick
    if trick.plays:
        led_tile = trick.plays[0][1]
        if effective_suit(led_tile, trump, None) != trick.led_suit:
            return None
    trick.trump = trump
    return trick


def _simulate_from_deal(
    hand: HandState, deal: Dict[int, List[Tile]]
) -> Optional[Tuple[bool, int]]:
    """Play out the remainder of the hand from a sampled deal.

    Returns ``(contract_made, defending_team_final_points)``, or ``None`` if the
    state turned out to be inconsistent enough that the simulation can't finish
    (a seat out of tiles, a trick with no determinable winner).

    Handles being called **mid-trick**: the in-progress trick is deep-copied and
    finished in place rather than reconstructed from its (seat, tile) pairs.
    Replaying those pairs into a fresh Trick would lose ``leader``/
    ``entitled_leader`` as mutated by an out-of-turn lead and could re-derive
    ``_led_suit`` differently; continuing the real object avoids all of that.
    """
    if hand.contract is None or hand.scorer is None:
        return None
    trump = hand.trump_tracker.confirmed
    if trump is None:
        return None

    bidding_team = team_of(hand.contract.bidder)
    bidding_points = hand.scorer.bidding_team_points
    defending_points = hand.scorer.defending_team_points
    swept = True  # only relevant for requires_sweep contracts

    hands = {p: list(deal.get(p, [])) for p in range(4)}

    def score(trick: Trick) -> bool:
        nonlocal bidding_points, defending_points, swept
        winner = trick.winner
        if winner is None:
            # Defensive only: a force-closed trick always has at least one real
            # play (see Trick.play's force-close threshold), so HandState never
            # hands us an empty completed trick. Guarded anyway rather than
            # letting a None reach the arithmetic.
            return False
        if team_of(winner) == bidding_team:
            bidding_points += 1 + trick.count_value
        else:
            defending_points += 1 + trick.count_value
            swept = False
        return True

    tricks_remaining = TRICKS_PER_HAND - len(hand.tricks)
    leader: Optional[int] = None

    current = hand._current_trick
    if tricks_remaining > 0 and current is not None and not current.is_complete:
        trick = _rebased_copy(current, trump)
        if trick is None:
            return None
        if not _finish_trick(trick, trump, hands):
            return None
        if not score(trick):
            return None
        tricks_remaining -= 1
        leader = trick.winner

    if leader is None:
        leader = _next_leader(hand)

    for _ in range(max(0, tricks_remaining)):
        trick = _play_out_trick(leader, trump, hands)
        if trick is None or not score(trick):
            return None
        leader = trick.winner  # type: ignore[assignment]

    if hand.contract.requires_sweep:
        made = swept
    else:
        made = bidding_points >= hand.contract.points_needed
    return made, defending_points


def _next_leader(hand: HandState) -> int:
    """Who leads the next fresh trick. Only consulted when no in-progress trick
    was continued, in which case the last completed trick's winner (or the
    trump caller, at the top of the hand) leads."""
    if hand.tricks:
        winner = hand.tricks[-1].winner
        if winner is not None:
            return winner
    return hand.contract.trump_caller if hand.contract is not None else 0


def _has_occluded_trick(hand: HandState) -> bool:
    """A trick that force-closed *short* -- fewer than 4 real plays -- means
    tiles physically went unread. The estimator then strands the missed seats'
    surplus tiles unplayed and under-totals the hand's 42-point budget (56% of
    sampled playouts in the repro, by up to 15 points), with nothing surfacing
    it. Treated like ``hand.disputed``: suppress the viewer-only estimate.

    Deliberately narrow. ``Trick.closed_early_for_new_trick`` is NOT included:
    that marks a trick HandState closed because it saw the next trick's lead, a
    fully-resolved state with no missing tiles, and suppressing on it would blank
    the equity bar on an ordinary next-trick lead.
    """
    return any(t.force_closed and len(t.plays) < 4 for t in hand.tricks)


def _bail_budget(samples: int) -> int:
    return max(1, samples // MAX_FAILURE_FRACTION)


# ---- determinism -----------------------------------------------------------


def _state_seed(hand: HandState) -> int:
    """A stable seed derived from the hand's observable state, so two calls on
    an unchanged hand give byte-identical answers.

    Deliberately *not* Python's ``hash()``: PYTHONHASHSEED randomization makes
    that unstable across processes, which would make the displayed probability
    jitter between runs for no reason. blake2b over an explicit integer encoding
    (not ``repr()`` strings, which are ambiguous about field boundaries) of the
    contract, the confirmed trump, and the full ordered play history.
    """
    digest = hashlib.blake2b(digest_size=8)

    def put(*values: int) -> None:
        for value in values:
            digest.update(int(value).to_bytes(4, "big", signed=True))

    contract = hand.contract
    if contract is not None:
        put(contract.bidder, contract.kind.value, contract.amount)
    trump = hand.trump_tracker.confirmed
    put(-1 if trump is None else trump)

    for trick in hand.tricks:
        put(-2, trick.leader)
        for player, tile in trick.plays:
            put(player, tile.high, tile.low)
    current = hand._current_trick
    if current is not None:
        put(-3, current.leader)
        for player, tile in current.plays:
            put(player, tile.high, tile.low)

    return int.from_bytes(digest.digest(), "big")


def _sampling_inputs(hand: HandState) -> Optional[tuple]:
    """``(unseen, hand_sizes)`` for the sampler, or ``None`` if the two can't
    possibly be reconciled (cheap pre-flight insurance against an over-played
    seat / misdeal; it does *not* catch void-driven unsatisfiability, which is
    handled by the sampler's ordering plus ``_try_sample``)."""
    unseen = sorted(full_set() - hand.played_tiles, key=lambda t: (t.high, t.low))
    hand_sizes = {p: hand.remaining_hand_size(p) for p in range(4)}
    if sum(hand_sizes.values()) != len(unseen):
        return None
    return unseen, hand_sizes


def _trump_or_none(hand: HandState) -> Optional[int]:
    """The confirmed trump.

    Raises only for genuine API misuse — asking for an estimate on a hand whose
    trump was *never* confirmed (i.e. still in bidding). A hand that had a
    confirmed trump and then had a soft/inferred confirmation legitimately
    reopened (see TrumpHypothesisTracker) is a real, valid, in-progress state
    and must come back as ``None``/unavailable, never as an exception.
    """
    trump = hand.trump_tracker.confirmed
    if trump is None and not hand.trump_tracker.ever_confirmed:
        raise ValueError("trump must be confirmed before estimating probabilities")
    return trump


# ---- public API ------------------------------------------------------------


def estimate_bid_probability(
    hand: HandState, samples: int = 1000, rng: Optional[random.Random] = None
) -> Optional[BidOutcomeEstimate]:
    """P(the bidding team still makes their contract) plus a 95% interval, the
    sample count actually used, and the defending team's expected final points
    (``None`` on the locked-set / already-guaranteed short-circuits, which
    answer the win question analytically and simulate nothing -- see
    ``BidOutcomeEstimate``).

    Returns ``None`` when the estimate is unavailable: an unresolved
    irregularity (a revoke, a tile that didn't match a known hand), a suspected
    misdeal, a trump confirmation that has been reopened, or a hand-size/tile
    bookkeeping state the deal sampler can't satisfy. A disputed or misdealt
    hand's voids/hand-size bookkeeping may no longer be internally consistent,
    which could otherwise make the sampler unsatisfiable — suppress the
    (viewer-only) estimate rather than risk crashing on it.
    """
    if hand.contract is None or hand.scorer is None:
        raise ValueError("bidding must be resolved and scoring started first")

    # A force-closed-short trick corrupts is_locked_set/bidding_team_points
    # themselves (see HandState.scoring_disputed), so unlike the general
    # disputed/occlusion guard further down, this must run before the exact
    # short-circuits below rather than after them -- those short-circuits
    # would otherwise answer confidently from the same broken arithmetic.
    if hand.scoring_disputed:
        return None

    # Exact answers first: these are pure arithmetic on the recorded score and
    # must never be suppressed by the sampling-safety guards below.
    # These two decide the win/loss question outright but play out nothing, so
    # there is no projected *final* defending-team total to report -- see
    # BidOutcomeEstimate. The running tally is deliberately not substituted.
    if hand.scorer.is_locked_set:
        return _degenerate(0.0, None)
    if (
        not hand.contract.requires_sweep
        and hand.scorer.bidding_team_points >= hand.contract.points_needed
    ):
        return _degenerate(1.0, None)
    # A hand that has played all seven tricks has nothing left to project: the
    # running tally is the final figure.
    if len(hand.tricks) == TRICKS_PER_HAND:
        made = hand.scorer.bidding_team_points >= hand.contract.points_needed
        return _degenerate(1.0 if made else 0.0, float(hand.scorer.defending_team_points))

    trump = _trump_or_none(hand)
    if trump is None:
        return None  # confirmation was reopened; nothing to sample against
    if hand.disputed or _has_occluded_trick(hand):
        return None

    if hand.misdeal_suspected:
        return None

    prepared = _sampling_inputs(hand)
    if prepared is None:
        return None
    unseen, hand_sizes = prepared

    rng = rng or random.Random(_state_seed(hand))

    voids = _voids_for(hand, trump)

    made_count = 0
    used = 0
    defense_total = 0
    consecutive_failures = 0
    failures = 0
    failure_budget = _bail_budget(samples)
    for _ in range(samples):
        deal = _try_sample(unseen, hand_sizes, voids, trump, rng)
        if deal is None:
            consecutive_failures += 1
            failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_SAMPLE_FAILURES or failures > failure_budget:
                # Unsatisfiable or pathologically slow, not unlucky. Break rather
                # than `return None`: samples already collected are perfectly good
                # and a partial estimate beats turning them into "unavailable".
                break
            continue
        consecutive_failures = 0
        result = _simulate_from_deal(hand, deal)
        if result is None:
            continue
        made, defense_points = result
        used += 1
        defense_total += defense_points
        if made:
            made_count += 1

    if used == 0:
        return None
    return BidOutcomeEstimate(
        probability=made_count / used,
        confidence_interval=wilson_interval(made_count, used),
        samples=used,
        expected_defense_points=defense_total / used,
    )


def tile_hold_probability(
    hand: HandState, tile: Tile, player: int, samples: int = 1000, rng: Optional[random.Random] = None
) -> Optional[float]:
    """P(``player`` holds ``tile``), 0.0-1.0, among tiles not yet played, or
    ``None`` if the estimate is unavailable (see ``estimate_bid_probability``).
    A tile already observed played is answered directly regardless -- that's an
    observed fact, not something the sampler needs to be consistent for.

    Keeps the simple ``Optional[float]`` contract: this is a narrower question
    than "does the bid make it" and has no use for an interval or a point total.
    """
    if player not in range(4):
        # API misuse rather than a table event, so None/unavailable -- this
        # function's contract for "can't answer that" -- rather than raising.
        return None

    if tile in hand.played_tiles:
        return 1.0 if hand._seen_tiles.get(tile) == player else 0.0

    trump = _trump_or_none(hand)
    if hand.disputed:
        return None

    # Both remaining guards sit *after* the exact-answer branches above: an
    # answer computable without sampling is never suppressed by a
    # sampling-safety check. (A short constructed test deal trips
    # misdeal_suspected by design, and still has an exact answer.)
    if trump is None:
        return None
    if hand.misdeal_suspected:
        return None

    # Analytic short-circuits: the two cases voids alone actually decide.
    # Anything beyond these is a per-player-constrained assignment problem, not
    # a hypergeometric ratio, so it goes to the sampler rather than a formula.
    voids = _voids_for(hand, trump)
    candidates = [
        p
        for p in range(4)
        if hand.remaining_hand_size(p) > 0 and _eligible(tile, trump, voids.get(p, set()))
    ]
    if player not in candidates:
        return 0.0
    if len(candidates) == 1:
        return 1.0

    prepared = _sampling_inputs(hand)
    if prepared is None:
        return None
    unseen, hand_sizes = prepared

    rng = rng or random.Random(_state_seed(hand))

    hits = 0
    used = 0
    consecutive_failures = 0
    failures = 0
    failure_budget = _bail_budget(samples)
    for _ in range(samples):
        deal = _try_sample(unseen, hand_sizes, voids, trump, rng)
        if deal is None:
            consecutive_failures += 1
            failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_SAMPLE_FAILURES or failures > failure_budget:
                break  # same total-failure bound as estimate_bid_probability
            continue
        consecutive_failures = 0
        used += 1
        if tile in deal.get(player, []):
            hits += 1
    if used == 0:
        return None
    return hits / used
