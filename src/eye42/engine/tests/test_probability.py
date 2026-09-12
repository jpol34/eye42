from __future__ import annotations

import random
import time

from eye42.engine.events import TilePlayed, TilesDealt
from eye42.engine.hand import HandState
from eye42.engine import probability
from eye42.engine.probability import (
    _choose_play,
    _rebased_copy,
    estimate_bid_probability,
    tile_hold_probability,
    wilson_interval,
)
from eye42.engine.scoring import TOTAL_HAND_POINTS
from eye42.engine.tiles import Tile
from eye42.engine.trick import Trick


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
    result = estimate_bid_probability(hand)
    assert result.probability == 0.0
    # A deterministic short-circuit: no sampling, interval collapsed onto the point.
    assert result.samples == 0
    assert result.confidence_interval == (0.0, 0.0)
    # Nothing was simulated, so there is no *projected final* defending total to
    # report -- and the running tally (27 here, with tricks still unplayed) is a
    # different quantity from what every sampled/simulated path returns.
    assert hand.scorer.defending_team_points == 27  # the running tally exists...
    assert len(hand.tricks) < 7  # ...but the hand is not over, so it isn't final
    assert result.expected_defense_points is None


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
    result = estimate_bid_probability(hand)
    assert result.probability == 1.0
    assert result.samples == 0
    assert result.confidence_interval == (1.0, 1.0)
    # Same semantics as the locked-set short-circuit: guaranteed *win*, but the
    # final point split would need the remaining 4 tricks simulated.
    assert result.expected_defense_points is None


def test_completed_hand_reports_its_actual_final_defense_points():
    """The seven-trick short-circuit is *not* a "would need simulation" case --
    there is nothing left to project, so the running tally is the final figure
    and must be reported as a real number.

    Uses a marks (sweep-required) contract deliberately: for a POINTS contract
    the seventh trick always lands on the locked-set or already-guaranteed
    branch first, so a swept sweep-contract is the way to actually reach this
    branch.
    """
    hand = HandState(dealer=3)
    hand.bid(0, marks=1)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)

    leads = [Tile.of(6, n) for n in (6, 5, 4, 3, 2, 1, 0)]
    fillers = [
        [Tile.of(0, 0), Tile.of(1, 0), Tile.of(1, 1)],
        [Tile.of(2, 0), Tile.of(2, 1), Tile.of(2, 2)],
        [Tile.of(3, 0), Tile.of(3, 1), Tile.of(3, 3)],
        [Tile.of(3, 2), Tile.of(4, 0), Tile.of(4, 2)],
        [Tile.of(4, 3), Tile.of(4, 4), Tile.of(5, 1)],
        [Tile.of(4, 1), Tile.of(5, 2), Tile.of(5, 3)],
        [Tile.of(5, 4), Tile.of(5, 5), Tile.of(5, 0)],
    ]
    for lead, (f1, f2, f3) in zip(leads, fillers):
        for player, tile in [(0, lead), (1, f1), (2, f2), (3, f3)]:
            hand.play_tile(TilePlayed(player=player, tile=tile))

    assert len(hand.tricks) == 7
    result = estimate_bid_probability(hand)
    assert result.samples == 0
    assert result.expected_defense_points == float(hand.scorer.defending_team_points)


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


# ---------------------------------------------------------------------------
# deal sampler: most-constrained-first ordering
# ---------------------------------------------------------------------------

def _trump_led_first_trick_hand() -> HandState:
    """The scenario that used to exhaust the sampler: an entirely ordinary
    trick 1 led in trump with all three other seats showing off, so they are
    each void in trump. 24 unseen tiles, 24 slots -- the sum-based pre-flight
    check does *not* catch this; only the eligible-count ordering does.

    Under the old hand-size ordering all four seats tie on 6 tiles, the
    unconstrained leader sorts first, and a valid deal only comes out if it
    happens to draw all 6 remaining trumps (1 in 134,596 per attempt) -- so all
    200 attempts failed and it raised RuntimeError.
    """
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
    return hand


def test_all_other_seats_void_in_trump_still_samples():
    hand = _trump_led_first_trick_hand()
    assert hand.voids[1] == {6} and hand.voids[2] == {6} and hand.voids[3] == {6}
    assert hand.remaining_hand_size(0) == 6  # sum of hand sizes == unseen count

    result = estimate_bid_probability(hand, samples=50, rng=random.Random(7))
    assert result is not None  # used to raise RuntimeError out of the sampler
    assert isinstance(result.probability, float)
    assert 0.0 <= result.probability <= 1.0
    assert result.samples == 50  # every attempt satisfied, not just some


def test_mid_trick_estimate_does_not_crash_and_is_sane():
    """Called with a trick in progress, the estimator continues the real
    in-flight trick rather than assuming a clean trick boundary."""
    hand = _trump_led_first_trick_hand()
    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 5)))
    hand.play_tile(TilePlayed(player=1, tile=Tile.of(1, 1)))
    assert hand._current_trick is not None
    assert len(hand._current_trick.plays) == 2  # genuinely mid-trick

    result = estimate_bid_probability(hand, samples=50, rng=random.Random(7))
    assert result is not None
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.expected_defense_points <= TOTAL_HAND_POINTS


# ---------------------------------------------------------------------------
# playout policy
# ---------------------------------------------------------------------------

def _trick_with(trump: int, leader: int, plays) -> Trick:
    trick = Trick(leader=leader, trump=trump)
    for player, tile in plays:
        trick.play(player, tile)
    return trick


def test_policy_does_not_overtrump_a_trick_it_cannot_win():
    """Seat 2 holds a 10-count trump that cannot beat the trump already on the
    table. Burning it is a pure loss -- shed the cheap tile instead."""
    trick = _trick_with(6, 0, [(0, Tile.of(3, 3)), (1, Tile.of(6, 5))])
    hand_tiles = [Tile.of(6, 4), Tile.of(2, 1)]
    assert _choose_play(trick, 2, hand_tiles, trump=6) == Tile.of(2, 1)


def test_policy_does_win_when_the_candidate_actually_beats_the_table():
    trick = _trick_with(6, 0, [(0, Tile.of(3, 3)), (1, Tile.of(6, 5))])
    hand_tiles = [Tile.of(6, 6), Tile.of(2, 1)]
    assert _choose_play(trick, 2, hand_tiles, trump=6) == Tile.of(6, 6)


def test_policy_dumps_count_on_partner_only_when_last_to_act():
    """Seat 3's partner (seat 1) has the trick and seat 3 is the 4th and final
    play -- nobody can take it back, so the 5-count is safe to shed there."""
    trick = _trick_with(6, 0, [(0, Tile.of(3, 0)), (1, Tile.of(3, 3)), (2, Tile.of(3, 1))])
    assert trick.winner == 1
    assert len(trick.plays) == 3
    hand_tiles = [Tile.of(4, 1), Tile.of(2, 0)]  # 5-count and 0-count
    assert _choose_play(trick, 3, hand_tiles, trump=6) == Tile.of(4, 1)


def test_policy_does_not_dump_count_on_partner_with_an_opponent_still_to_play():
    """Same shape, but seat 2 acts third: its partner (seat 0) is only
    provisionally winning, since seat 3 still plays after. Dumping a counter
    here is a real strategic error, not a harmless simplification."""
    trick = _trick_with(6, 0, [(0, Tile.of(3, 3)), (1, Tile.of(3, 0))])
    assert trick.winner == 0
    assert len(trick.plays) == 2
    hand_tiles = [Tile.of(4, 1), Tile.of(2, 0)]
    assert _choose_play(trick, 2, hand_tiles, trump=6) == Tile.of(2, 0)


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def test_wilson_interval_is_bounded_and_contains_the_point_estimate():
    for successes, n in [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (500, 1000)]:
        low, high = wilson_interval(successes, n)
        assert 0.0 <= low <= high <= 1.0
        assert low <= successes / n <= high


def test_wilson_interval_narrows_with_more_samples():
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_handles_zero_samples():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_sampled_estimate_reports_an_interval_around_its_point_value():
    hand = _trump_led_first_trick_hand()
    result = estimate_bid_probability(hand, samples=40, rng=random.Random(3))
    low, high = result.confidence_interval
    assert 0.0 <= low <= result.probability <= high <= 1.0
    assert result.samples == 40


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_repeated_calls_on_unchanged_state_are_identical():
    """No explicit rng: the seed is derived from the hand's own play history, so
    the displayed number must not jitter between calls (or processes)."""
    hand = _trump_led_first_trick_hand()
    first = estimate_bid_probability(hand, samples=40)
    second = estimate_bid_probability(hand, samples=40)
    assert first == second

    held_first = tile_hold_probability(hand, Tile.of(5, 5), player=1, samples=40)
    held_second = tile_hold_probability(hand, Tile.of(5, 5), player=1, samples=40)
    assert held_first == held_second


def test_explicit_rng_is_still_honored():
    hand = _trump_led_first_trick_hand()
    a = estimate_bid_probability(hand, samples=40, rng=random.Random(42))
    b = estimate_bid_probability(hand, samples=40, rng=random.Random(42))
    assert a == b


# ---------------------------------------------------------------------------
# tile_hold_probability analytic short-circuits
# ---------------------------------------------------------------------------

def test_tile_hold_short_circuits_without_sampling():
    """samples=0 makes the sampling path impossible -- so a real answer here
    proves it came from the analytic void check, not Monte Carlo."""
    hand = _trump_led_first_trick_hand()
    # Seats 1-3 are void in trump, so the remaining trumps can only be seat 0's.
    assert tile_hold_probability(hand, Tile.of(6, 5), player=0, samples=0) == 1.0
    assert tile_hold_probability(hand, Tile.of(6, 5), player=2, samples=0) == 0.0


def test_tile_hold_falls_back_to_sampling_when_genuinely_ambiguous():
    hand = _trump_led_first_trick_hand()
    # 5-5 is not trump and nobody is void in 5s: genuinely ambiguous.
    value = tile_hold_probability(hand, Tile.of(5, 5), player=1, samples=100, rng=random.Random(5))
    assert value is not None
    assert 0.0 < value < 1.0


def test_tile_hold_for_a_seat_with_no_tiles_left_is_zero():
    hand = _trump_led_first_trick_hand()
    for player, tile in [
        (0, Tile.of(6, 5)), (1, Tile.of(1, 1)), (2, Tile.of(2, 0)), (3, Tile.of(2, 2))
    ]:
        hand.play_tile(TilePlayed(player=player, tile=tile))
    assert hand.remaining_hand_size(0) == 5
    # A still-unseen trump can only be seat 0's; seat 1 is void in trump.
    assert tile_hold_probability(hand, Tile.of(6, 4), player=1, samples=0) == 0.0


# ---------------------------------------------------------------------------
# misdeal guard ordering
# ---------------------------------------------------------------------------

def test_misdeal_guard_does_not_block_an_exact_answer():
    """A suspected misdeal must not suppress an answer that needs no sampling at
    all -- an observed play is ground truth regardless of the deal's bookkeeping."""
    hand = HandState(dealer=3)
    hand.bid(0, 30)
    hand.bid_pass(1)
    hand.bid_pass(2)
    hand.bid_pass(3)
    hand.call_trump(0, trump=6)
    hand.record_deal(TilesDealt(dealer=3, counts={0: 8, 1: 7, 2: 7, 3: 6}))
    assert hand.misdeal_suspected is True

    hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
    assert tile_hold_probability(hand, Tile.of(6, 6), player=0) == 1.0
    assert tile_hold_probability(hand, Tile.of(6, 6), player=1) == 0.0


def test_misdeal_suppresses_the_sampled_estimate():
    hand = _trump_led_first_trick_hand()
    hand.record_deal(TilesDealt(dealer=3, counts={0: 8, 1: 7, 2: 7, 3: 6}))
    assert hand.misdeal_suspected is True
    assert estimate_bid_probability(hand, samples=10) is None


# ---------------------------------------------------------------------------
# voids recorded under a *different* trump hypothesis
# ---------------------------------------------------------------------------

def _hand_with_voids_under_a_stale_trump() -> HandState:
    """Trick 1 led in trump 6 records voids[1..3] == {6} under ``_voids_trump ==
    6``. The confirmed trump then changes value *mid-trick*: the legitimate
    trump-caller calls trump again with a different value, which hard-re-
    confirms it (there is no guard against calling twice), so ``HandState`` has
    not yet reached the trick close where it would drop the stale voids."""
    hand = _trump_led_first_trick_hand()
    assert hand._voids_trump == 6
    assert hand.voids[1] == {6}
    hand.call_trump(0, trump=3)
    assert hand._voids_trump == 6  # not cleared until the next trick closes
    return hand


def test_stale_voids_do_not_decide_tile_holdings_under_a_changed_trump():
    """Under trump 3 nobody has been shown void in anything -- seats 1-3 played
    off suit 6, which is only a suit-6 void if 6 is trump. The analytic
    short-circuits must not fire off a rejected hypothesis's constraints."""
    hand = _hand_with_voids_under_a_stale_trump()
    # samples=0 makes sampling impossible, so any non-None answer here could
    # only have come from the (now inapplicable) void short-circuits.
    assert tile_hold_probability(hand, Tile.of(6, 5), player=0, samples=0) is None
    assert tile_hold_probability(hand, Tile.of(6, 5), player=2, samples=0) is None


def test_stale_voids_are_ignored_by_the_sampler_too():
    """With sampling allowed, the same tile comes back genuinely ambiguous
    rather than a confident 1.0/0.0 -- every seat can still hold it under 3."""
    hand = _hand_with_voids_under_a_stale_trump()
    value = tile_hold_probability(hand, Tile.of(6, 5), player=2, samples=200, rng=random.Random(11))
    assert value is not None
    assert 0.0 < value < 1.0

    result = estimate_bid_probability(hand, samples=50, rng=random.Random(11))
    assert result is not None
    assert result.samples == 50  # the false constraints would have starved this


def test_voids_still_apply_under_the_trump_they_were_recorded_for():
    """The guard must not throw away legitimate voids -- same fixture, trump
    unchanged, still deterministic."""
    hand = _trump_led_first_trick_hand()
    assert tile_hold_probability(hand, Tile.of(6, 5), player=0, samples=0) == 1.0
    assert tile_hold_probability(hand, Tile.of(6, 5), player=2, samples=0) == 0.0


# ---------------------------------------------------------------------------
# unsatisfiable-void bailout (no hang on a live display)
# ---------------------------------------------------------------------------

def test_unsatisfiable_voids_bail_out_quickly_instead_of_hanging():
    """A seat void in every suit can hold nothing, so no deal exists. Without a
    consecutive-failure bailout each of the (default 1000) samples independently
    burns MAX_DEAL_ATTEMPTS shuffles -- ~14 seconds of a frozen display."""
    hand = _trump_led_first_trick_hand()
    hand.voids[1] = set(range(7))

    started = time.perf_counter()
    assert estimate_bid_probability(hand) is None
    assert tile_hold_probability(hand, Tile.of(5, 5), player=2) is None
    elapsed = time.perf_counter() - started
    # Generous versus the ~26s the unbailed version takes for both calls; the
    # point is the order of magnitude, not a tight benchmark on CI hardware.
    assert elapsed < 3.0


def test_the_bailout_does_not_fire_on_a_satisfiable_state():
    """The accuracy half of the fix: a perfectly ordinary (and, per
    _trump_led_first_trick_hand, void-constrained) state must still deliver the
    full requested sample count, not a truncated one."""
    hand = _trump_led_first_trick_hand()
    result = estimate_bid_probability(hand, samples=1000, rng=random.Random(2))
    assert result is not None
    assert result.samples == 1000

    value = tile_hold_probability(hand, Tile.of(5, 5), player=1, samples=1000, rng=random.Random(2))
    assert value is not None


# ---------------------------------------------------------------------------
# mid-trick simulation runs under the trump the estimate is computed with
# ---------------------------------------------------------------------------

def test_mid_trick_copy_is_rebased_onto_the_estimating_trump():
    """``_rebased_copy`` is what keeps Trick.legal_plays/Trick.winner deciding
    under the same trump the playout policy ranks with."""
    trick = _trick_with(6, 0, [(0, Tile.of(3, 3)), (1, Tile.of(6, 5))])
    assert trick.led_suit == 3

    same = _rebased_copy(trick, 6)
    assert same is not trick and same.trump == 6

    # 3-3 is a double, so it still leads suit 3 under trump 5: safely rebasable.
    rebased = _rebased_copy(trick, 5)
    assert rebased is not None
    assert rebased.trump == 5
    assert rebased.led_suit == 3
    assert trick.trump == 6  # the real trick is untouched


def test_mid_trick_copy_bails_when_the_led_suit_cannot_be_rebased():
    """Led 6-5 under trump 6 leads suit 6; under trump 5 that very tile would be
    trump instead. Rewriting the led suit would retroactively change which
    already-played tiles followed, so the estimate goes unavailable."""
    trick = _trick_with(6, 0, [(0, Tile.of(6, 5))])
    assert trick.led_suit == 6
    assert _rebased_copy(trick, 5) is None


# ---------------------------------------------------------------------------
# bug-hunt fix pass: sampler failure budget (B2), seat guard (B6), Wilson
# clamp (B7)
# ---------------------------------------------------------------------------

def test_wilson_interval_clamps_out_of_range_success_counts():
    """B7. ``successes`` outside [0, n] reached ``math.sqrt`` of a negative and
    raised a ValueError out of a public, directly-tested helper."""
    assert wilson_interval(15, 10) == wilson_interval(10, 10)
    assert wilson_interval(-3, 10) == wilson_interval(0, 10)
    low, high = wilson_interval(15, 10)
    assert 0.0 <= low <= high <= 1.0


def test_tile_hold_probability_returns_none_for_an_out_of_range_seat():
    """B6. An out-of-range seat is API misuse, not a table event -- this
    module's contract for "can't answer that" is ``None``, never an exception."""
    hand = _trump_led_first_trick_hand()
    for player in (-1, 4, 99):
        assert tile_hold_probability(hand, Tile.of(6, 1), player=player) is None


def test_total_failure_budget_bails_out_when_failures_are_interleaved():
    """B2. The consecutive-failure guard resets on every success, so a void
    shape that fails most attempts but succeeds occasionally never tripped it --
    measured at 9.7s for one default-``samples`` call, the exact live-display
    hang that constant exists to prevent. A total-failure bound catches it."""
    calls = {"n": 0}
    real = probability._try_sample

    def flaky(*args, **kwargs):
        calls["n"] += 1
        # Fail three out of every four attempts, so the consecutive counter is
        # reset before it can ever reach MAX_CONSECUTIVE_SAMPLE_FAILURES.
        if calls["n"] % 4:
            return None
        return real(*args, **kwargs)

    hand = _trump_led_first_trick_hand()
    probability._try_sample = flaky
    try:
        result = estimate_bid_probability(hand, samples=400)
    finally:
        probability._try_sample = real

    assert calls["n"] < 400  # bailed out early instead of grinding to the end
    # ...and the samples it did collect are still reported, not thrown away.
    assert result is not None
    assert 0 < result.samples < 400


def test_the_failure_budget_still_returns_none_when_nothing_is_satisfiable():
    """The partial-result path must not turn a genuinely unsatisfiable state
    into a fabricated estimate: with no usable samples at all, it is still
    ``None``."""
    hand = _trump_led_first_trick_hand()
    real = probability._try_sample
    probability._try_sample = lambda *a, **k: None
    try:
        assert estimate_bid_probability(hand, samples=400) is None
        # A non-trump tile, so no void-based analytic short-circuit answers it
        # first and the question really does reach the sampler.
        assert tile_hold_probability(hand, Tile.of(5, 4), player=1, samples=400) is None
    finally:
        probability._try_sample = real
