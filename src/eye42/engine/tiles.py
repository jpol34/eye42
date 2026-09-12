"""Domino tile primitives for a double-six set and Texas 42 suit/trump rules.

A non-double tile belongs to two suits (its two pip counts) at once — which one is
"in play" for a given trick depends on what's trump and what was led. See
``effective_suit`` for that resolution; ``suits_of``/``is_trump`` are the raw facts
a tile carries independent of any trick context.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import FrozenSet, Optional


@dataclass(frozen=True, order=True)
class Tile:
    """A domino tile, stored with the higher pip count first (canonical form)."""

    high: int
    low: int

    def __post_init__(self) -> None:
        if not (0 <= self.low <= self.high <= 6):
            raise ValueError(f"invalid tile ends: {self.high}-{self.low}")

    @classmethod
    def of(cls, a: int, b: int) -> "Tile":
        return cls(high=max(a, b), low=min(a, b))

    @property
    def is_double(self) -> bool:
        return self.high == self.low

    @property
    def pip_total(self) -> int:
        return self.high + self.low

    @property
    def count_value(self) -> int:
        """Points this tile is worth toward the 35 count-points (0 if not a counter)."""
        if self in (Tile(5, 0), Tile(4, 1), Tile(3, 2)):
            return 5
        if self in (Tile(6, 4), Tile(5, 5)):
            return 10
        return 0

    def __str__(self) -> str:
        return f"{self.high}-{self.low}"


def full_set() -> FrozenSet[Tile]:
    """All 28 tiles of a double-six set."""
    return frozenset(Tile.of(a, b) for a, b in combinations_with_replacement(range(7), 2))


def suits_of(tile: Tile) -> FrozenSet[int]:
    """The suit(s) a tile belongs to, independent of trump.

    A double belongs to exactly one suit (its own number). A non-double belongs to
    both of its numbers — which one actually governs a given trick depends on trump
    and what was led; see ``effective_suit``.
    """
    if tile.is_double:
        return frozenset({tile.high})
    return frozenset({tile.high, tile.low})


def is_trump(tile: Tile, trump: int) -> bool:
    """True if either end of the tile matches the trump number."""
    return trump in suits_of(tile)


def effective_suit(tile: Tile, trump: int, led_suit: Optional[int] = None) -> int:
    """The suit this tile counts as for the purpose of following/winning a trick.

    - If the tile is trump, its effective suit is always ``trump``.
    - Otherwise, if the tile can follow the led suit (one of its ends matches
      ``led_suit``), its effective suit is ``led_suit``.
    - Otherwise (off-suit, non-trump, no led suit context, or leading), a
      non-double's effective suit defaults to its higher end; a double is its own
      number.
    """
    if is_trump(tile, trump):
        return trump
    if led_suit is not None and led_suit in suits_of(tile):
        return led_suit
    return tile.high


def follows_suit(tile: Tile, trump: int, led_suit: int) -> bool:
    """True if this tile actually counts as the led suit once trump is applied.

    A trump tile only counts as following ``led_suit`` when trump itself was led
    (led_suit == trump) — otherwise a trump tile is never "the led suit," it's a
    trump play, which is only legal when the player holds no tile that follows.
    Whole-hand legality (must-follow-if-able) lives in the engine, not here.
    """
    return effective_suit(tile, trump, led_suit) == led_suit


_DOUBLE_SENTINEL = 7  # ranks above any real pip count (0-6) within its suit


def trick_rank(tile: Tile, trump: int, led_suit: int) -> tuple[int, int]:
    """Sort key for comparing tiles within one trick: higher wins.

    Ordering: trump beats everything; among trump, the double ranks highest, then
    by the non-trump end's pip count descending. Among non-trump tiles that follow
    the led suit, the same pattern applies relative to the led suit. A tile that
    neither trumps nor follows cannot win the trick.
    """
    if is_trump(tile, trump):
        other_end = _DOUBLE_SENTINEL if tile.is_double else (
            tile.low if tile.high == trump else tile.high
        )
        return (2, other_end)
    if led_suit in suits_of(tile):
        other_end = _DOUBLE_SENTINEL if tile.is_double else (
            tile.low if tile.high == led_suit else tile.high
        )
        return (1, other_end)
    return (0, -1)
