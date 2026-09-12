"""Single-trick play: legality and winner determination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .tiles import Tile, effective_suit, follows_suit, trick_rank


class IllegalPlayError(ValueError):
    pass


@dataclass
class Trick:
    leader: int
    trump: int
    plays: List[Tuple[int, Tile]] = field(default_factory=list)
    _led_suit: Optional[int] = field(default=None, init=False)

    @property
    def led_suit(self) -> Optional[int]:
        return self._led_suit

    @property
    def is_complete(self) -> bool:
        return len(self.plays) == 4

    @property
    def next_player(self) -> Optional[int]:
        if self.is_complete:
            return None
        return (self.leader + len(self.plays)) % 4

    def legal_plays(self, hand: Sequence[Tile]) -> List[Tile]:
        """Which tiles in ``hand`` are legal to play right now."""
        if self._led_suit is None:
            return list(hand)  # leading: anything is legal
        followers = [t for t in hand if follows_suit(t, self.trump, self._led_suit)]
        return followers if followers else list(hand)

    def play(self, player: int, tile: Tile, hand: Optional[Sequence[Tile]] = None) -> None:
        expected = self.next_player
        if expected is None:
            raise IllegalPlayError("trick already has 4 plays")
        if player != expected:
            raise IllegalPlayError(f"expected seat {expected} to play next, got {player}")

        if self._led_suit is None:
            self._led_suit = effective_suit(tile, self.trump, None)
        elif hand is not None and tile not in self.legal_plays(hand):
            raise IllegalPlayError(
                f"seat {player} must follow suit {self._led_suit} if able; "
                f"played {tile} instead"
            )

        self.plays.append((player, tile))

    @property
    def winner(self) -> Optional[int]:
        if not self.plays:
            return None
        assert self._led_suit is not None
        best_player, best_tile = self.plays[0]
        best_rank = trick_rank(best_tile, self.trump, self._led_suit)
        for player, tile in self.plays[1:]:
            rank = trick_rank(tile, self.trump, self._led_suit)
            if rank > best_rank:
                best_player, best_tile, best_rank = player, tile, rank
        return best_player

    @property
    def count_value(self) -> int:
        return sum(t.count_value for _, t in self.plays)
