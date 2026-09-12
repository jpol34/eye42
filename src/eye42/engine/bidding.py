"""Bidding rotation for a Texas 42 hand, including splash/plunge and forced-30.

House rules modeled (confirmed for this table): standard point bids, splash (3
doubles, minimum 2 marks), plunge (4 doubles, minimum 4 marks), all-pass forces the
dealer to bid 30 rather than a redeal. No nello, no sevens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .events import BidMade, Passed


# House-rule thresholds for the two doubles-based marks bids. Named constants
# rather than inline literals so any caller of ``upgrade_to_splash_or_plunge``
# can check them against a bid before calling it (the method raises if they
# aren't met).
SPLASH_MIN_DOUBLES = 3
SPLASH_MIN_MARKS = 2
PLUNGE_MIN_DOUBLES = 4
PLUNGE_MIN_MARKS = 4


class BidKind(Enum):
    POINTS = auto()  # 30-41 point bid; contract is reaching that many points
    MARKS = auto()  # a plain marks bid; contract is winning all 7 tricks
    SPLASH = auto()  # 3-doubles marks bid, min 2 marks, partner calls trump/leads
    PLUNGE = auto()  # 4-doubles marks bid, min 4 marks, partner calls trump/leads


@dataclass(frozen=True)
class Contract:
    bidder: int
    kind: BidKind
    amount: int  # for POINTS: 30-41 (or 42). for MARKS/SPLASH/PLUNGE: number of marks.

    @property
    def trump_caller(self) -> int:
        """Seat that names trump and leads the first trick."""
        if self.kind in (BidKind.SPLASH, BidKind.PLUNGE):
            return partner_of(self.bidder)
        return self.bidder

    @property
    def requires_sweep(self) -> bool:
        """True if the contract requires winning every trick (marks-style bids)."""
        return self.kind in (BidKind.MARKS, BidKind.SPLASH, BidKind.PLUNGE)

    @property
    def points_needed(self) -> int:
        """Points (out of 42) the bidding team must secure to make the contract."""
        return 42 if self.requires_sweep else self.amount

    @property
    def marks_at_stake(self) -> int:
        if self.kind == BidKind.POINTS:
            return 1
        return self.amount


def partner_of(player: int) -> int:
    """Partners sit across the table: seats 0&2 and 1&3."""
    return (player + 2) % 4


def team_of(player: int) -> int:
    """0 or 1 — which partnership a seat belongs to."""
    return player % 2


class BiddingError(ValueError):
    pass


@dataclass
class BiddingRound:
    dealer: int
    _rotation: List[int] = field(init=False)
    _next_idx: int = field(default=0, init=False)
    _high_bid: Optional[BidMade] = field(default=None, init=False)
    _high_bid_kind_amount: Optional[tuple[BidKind, int]] = field(default=None, init=False)
    _acted: int = field(default=0, init=False)  # count of players who have bid/passed
    _passed: set = field(default_factory=set, init=False)
    forced_thirty: bool = field(default=False, init=False)
    contract: Optional[Contract] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rotation = [(self.dealer + 1 + i) % 4 for i in range(4)]

    @property
    def current_bidder(self) -> Optional[int]:
        if self._acted >= 4 or self.contract is not None:
            return None
        return self._rotation[self._acted]

    @property
    def is_done(self) -> bool:
        return self.contract is not None

    def _bid_kind_and_amount(self, bid: BidMade) -> tuple[BidKind, int]:
        if bid.marks and bid.marks >= 1:
            # Splash/plunge are distinguished by tile composition, which the
            # bidding round itself doesn't see — the caller (engine.hand) passes
            # that in via record_marks_bid when it knows the hand's doubles.
            return BidKind.MARKS, bid.marks
        if not (30 <= bid.amount <= 42):
            raise BiddingError(f"point bid must be 30-42, got {bid.amount}")
        return BidKind.POINTS, bid.amount

    def _rank(self, kind: BidKind, amount: int) -> int:
        """Comparable strength of a bid for 'must exceed current high bid'."""
        if kind == BidKind.POINTS:
            return amount
        return 42 + amount  # any marks bid outranks any point bid

    def record_bid(self, bid: BidMade, *, kind: Optional[BidKind] = None) -> None:
        if self.is_done:
            raise BiddingError("bidding already closed")
        expected = self.current_bidder
        if expected is None:
            raise BiddingError("no more bids expected this round")
        if bid.player != expected:
            raise BiddingError(f"expected bid from seat {expected}, got seat {bid.player}")

        inferred_kind, amount = self._bid_kind_and_amount(bid)
        bid_kind = kind or inferred_kind
        new_rank = self._rank(bid_kind, amount)
        if self._high_bid_kind_amount is not None:
            prev_kind, prev_amount = self._high_bid_kind_amount
            if new_rank <= self._rank(prev_kind, prev_amount):
                raise BiddingError("bid must exceed the current high bid")

        self._high_bid = bid
        self._high_bid_kind_amount = (bid_kind, amount)
        self._acted += 1
        self._maybe_close(bid.player, bid_kind, amount)

    def record_pass(self, passed: Passed) -> None:
        if self.is_done:
            raise BiddingError("bidding already closed")
        expected = self.current_bidder
        if expected is None:
            raise BiddingError("no more actions expected this round")
        if passed.player != expected:
            raise BiddingError(f"expected action from seat {expected}, got seat {passed.player}")

        self._passed.add(passed.player)
        self._acted += 1

        if self._acted == 4 and self._high_bid is None:
            # All four passed: dealer is forced to bid 30 (confirmed house rule).
            self.forced_thirty = True
            self.contract = Contract(bidder=self.dealer, kind=BidKind.POINTS, amount=30)
            return

        if self._acted == 4:
            self._close_with_high_bid()

    def _maybe_close(self, last_bidder: int, kind: BidKind, amount: int) -> None:
        if self._acted == 4:
            self._close_with_high_bid()

    def _close_with_high_bid(self) -> None:
        assert self._high_bid is not None and self._high_bid_kind_amount is not None
        kind, amount = self._high_bid_kind_amount
        self.contract = Contract(bidder=self._high_bid.player, kind=kind, amount=amount)

    def upgrade_to_splash_or_plunge(self, kind: BidKind) -> None:
        """Reclassify a marks bid as splash/plunge once the hand's doubles are
        known (the bidding round alone can't tell these apart from a plain marks
        bid — that requires knowing the bidder's tiles)."""
        if self.contract is None or self.contract.kind != BidKind.MARKS:
            raise BiddingError("can only upgrade an unresolved marks contract")
        if kind not in (BidKind.SPLASH, BidKind.PLUNGE):
            raise BiddingError("upgrade target must be SPLASH or PLUNGE")
        min_marks = SPLASH_MIN_MARKS if kind == BidKind.SPLASH else PLUNGE_MIN_MARKS
        if self.contract.amount < min_marks:
            raise BiddingError(f"{kind.name} requires at least {min_marks} marks")
        self.contract = Contract(bidder=self.contract.bidder, kind=kind, amount=self.contract.amount)
