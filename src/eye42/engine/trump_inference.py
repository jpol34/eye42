"""Weighted trump-hypothesis tracking for hands where trump is never clearly stated.

There is no standardized convention for "leading a double implies trump" or
"leading a non-double implies the high/low end is trump" (confirmed by research —
this is table folklore, not a rule). This tracker treats every non-explicit signal
as evidence of varying strength rather than ground truth, narrowing a weighted set
of candidates as tricks are observed — the same shape as bridge PIMC/constraint-
based hidden-state inference: maintain candidate worlds, shift weight as evidence
arrives, never let one low-confidence observation collapse everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .tiles import Tile, suits_of

CONFIRMED_THRESHOLD = 0.9
VIABILITY_FLOOR = 0.02
LOW_CONFIDENCE_MARGIN = 0.6  # below this, a tile read is too shaky to eliminate on


@dataclass
class TrumpHypothesisTracker:
    weights: Dict[int, float] = field(default_factory=lambda: {n: 1.0 / 7 for n in range(7)})
    confirmed: Optional[int] = None
    _last_weights_before_empty: Optional[Dict[int, float]] = field(default=None, init=False)

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed is not None

    @property
    def best_guess(self) -> int:
        if self.confirmed is not None:
            return self.confirmed
        return max(self.weights, key=lambda n: self.weights[n])

    @property
    def is_ambiguous(self) -> bool:
        """True if, absent a hard confirmation, more than one candidate is still
        plausible — this is what should route to a human-confirmation prompt if
        the hand ends without resolving further."""
        if self.confirmed is not None:
            return False
        live = [w for w in self.weights.values() if w > VIABILITY_FLOOR]
        return len(live) != 1

    def confirm(self, trump: int) -> None:
        """A hard confirmation: an explicit verbal call, or a weak cue resolved
        against a known led tile. Collapses the hypothesis set immediately."""
        self.confirmed = trump
        self.weights = {n: (1.0 if n == trump else 0.0) for n in range(7)}

    def observe_cue_on_lead(self, led_tile: Tile, cue: Optional[str]) -> None:
        """Update candidates from how the first trick was led, when no explicit
        trump call was heard.

        - A weak cue ("low"/"high") plus a non-double led tile hard-resolves it.
        - A led double with no cue is a *strong but unconfirmed* hint (common
          table habit, not a rule) — weight it heavily without eliminating others.
        - A led non-double with no cue leaves both its ends live, weighted
          slightly ahead of the rest but genuinely ambiguous.
        """
        if self.confirmed is not None:
            return

        if cue is not None and not led_tile.is_double:
            trump = led_tile.low if cue == "low" else led_tile.high
            self.confirm(trump)
            return

        if led_tile.is_double:
            self._bias_toward({led_tile.high}, weight=0.85)
        else:
            self._bias_toward(set(suits_of(led_tile)), weight=0.6)

    def _bias_toward(self, candidates: set[int], weight: float) -> None:
        others = [n for n in range(7) if n not in candidates]
        share_candidates = weight / len(candidates)
        share_others = (1.0 - weight) / len(others) if others else 0.0
        for n in range(7):
            self.weights[n] = share_candidates if n in candidates else share_others

    def observe_void(self, suit_led: int, confidence: float) -> None:
        """A player failed to follow ``suit_led``. On its own this only proves
        they're void in that suit *under whatever trump turns out to be* — it is
        NOT decisive by itself (playing off-suit is always legal). The real
        signal is a later contradiction (see ``observe_contradiction``); this
        method exists for symmetry/logging and currently only records evidence
        when paired with a contradiction, since a bare void proves nothing about
        which candidate is correct.
        """
        # Intentionally a no-op beyond documentation: a void alone is consistent
        # with every trump hypothesis under which the player lacks that suit, and
        # the tracker doesn't have per-player hand knowledge to evaluate that here.
        # Callers should use observe_contradiction once a real inconsistency is
        # detected by the hand-level solve (engine.hand), which does track hands.
        return

    def observe_contradiction(self, disproven: set[int], confidence: float) -> None:
        """A player was shown to have revoked under the given trump candidate(s)
        (played off a suit they're now known to still hold) — soft-eliminate
        those candidates, gated on confidence so a single shaky tile read can't
        knock out the truth.
        """
        if self.confirmed is not None or not disproven:
            return
        if confidence < LOW_CONFIDENCE_MARGIN:
            # Don't hard-eliminate on a low-margin read; just nudge weight down.
            penalty = 0.5
        else:
            penalty = 0.05

        for n in disproven:
            self.weights[n] *= penalty

        total = sum(self.weights.values())
        if total <= VIABILITY_FLOOR:
            # Every candidate got disproven -- almost certainly an earlier bad
            # tile read poisoned the evidence. Reopen the best prior state
            # instead of leaving trump undefined.
            if self._last_weights_before_empty:
                self.weights = dict(self._last_weights_before_empty)
            else:
                self.weights = {n: 1.0 / 7 for n in range(7)}
            return

        self._last_weights_before_empty = dict(self.weights)
        for n in self.weights:
            self.weights[n] /= total

        if max(self.weights.values()) >= CONFIRMED_THRESHOLD:
            self.confirm(max(self.weights, key=lambda n: self.weights[n]))
