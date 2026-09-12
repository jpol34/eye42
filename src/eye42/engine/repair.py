"""Provenance-tracked conflict/repair log for noisy perception input.

Corrections are logged as reversible proposals, never applied as silent in-place
overwrites — an edit made under a later-proven-wrong trump hypothesis must be
undoable, not permanent. This module currently covers the plumbing (recording
conflicts with enough context to investigate/undo); the full hand-level
backtracking re-solve described in the plan is future work once real perception
data exists to design against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .events import TilePlayed


@dataclass(frozen=True)
class Conflict:
    event: TilePlayed
    reason: str


@dataclass
class RepairLog:
    conflicts: List[Conflict] = field(default_factory=list)

    def log_conflict(self, event: TilePlayed, reason: str) -> None:
        self.conflicts.append(Conflict(event=event, reason=reason))

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)
