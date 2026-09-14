"""Aligns two ordered ``TilePlayed`` event sequences for manual review."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Tuple

PlayKey = Tuple[Any, int, int, int]


def _play_key(payload: Dict[str, Any]) -> PlayKey:
    tile = payload["tile"]
    return (payload.get("hand_index"), payload["player"], tile["high"], tile["low"])


def diff_play_sequences(original: List[Dict[str, Any]], replay: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aligns two ordered ``TilePlayed`` payload sequences and reports where
    they agree/diverge. Both sequences may come from the same detector run
    at different times against different frame streams, so this is not a
    ground-truth accuracy check -- it surfaces candidates for manual review."""
    a = [_play_key(p) for p in original]
    b = [_play_key(p) for p in replay]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    only_in_original: List[PlayKey] = []
    only_in_replay: List[PlayKey] = []
    matched = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        else:
            only_in_original.extend(a[i1:i2])
            only_in_replay.extend(b[j1:j2])
    return {"matched": matched, "only_in_original": only_in_original, "only_in_replay": only_in_replay}
