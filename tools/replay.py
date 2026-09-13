#!/usr/bin/env python3
"""Deterministic-replay regression check: re-feed a recorded session's event
log through a fresh engine and confirm the resulting live state matches.

Real ``events.py`` dataclasses are replayed via ``HandState.ingest``. The
live-session harness (tools/live_view.py) also logs two synthetic, non-
``events.py`` control rows -- ``"HandEnded"`` and ``"DealerObserved"`` -- which
this module replays by calling the same ``GameState``/``HandState`` methods
the harness itself calls for those actions (``classify_irregular_end`` +
``record_hand`` + ``new_hand``, and ``observe_next_dealer`` respectively), so a
multi-hand session replays across every hand boundary, not just the first.

Usage:
    python tools/replay.py <db_path> <session_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eye42.engine.events import (  # noqa: E402
    BidMade,
    IrregularEndSignal,
    Passed,
    TilePlayed,
    TilesDealt,
    TrumpCalled,
    TrumpCueHeard,
)
from eye42.engine.game import GameState  # noqa: E402
from eye42.engine.tiles import Tile  # noqa: E402
from eye42.telemetry import EventStore  # noqa: E402

_EVENT_TYPES = {
    "BidMade": BidMade,
    "Passed": Passed,
    "TrumpCalled": TrumpCalled,
    "TrumpCueHeard": TrumpCueHeard,
    "TilePlayed": TilePlayed,
    "TilesDealt": TilesDealt,
    "IrregularEndSignal": IrregularEndSignal,
}


def _rebuild_event(kind: str, payload: dict) -> object:
    cls = _EVENT_TYPES[kind]
    if cls is TilePlayed:
        payload = {**payload, "tile": Tile(**payload["tile"])}
    if cls is TilesDealt:
        payload = {**payload, "counts": {int(k): v for k, v in payload["counts"].items()}}
    return cls(**payload)


def replay(db_path: str, session_id: str):
    """Returns ``(game, current_hand)`` -- ``current_hand`` is whichever hand
    is in progress (or was last touched) when the recorded log ends."""
    import json

    store = EventStore(db_path)
    rows = store.events_for_session(session_id)
    game = GameState(session_id=session_id)
    hand = game.new_hand()
    for row in rows:
        payload = json.loads(row["payload_json"])
        if row["kind"] == "HandEnded":
            tricks_played_so_far = payload.get("tricks_played_so_far")
            if tricks_played_so_far is not None:
                hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=tricks_played_so_far))
            game.record_hand(hand)
            hand = game.new_hand()
        elif row["kind"] == "DealerObserved":
            game.observe_next_dealer(payload["observed_dealer"])
        elif row["kind"] in _EVENT_TYPES:
            event = _rebuild_event(row["kind"], payload)
            hand.ingest(event)
    store.close()
    return game, hand


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <db_path> <session_id>")
    _, hand = replay(sys.argv[1], sys.argv[2])
    print(f"replayed session {sys.argv[2]!r}: {hand.live_state()}")


if __name__ == "__main__":
    main()
