"""Local SQLite (WAL-mode) sink for engine events and irregularities.

Persists whatever dataclass crosses ``HandState.ingest`` or gets logged through
``RepairLog`` generically -- via ``dataclasses.asdict`` plus the class name as a
type tag -- so a new event or irregularity kind added to ``engine.events`` /
``engine.repair`` is captured automatically, with no per-kind serialization code
to add here. Every row is optionally correlated to a saved camera frame, so a
later review can line up an engine decision with what the camera actually saw.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .engine.repair import Conflict, Irregularity


def _payload(obj: object) -> Dict[str, Any]:
    """Serialize a real ``events.py``/``repair.py`` dataclass generically, or
    accept a plain dict for a synthetic, non-dataclass control row (e.g. the
    live-session harness's "HandEnded"/"DealerObserved" markers)."""
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(vars(obj))


class EventStore:
    """One SQLite file per install; many sessions/hands recorded into it."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # One connection is shared across the REPL thread and Flask's request
        # thread (tools/live_view.py) -- sqlite3 doesn't serialize concurrent
        # statement execution on one connection across threads on its own.
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                hand_index INTEGER,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                frame_path TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS irregularities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                hand_index INTEGER,
                kind TEXT NOT NULL,
                reason TEXT,
                player INTEGER,
                needs_confirmation INTEGER,
                confidence REAL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # ---- writes ---------------------------------------------------------

    def log_event(
        self,
        event: object,
        *,
        session_id: str,
        hand_index: Optional[int] = None,
        frame_path: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> None:
        """Generic write for any dataclass event -- including synthetic,
        non-``events.py`` control rows (``kind="HandEnded"`` etc.) a caller
        constructs by hand rather than passing a real dataclass."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (session_id, ts, hand_index, kind, payload_json, frame_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    time.time(),
                    hand_index,
                    kind or type(event).__name__,
                    json.dumps(_payload(event), default=str),
                    frame_path,
                ),
            )
            self._conn.commit()

    def log_irregularity(
        self, item: Union[Irregularity, Conflict], *, session_id: str, hand_index: Optional[int] = None
    ) -> None:
        payload = _payload(item)
        if isinstance(item, Conflict):
            kind, reason, player = "Conflict", item.reason, None
            needs_confirmation, confidence = None, None
        else:
            kind = item.kind.name
            reason = item.reason
            player = item.player
            needs_confirmation = int(item.needs_confirmation)
            confidence = item.confidence
        with self._lock:
            self._conn.execute(
                "INSERT INTO irregularities "
                "(session_id, ts, hand_index, kind, reason, player, needs_confirmation, confidence, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    time.time(),
                    hand_index,
                    kind,
                    reason,
                    player,
                    needs_confirmation,
                    confidence,
                    json.dumps(payload, default=str),
                ),
            )
            self._conn.commit()

    # ---- reads ------------------------------------------------------------

    def recent(self, session_id: str, *, since_ts: Optional[float] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Tail of both tables, merged and ordered oldest-to-newest -- the log
        panel's feed.

        ``since_ts`` uses ``>=``, not ``>``: timestamp resolution (coarse on
        Windows) means a burst of same-tick rows is possible, and a strict
        ``>`` would silently drop any row exactly at the caller's last-seen
        cursor. A row at the boundary can therefore repeat across polls; the
        caller (tools/static/live_view.html) dedups by (table, id).
        """
        rows: List[Dict[str, Any]] = []
        with self._lock:
            for table in ("events", "irregularities"):
                clause = "WHERE session_id = ?" + (" AND ts >= ?" if since_ts is not None else "")
                params: tuple = (session_id, since_ts) if since_ts is not None else (session_id,)
                cur = self._conn.execute(
                    f"SELECT * FROM {table} {clause} ORDER BY ts DESC LIMIT ?", (*params, limit)
                )
                cols = [d[0] for d in cur.description]
                rows.extend({**dict(zip(cols, row)), "table": table} for row in cur.fetchall())
        rows.sort(key=lambda r: r["ts"])
        return rows[-limit:]

    def events_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Full, ordered event history for a session -- the deterministic
        replay tape (``tools/replay.py``). Irregularities are derived output,
        not input, so they're excluded here on purpose."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY id ASC", (session_id,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def repair_sink(
    store: "EventStore", session_id: str, hand_index_fn: Callable[[], Optional[int]]
) -> Callable[[object], None]:
    """Builds a ``RepairLog`` sink bound to a store/session. ``hand_index_fn``
    is called on every write rather than resolved once, since a caller like
    ``GameState`` computes its hand index (``len(hands_played)``) freshly each
    time -- a captured value would go stale after the first hand."""

    def _sink(item: object) -> None:
        store.log_irregularity(item, session_id=session_id, hand_index=hand_index_fn())

    return _sink
