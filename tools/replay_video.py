#!/usr/bin/env python3
"""Replays a saved video file through the same perception pipeline
tools/live_view.py runs live, for reviewing detector behavior against
recorded footage without a live camera or table present.

Recorded video is not a frame-for-frame copy of what perception saw live:
``VideoRecorder`` resamples to a steady wall-clock frame rate and pads gaps
by duplicating frames, while the live consumer thread ran on its own
cadence against whatever frame the camera had most recently captured. A
replay run can diverge from the original session's log for that reason
alone, even with an unchanged detector -- ``--compare-db`` output is a
starting point for manual review, not a pass/fail regression signal.

Usage:
    python tools/replay_video.py <video_path> <calibration_path>
        [--compare-db <db_path> --session-id <original_session_id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eye42.perception.replay_diff import diff_play_sequences  # noqa: E402
from eye42.telemetry import EventStore  # noqa: E402
from live_view import Session, load_perception  # noqa: E402


class _OfflineSession(Session):
    """A ``Session`` with no camera -- replay only needs the engine/
    perception side effects, not a per-play JPEG snapshot."""

    def _snapshot_path(self):
        return None


def replay_video(
    video_path: Path,
    calibration_path: Path,
    store: EventStore,
    session_id: str,
    max_frames: Optional[int] = None,
) -> Session:
    if calibration_path.stat().st_mtime > video_path.stat().st_mtime:
        print(
            f"warning: {calibration_path} was modified after {video_path} was recorded -- "
            "replay geometry may not match what was used live"
        )

    perception = load_perception(calibration_path)
    if perception is None:
        raise SystemExit(f"no calibration at {calibration_path}")

    session = _OfflineSession(store, session_id, camera=None, frames_dir=Path("."), perception=perception)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    frame_id = 0
    try:
        while max_frames is None or frame_id < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            session.on_frame(frame, frame_id)
    finally:
        cap.release()

    return session


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {**json.loads(row["payload_json"]), "hand_index": row["hand_index"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("calibration_path", type=Path)
    parser.add_argument("--compare-db", type=Path, help="original session's SQLite DB to diff against")
    parser.add_argument("--session-id", help="original session_id within --compare-db")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames, for a quick smoke test")
    args = parser.parse_args()

    store = EventStore(":memory:")
    replay_video(args.video_path, args.calibration_path, store, session_id="replay", max_frames=args.max_frames)
    replayed_plays = [
        e for e in store.events_for_session("replay") if e["kind"] == "TilePlayed"
    ]
    replayed_irregularities = store.irregularities_for_session("replay")

    print(f"replay: {len(replayed_plays)} play(s) detected, {len(replayed_irregularities)} irregularit(y/ies)")
    for row in replayed_irregularities:
        print(f"  irregularity: {row['kind']} player={row['player']} reason={row['reason']}")

    if args.compare_db:
        if not args.session_id:
            raise SystemExit("--compare-db requires --session-id")
        original_store = EventStore(args.compare_db)
        try:
            original_plays = [
                e for e in original_store.events_for_session(args.session_id) if e["kind"] == "TilePlayed"
            ]
        finally:
            original_store.close()
        diff = diff_play_sequences(
            [_payload(e) for e in original_plays],
            [_payload(e) for e in replayed_plays],
        )
        print(
            f"compared against {args.compare_db}#{args.session_id}: "
            f"{diff['matched']} matched, {len(diff['only_in_original'])} only in original, "
            f"{len(diff['only_in_replay'])} only in replay"
        )
        for hand_index, player, high, low in diff["only_in_original"]:
            print(f"  only in original: hand={hand_index} player={player} tile={high}-{low}")
        for hand_index, player, high, low in diff["only_in_replay"]:
            print(f"  only in replay: hand={hand_index} player={player} tile={high}-{low}")


if __name__ == "__main__":
    main()
