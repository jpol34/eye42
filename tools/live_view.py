#!/usr/bin/env python3
"""Host-side harness for a real test session: drives one GameState from a
camera-fed perception pipeline (once calibrated) and/or a stdin REPL, and
serves a diagnostic page (camera snapshot + live engine state + log tail) for
Claude to look at via browser automation while Jordan plays.

Perception is active once ``tools/calibrate.py`` has written a
calibration.json: every captured frame is rectified, localized, and
identified, and a confirmed play is auto-ingested the same way a manual REPL
command is. Without one, or for anything perception doesn't observe (bids,
trump -- speech isn't built), the REPL remains the only input path, and stays
available as a manual override afterward for correcting a misread. This is
not the planned Phase 4 dashboard -- it's a test-session observability tool,
run directly on the Windows host (never in Docker: USB webcam passthrough into
a WSL2 container is not practical).

REPL commands (one event per line):
    deal <dealer> <c0> <c1> <c2> <c3>   TilesDealt(dealer, counts={0..3: c_i})
    bid <player> <amount> [marks]       BidMade
    pass <player>                       Passed
    trump <caller> <trump>              TrumpCalled
    cue <speaker> <low|high>            TrumpCueHeard
    play <player> <high>-<low>          TilePlayed, e.g. "play 2 5-3"
    irregular <tricks_played_so_far>    IrregularEndSignal
    end_hand [tricks_played_so_far]     classify_irregular_end (if given) +
                                         game.record_hand + game.new_hand
    dealer <seat>                       game.observe_next_dealer(seat)
    quit                                stop the harness

Usage:
    python tools/live_view.py [--port 8420] [--camera 0] [--db eye42_live.db]
                               [--calibration calibration.json]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import sounddevice as sd
from flask import Flask, Response, jsonify, request, send_from_directory

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
from eye42.engine.scoring import TRICKS_PER_HAND  # noqa: E402
from eye42.engine.tiles import Tile  # noqa: E402
from eye42.perception.hand_detect import HandDetector, HandGate  # noqa: E402
from eye42.perception.tile_detect import (  # noqa: E402
    EventSegmenter,
    HandBoundaryDetector,
    OpenCVTileClassifier,
    TableRectifier,
    TileLocalizer,
    observe_frame,
)
from eye42.telemetry import EventStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
SEAT_CORNER_ORDER = ("N", "E", "S", "W")  # matches tools/calibrate.py's TL/TR/BR/BL corner labeling


class Camera:
    """Captures frames on its own tight, unblocked loop. Consumers pull the
    latest frame via ``latest_frame()`` at whatever pace suits them, rather
    than running inside this loop -- a slow consumer (perception's rectify/
    localize/classify pass over a 1920x1080 frame takes real time) must
    never throttle the raw capture rate, since the video recorder and the
    live snapshot view both depend on frames actually arriving at close to
    the camera's real rate. Confirmed necessary, not a preemptive
    optimization: an earlier synchronous-subscriber design measured actual
    throughput collapsing to ~6fps under perception load despite the camera
    itself sustaining 30fps, which silently desynced recorded video's
    declared frame rate from wall-clock time."""

    def __init__(self, index: int) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # OpenCV's own default (640x480 on this hardware) is too coarse for
        # reliable pip reads; 1920x1080 measured at ~30fps on this camera --
        # no frame-rate cost to asking for it outright, unlike some
        # intermediate resolutions this camera exposes.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def frame_size(self) -> tuple:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
            time.sleep(1 / 30)

    def latest_frame(self):
        """Returns (frame, frame_id); frame_id lets a caller notice whether
        it's already seen this exact frame."""
        with self._lock:
            return self._frame, self._frame_id

    def snapshot_jpeg(self) -> Optional[bytes]:
        frame, _ = self.latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None

    def save_snapshot(self, path: Path) -> Optional[str]:
        data = self.snapshot_jpeg()
        if data is None:
            return None
        path.write_bytes(data)
        return str(path)

    def start_consumer(self, on_frame: Callable[[np.ndarray, int], None], interval: float) -> None:
        """Runs ``on_frame(latest_frame, frame_id)`` in its own daemon thread
        on a fixed real-time cadence, independent of both the capture loop
        and every other consumer -- this is what keeps a slow consumer (or
        one that must stay wall-clock-accurate, like video recording) from
        affecting any other. ``frame_id`` lets a consumer tell "the camera
        genuinely stalled" apart from "I just haven't been scheduled in a
        while" (see ``VideoRecorder``)."""

        def _loop() -> None:
            while not self._stop:
                frame, frame_id = self.latest_frame()
                if frame is not None:
                    on_frame(frame, frame_id)
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop = True
        self._cap.release()


class VideoRecorder:
    """Writes captured frames to a video file for later playback (synced
    against the separately wall-clock-timed audio recording) -- distinct
    from the SQLite log's per-play snapshots, which capture only the
    instant a play settles, not the continuous footage in between.

    Wall-clock-paced, not call-paced: the consumer thread driving this wakes
    up on a best-effort schedule that real thread/GIL contention (perception
    running concurrently) can delay unpredictably -- a naive one-write-per-
    call approach falls to half the target frame rate under load. This is
    the standard, if unglamorous, workaround for a real, documented
    `cv2.VideoWriter` limitation: it has no timestamp/PTS concept at all and
    just assumes every ``write()`` call is evenly spaced at the declared
    fps, so an irregular producer desyncs playback speed unless something
    else guarantees that assumption holds. Every call compares real elapsed
    time against the declared fps and writes however many frames
    (duplicating the current one if behind) are needed to catch up, so the
    file's own duration (frame_count / fps) always matches real elapsed
    time regardless of how irregularly this gets called.

    Padding is driven off ``frame_id`` (see ``Camera.latest_frame``), not
    bare elapsed time, so a genuinely stalled camera -- not just a slow
    polling thread, which padding is supposed to paper over -- is at least
    logged once rather than silently padded forever with a stale frame.
    """

    _STALL_WARNING_SECONDS = 1.0

    def __init__(self, path: Path, frame_size: tuple, fps: float = 30.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        self._fps = fps
        self._start: Optional[float] = None
        self._frames_written = 0
        self._last_frame_id: Optional[int] = None
        self._last_frame_id_since: Optional[float] = None
        self._stall_warned = False
        self.path = path

    def __call__(self, frame: np.ndarray, frame_id: int) -> None:
        now = time.monotonic()
        if self._start is None:
            self._start = now

        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._last_frame_id_since = now
            self._stall_warned = False
        elif not self._stall_warned and now - self._last_frame_id_since > self._STALL_WARNING_SECONDS:
            print(f"warning: camera hasn't produced a new frame in over {self._STALL_WARNING_SECONDS:.0f}s "
                  f"-- video is padding with a stale frame")
            self._stall_warned = True

        expected = int((now - self._start) * self._fps) + 1
        while self._frames_written < expected:
            self._writer.write(frame)
            self._frames_written += 1

    def close(self) -> None:
        self._writer.release()


def _find_input_device(name_substring: str) -> Optional[int]:
    """First input-capable device whose name contains ``name_substring``
    (case-insensitive), or None to fall back to the system default mic --
    used to prefer the webcam's own mic (physically at the table) over a
    laptop mic that might be far from the players."""
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0 and name_substring.lower() in info["name"].lower():
            return index
    return None


class AudioRecorder:
    """Records table audio to a WAV file alongside the video, so a session
    can be reviewed with sound afterward -- there's no live speech parsing
    (Phase 3 isn't built), so bids/trump calls aren't tracked automatically,
    but a synced recording lets them be filled in by ear on review. Runs on
    its own callback-driven stream, independent of the camera's frame loop.

    Wall-clock-paced, not callback-paced, for the same reason VideoRecorder
    is: `sounddevice` reports a dropped/overflowed callback via its `status`
    argument rather than raising, so a naive writer silently produces a WAV
    shorter than the real session under the same thread contention already
    measured to affect the video recorder -- confirmed against real
    recordings, not assumed: two real sessions' WAV files ran 54s and 102s
    short of their matching video's duration, a 10-17% deficit, with no
    error or warning at the time. Every callback pads with silence for any
    gap between wall-clock elapsed time and samples written so far, so total
    file duration keeps matching real elapsed time regardless of dropped
    callbacks, and a dropped/overflowed callback is at least logged (once
    close() runs) rather than silently invisible.
    """

    def __init__(self, path: Path, device: Optional[int] = None, samplerate: int = 44100, channels: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._samplerate = samplerate
        self._channels = channels
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(channels)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(samplerate)
        self._start: Optional[float] = None
        self._samples_written = 0
        self._flagged_callbacks = 0
        self._stream = sd.InputStream(
            device=device, channels=channels, samplerate=samplerate, dtype="int16", callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        now = time.monotonic()
        if self._start is None:
            self._start = now
        if status:
            self._flagged_callbacks += 1

        expected = int((now - self._start) * self._samplerate)
        gap = expected - self._samples_written - frames
        if gap > 0:
            self._wav.writeframes(np.zeros((gap, self._channels), dtype=np.int16).tobytes())
            self._samples_written += gap

        self._wav.writeframes(indata.tobytes())
        self._samples_written += frames

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._wav.close()
        if self._flagged_callbacks:
            print(f"warning: audio recording had {self._flagged_callbacks} dropped/overflowed callback(s) "
                  f"(padded with silence to keep {self.path} wall-clock-accurate)")


def load_perception(calibration_path: Path):
    """Builds the rectifier/localizer/classifier/segmenter from a
    tools/calibrate.py-produced calibration.json, or returns None if no
    calibration exists yet -- a session without one just runs on manual REPL
    entry, same as before perception existed."""
    if not calibration_path.exists():
        return None
    calibration = json.loads(calibration_path.read_text())
    corners_by_seat = calibration["corners"]
    width, height = calibration["rectified_size"]
    corners = [corners_by_seat[seat] for seat in SEAT_CORNER_ORDER]
    seats = {"N": (0.0, 0.0), "E": (width - 1.0, 0.0), "S": (width - 1.0, height - 1.0), "W": (0.0, height - 1.0)}
    rectifier = TableRectifier(corners, (width, height))
    return (
        rectifier,
        TileLocalizer(),
        OpenCVTileClassifier(),
        EventSegmenter(seats=seats),
        HandBoundaryDetector(),
        HandGate(HandDetector(), rectifier),
    )


class Session:
    """Owns the live GameState/HandState pair and every event that reaches
    it -- the single source of truth the Flask routes, the REPL, and the
    camera-driven perception pipeline (when calibrated) all read from/write
    through."""

    def __init__(
        self,
        store: EventStore,
        session_id: str,
        camera: Camera,
        frames_dir: Path,
        perception=None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.camera = camera
        self.frames_dir = frames_dir
        self.game = GameState(store=store, session_id=session_id)
        self.hand = self.game.new_hand()
        self.lock = threading.Lock()
        self._perception = perception
        self._frame_index = 0

    def _snapshot_path(self) -> Optional[str]:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        return self.camera.save_snapshot(self.frames_dir / f"{time.time():.3f}.jpg")

    def ingest(self, event: object) -> None:
        with self.lock:
            self.hand.ingest(event, frame_path=self._snapshot_path())

    def on_frame(self, frame: np.ndarray, frame_id: int) -> None:
        """Camera consumer callback: runs perception on every captured
        frame and auto-ingests any play it confirms. A no-op until
        calibration.json exists -- the REPL remains the only input path
        until then, and stays available as a manual override afterward (for
        correcting a misread, or for bids/trump, which perception doesn't
        observe).

        A detected reshuffle ends the current hand before anything else runs
        this frame, since a hand boundary invalidates whatever the segmenter
        was tracking for the hand that just ended.

        Every confirmed play is always ingested, even before a contract
        exists: HandState.play_tile already buffers a pre-contract play in
        _orphan_plays (replayed once bidding resolves) and logs an
        IrregularityKind.PLAY_BEFORE_CONTRACT irregularity, so there is no
        need to withhold plays here."""
        if self._perception is None:
            return
        rectifier, localizer, classifier, segmenter, hand_boundary, hand_gate = self._perception
        self._frame_index += 1
        rectified, observations = observe_frame(rectifier, localizer, classifier, frame, self._frame_index)
        observations = hand_gate.filter(frame, observations, segmenter.last_motion_mask)
        played = segmenter.feed(rectified, observations)

        if segmenter.last_motion_mask is not None and hand_boundary.is_new_hand_starting(segmenter.last_motion_mask):
            tricks_so_far = len(self.hand.tricks)
            self.end_hand(None if tricks_so_far >= TRICKS_PER_HAND else tricks_so_far)
            segmenter.reset()
            return

        for play in played:
            self.ingest(play)

    def end_hand(self, tricks_played_so_far: Optional[int]) -> None:
        with self.lock:
            if tricks_played_so_far is not None:
                self.hand.classify_irregular_end(IrregularEndSignal(tricks_played_so_far=tricks_played_so_far))
            self.store.log_event(
                {"tricks_played_so_far": tricks_played_so_far},
                session_id=self.session_id,
                hand_index=self.hand.hand_index,
                kind="HandEnded",
            )
            self.game.record_hand(self.hand)
            self.hand = self.game.new_hand()

    def observe_dealer(self, seat: int) -> None:
        with self.lock:
            result = self.game.observe_next_dealer(seat)
            self.store.log_event(
                {"observed_dealer": seat, "result": result},
                session_id=self.session_id,
                hand_index=self.hand.hand_index,
                kind="DealerObserved",
            )

    def live_state(self) -> dict:
        with self.lock:
            return {**self.game.live_state(), "hand": self.hand.live_state()}


def _parse_tile(spec: str) -> Tile:
    high, low = spec.split("-")
    return Tile.of(int(high), int(low))


def run_repl(session: Session, shutdown: threading.Event) -> None:
    """Runs on its own thread, never the main one: a closed/redirected
    stdin (any non-interactive launcher -- a background job runner, a CI
    step) hits immediate EOF on the very first read, which would otherwise
    make `for line in sys.stdin` return right away -- indistinguishable
    from a deliberate `quit` if that tore the whole session down. Recording
    and perception don't need a REPL to keep running, so an exhausted
    stdin here just ends this thread quietly; the process as a whole keeps
    running (its daemon threads: camera, recorders, perception, Flask)
    until ``shutdown`` is set, either by a real `quit` below or by a
    signal handler."""
    print("eye42 live-session REPL. Type 'help' for commands, 'quit' to stop.")
    try:
        for line in sys.stdin:
            parts = line.split()
            if not parts:
                continue
            cmd, *args = parts
            try:
                if cmd == "quit":
                    shutdown.set()
                    break
                elif cmd == "help":
                    print(__doc__)
                elif cmd == "deal":
                    dealer, c0, c1, c2, c3 = args
                    counts = {0: int(c0), 1: int(c1), 2: int(c2), 3: int(c3)}
                    session.ingest(TilesDealt(dealer=int(dealer), counts=counts))
                elif cmd == "bid":
                    player, amount = args[0], args[1]
                    marks = int(args[2]) if len(args) > 2 else 0
                    session.ingest(BidMade(player=int(player), amount=int(amount), marks=marks))
                elif cmd == "pass":
                    session.ingest(Passed(player=int(args[0])))
                elif cmd == "trump":
                    session.ingest(TrumpCalled(caller=int(args[0]), trump=int(args[1])))
                elif cmd == "cue":
                    session.ingest(TrumpCueHeard(speaker=int(args[0]), cue=args[1]))
                elif cmd == "play":
                    session.ingest(TilePlayed(player=int(args[0]), tile=_parse_tile(args[1])))
                elif cmd == "irregular":
                    session.ingest(IrregularEndSignal(tricks_played_so_far=int(args[0])))
                elif cmd == "end_hand":
                    tricks = int(args[0]) if args else None
                    session.end_hand(tricks)
                elif cmd == "dealer":
                    session.observe_dealer(int(args[0]))
                else:
                    print(f"unrecognized command: {cmd!r} (try 'help')")
            except (ValueError, IndexError) as exc:
                print(f"bad command {line.strip()!r}: {exc}")
    except Exception:
        # This now runs on its own thread (see the module-level docstring
        # above), so an unexpected bug here would otherwise just kill this
        # thread silently -- main() would stay blocked on `shutdown.wait()`
        # forever with no REPL and no cleanup(), never finalizing the
        # recording. Set shutdown before re-raising so the process still
        # winds down the same way an uncaught exception here would have on
        # the main thread.
        traceback.print_exc()
        shutdown.set()
        raise


def build_app(session: Session) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "live_view.html")

    @app.get("/snapshot.jpg")
    def snapshot():
        data = session.camera.snapshot_jpeg()
        if data is None:
            return Response(status=503)
        return Response(data, mimetype="image/jpeg")

    @app.get("/state.json")
    def state():
        return jsonify(session.live_state())

    @app.get("/log")
    def log():
        since = request.args.get("since", type=float)
        return jsonify(session.store.recent(session.session_id, since_ts=since, limit=100))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--db", default="eye42_live.db")
    parser.add_argument("--session-id", default=None, help="defaults to the current timestamp")
    parser.add_argument("--calibration", default="calibration.json", help="from tools/calibrate.py")
    parser.add_argument("--no-record", action="store_true", help="skip saving continuous video for later playback")
    args = parser.parse_args()

    session_id = args.session_id or f"session_{int(time.time())}"
    db_path = Path(args.db)
    store = EventStore(db_path)
    camera = Camera(args.camera)
    frames_dir = db_path.parent / f"{db_path.stem}_frames" / session_id
    perception = load_perception(Path(args.calibration))
    session = Session(store=store, session_id=session_id, camera=camera, frames_dir=frames_dir, perception=perception)
    # Perception runs as fast as it can on its own thread (its own
    # processing time dominates any fixed interval here) -- decoupled from
    # capture so it can never throttle the video recorder below.
    camera.start_consumer(session.on_frame, interval=0.01)
    print(
        f"perception: {'active (' + args.calibration + ')' if perception else 'inactive -- no calibration.json, REPL-only'}"
    )

    recorder = None
    audio_recorder = None
    if not args.no_record:
        video_path = db_path.parent / f"{db_path.stem}_videos" / f"{session_id}.mp4"
        recorder = VideoRecorder(video_path, camera.frame_size)
        # A fixed 1/30s cadence, independent of perception, is what keeps
        # the recorded file's declared frame rate matching wall-clock time
        # (and therefore in sync with the separately wall-clock-timed audio
        # recording) even while perception runs far slower per frame.
        camera.start_consumer(recorder, interval=1 / 60)
        print(f"recording video to {video_path}")

        audio_path = db_path.parent / f"{db_path.stem}_audio" / f"{session_id}.wav"
        try:
            audio_recorder = AudioRecorder(audio_path, device=_find_input_device("j5"))
            print(f"recording audio to {audio_path}")
        except Exception as exc:
            # Audio is a nice-to-have for later review, not load-bearing for
            # the session -- a missing/busy mic shouldn't stop video/
            # perception from running, same spirit as Camera's own
            # never-block-on-missing-hardware handling.
            print(f"audio recording unavailable ({exc}); continuing without it")

    app = build_app(session)
    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()
    print(f"session {session_id!r}: view at http://127.0.0.1:{args.port}/")

    _cleaned_up = False

    def cleanup() -> None:
        # Idempotent and shared between the normal REPL-exit path and a
        # signal handler: cv2.VideoWriter only finalizes a playable mp4
        # (writes its moov atom) on release(), so a process killed without
        # running this leaves the video file corrupt and unreadable --
        # confirmed, not assumed: an ungraceful stop during testing produced
        # exactly that ("moov atom not found").
        nonlocal _cleaned_up
        if _cleaned_up:
            return
        _cleaned_up = True
        camera.stop()
        if recorder is not None:
            recorder.close()
        if audio_recorder is not None:
            audio_recorder.close()
        store.close()

    def handle_signal(signum, frame) -> None:
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    shutdown = threading.Event()
    threading.Thread(target=run_repl, args=(session, shutdown), daemon=True).start()
    try:
        shutdown.wait()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
