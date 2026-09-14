# Live A/V findings & issues

Tracks bugs and gaps surfaced by reviewing real camera+mic recordings against
the SQLite event log they produce, now that `tools/live_view.py` can record a
full session to disk (video + audio) for later review. This is the working
list for the iterative loop: record → review `events`/`irregularities` in the
session DB against the video/audio at that timestamp → root-cause → fix →
re-record. Update this file as items are resolved or new ones turn up; don't
leave a resolved-items trail here (see ROADMAP.md's own maintenance rule).

## Confirmed bugs

### 1. `VideoRecorder`'s output has no crash/power-loss resilience

`cv2.VideoWriter` only writes its finalizing moov atom on `release()`
(`VideoRecorder.close()`), so any non-graceful end of the process — a
crash, a killed process, a power loss — leaves an mp4 with no moov atom,
unrecoverable by normal playback (`moov atom not found`). Confirmed for
real: a mid-session laptop power loss lost a 2.3GB/~36-minute video
recording outright, while the matching WAV audio (no comparable finalize
step) survived intact. Worth a more resilient container/write strategy
(e.g. periodic remuxing, or a format that doesn't need a trailing index)
before relying on this for a session that can't be easily redone.

### 2. Touching/adjacent tiles during ordinary play still produce spurious plays, distinct from the sweep/cluster cases already fixed

`session_6` (real recorded play) logged three `TilePlayed(player=0, tile=1-1, player_confidence=0.0)` events at 831.9s, 881.5s, and 881.5s again (57ms apart) — all before the *real* 1-1 tile was legitimately played, correctly, at 905.7s (`player=3, confidence=1.0, player_confidence≈1.0`). A domino set has exactly one 1-1 tile, so the three earlier firings are misreads, not real plays.

This happened during ordinary gameplay with small, already-won trick piles (3-4 touching tiles each) sitting in front of players rather than being swept away — not a sweep in progress, and not a large enough merged blob to trigger `unseparated_tile_cluster_regions` (measured largest merged contour in this frame: 37,043px², under the 60,000px² cluster threshold, and with a plausible-tile-shaped aspect ratio of 1.70 — well inside `TileLocalizer`'s own 1.4-3.2 acceptance band, so a naive aspect-ratio-based cluster check wouldn't have caught it either, confirmed by direct measurement before trying that fix).

This is the same class of problem ROADMAP.md's "TileLocalizer can't yet split touching clusters" already tracks (its own real-footage test fixture, `fixtures/real_footage_touching_cluster.jpg`, documents the identical failure mode) — a real fix needs actual tile/cluster separation (e.g. detecting the divider line between two touching tiles), not another motion- or area-based gate like the sweep/cluster fixes above. Flagging here rather than attempting an under-evidenced threshold change.

An offline replay of `session_3` (`tools/replay_video.py`) independently reproduces the same signature — repeated spurious `1-1` plays logged as `PLAY_BEFORE_CONTRACT` irregularities before a real play — corroborating this as a reproducible detector limitation rather than a one-off misread.

This is also the root cause blocking several other real-footage-dependent items: `session_5`'s hand-boundary detection never advanced past `hand_index=0` across its full ~16-minute recording (263 logged `TilePlayed` events, only 21 distinct tiles), so no clean multi-trick real hand currently exists to validate anything that needs one against (per-tile accuracy scoring, trump-inference-beyond-first-lead design validation, etc.).

A classical-CV split (distance-transform + watershed, seeded by tile-width-spaced local maxima) was tried against the real `fixtures/real_footage_touching_cluster.jpg` fixture and did not cleanly separate a known 2-tile merged blob on a first attempt — not proven to be a viable fix. The actual integration path already exists in-repo: `eye42.perception.tile_segment.TileSegmenter` wraps a YOLOv8-seg ONNX model and is structurally compatible with `TileLocalizer.find_tiles`'s output (`TileRegion` list), with its pre/postprocessing already unit-tested; `test_segmenter_splits_the_real_touching_cluster_fixture` is currently skipped for lack of a trained model at `models/tile_segmenter.onnx`. What's missing is the trained model artifact (via `tools/train_tile_segmenter.py`, which needs the `ultralytics` training extra) and wiring `TileSegmenter` into `tile_detect.py`/`tools/live_view.py` for oversized/ambiguous cluster crops. Jordan is separately setting up AI model training infrastructure for use across projects, including this one — that's the natural path to producing the trained model this already-built integration is waiting on.

## Things to keep watching

- A laptop sits on the table inside the calibrated capture corners in this
  setup, so it's part of `EventSegmenter`'s rectified table image.
  `_sweep_in_progress` treats >15% motion across that whole image as a
  trick-sweep; scrolling terminal text or a hand near the laptop could hold
  the segmenter in a false "sweeping" state for real stretches of play. A
  play made during a false sweep sits pending and delayed until the sweep
  clears rather than confirming promptly — watch for plays that take
  unusually long to show up in `events` while the laptop is in frame.

## Now-unblocked ROADMAP items

Real recorded gameplay + audio now exists (video: `live_recordings/`, audio:
`game1_boundary.wav`/`game1_final.wav` from a prior session). Everything
ROADMAP.md's "Blocked on real footage" section listed as waiting on exactly
this is a candidate to pick up now:

- Engine-side conflict reconciliation using `TileIdentityClassifier`'s
  ranked candidates (the classifier half is done; this needs its own
  design pass — see ROADMAP.md).
- Trump inference via observed void contradictions beyond the first lead
  (design questions are answerable now, but validating against a real hand
  needs issue #2 above resolved first).
- Force-closed trick recovery via the trick winner's pile.
- Automated retroactive reinterpretation of a late-discovered revoke.
- Confidence-tuned, multi-tier hand-outcome classification (needs a real
  misdeal/throw-in example to calibrate against; none of the recorded
  sessions has one yet).
- Phase 2-4 numeric acceptance targets (per-tile accuracy, exact final-score
  match): `tools/replay_video.py` can now run the detector against saved
  footage and diff its output against a session's original log, but an
  actual accuracy number against ROADMAP's target still needs a hand-authored
  ground-truth annotation of a real session, which doesn't exist yet.

Real table/camera measurements for `simgen`'s `TABLE_SIZE_M`, zone
coordinates, and `default_camera()`'s framing distance remain blocked, not
unblocked: `calibration.json` only holds a pixel-space homography with no
real-world unit mapping, and no camera calibration (e.g. a checkerboard
capture) has been taken. This needs Jordan to physically measure the table
or capture a calibration checkerboard shot before it's buildable.

## Iterative loop

1. Run `tools/live_view.py` (with `calibration.json` present) during real
   play; it records video + audio and logs every ingested event/irregularity
   to the session's SQLite DB.
2. Stop it cleanly (`quit` in its REPL — never kill the process; a killed
   `cv2.VideoWriter` never writes its moov atom and the file won't play).
3. Query `events`/`irregularities` for that session, cross-reference
   suspicious entries (duplicates, unexpected irregularities, zero/low
   confidence) against the recorded video/audio at that timestamp. Or run
   `tools/replay_video.py <video> calibration.json --compare-db <db>
   --session-id <id>` to re-run the detector against the saved video
   offline and diff its output against the session's original log — useful
   for reviewing detector behavior without a live camera, though recorded
   video is a wall-clock-resampled reconstruction of what perception saw
   live, so a diff is a starting point for manual review, not a pass/fail
   regression signal.
4. Root-cause in the relevant perception/engine module, fix, and where
   practical add a regression test using this real footage's observed shape
   as the fixture (not just synthetic data).
5. Re-record and repeat.
