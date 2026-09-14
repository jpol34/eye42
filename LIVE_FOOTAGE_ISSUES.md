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

- Ranked multi-candidate tile classification + engine-side conflict
  reconciliation (`TileIdentityClassifier.classify`).
- Trump inference via observed void contradictions beyond the first lead.
- Force-closed trick recovery via the trick winner's pile.
- Automated retroactive reinterpretation of a late-discovered revoke.
- Confidence-tuned, multi-tier hand-outcome classification.
- Real table/camera measurements for `simgen`'s `TABLE_SIZE_M`, zone
  coordinates, and `default_camera()`'s framing distance.
- Phase 2-4 numeric acceptance targets (per-tile accuracy, exact final-score
  match), validated against this real recorded data instead of synthetic.

## Iterative loop

1. Run `tools/live_view.py` (with `calibration.json` present) during real
   play; it records video + audio and logs every ingested event/irregularity
   to the session's SQLite DB.
2. Stop it cleanly (`quit` in its REPL — never kill the process; a killed
   `cv2.VideoWriter` never writes its moov atom and the file won't play).
3. Query `events`/`irregularities` for that session, cross-reference
   suspicious entries (duplicates, unexpected irregularities, zero/low
   confidence) against the recorded video/audio at that timestamp.
4. Root-cause in the relevant perception/engine module, fix, and where
   practical add a regression test using this real footage's observed shape
   as the fixture (not just synthetic data).
5. Re-record and repeat.
