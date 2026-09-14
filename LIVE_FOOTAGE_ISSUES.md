# Live A/V findings & issues

Tracks bugs and gaps surfaced by reviewing real camera+mic recordings against
the SQLite event log they produce, now that `tools/live_view.py` can record a
full session to disk (video + audio) for later review. This is the working
list for the iterative loop: record → review `events`/`irregularities` in the
session DB against the video/audio at that timestamp → root-cause → fix →
re-record. Update this file as items are resolved or new ones turn up; don't
leave a resolved-items trail here (see ROADMAP.md's own maintenance rule).

## Confirmed bugs

### 1. `live_view.py`'s REPL requires a live console — exits silently under a closed/redirected stdin

`run_repl`'s `for line in sys.stdin:` returns immediately (no error, no
warning) when stdin is closed or non-interactive (e.g. launched from a
background job runner, a CI step, or any headless launcher), which then runs
`cleanup()` and exits with status 0 — indistinguishable in the log from a
deliberate `quit`. Anyone scripting a session launch needs a real attached
console (or a REPL redesign that doesn't block the whole program on stdin)
or the session silently never actually starts recording for more than a
second. Worth a fix (e.g. don't tie process lifetime to stdin at all — drive
shutdown from the Flask app or a signal only) before this gets automated
further.

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
