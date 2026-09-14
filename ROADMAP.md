# Roadmap

This file is the single place for eye42's not-yet-built work, known gaps, and
ideas under consideration. It is not a specification and nothing in it is
committed — every item here is a suggestion to weigh when its time comes, not
a promise about what will be built or how. Other work (a design decision, a
bug fix, real footage exposing a different pain point) can change what's
actually worth doing or even possible, and an item can turn out to be wrong,
unnecessary, or already superseded by the time anyone picks it up.

**Maintenance rule for whoever (human or model) edits this file:** keep it
current, not historical. When an item ships, gets rejected, or goes stale,
remove it — don't leave a trail of "done"/"superseded"/"no longer relevant"
entries here. This file should always read as "here's what's actually still
open," never as a log of what used to be open. (The project's build history —
what was done, when, and why — lives in the pr history for the gh repo)

## Blocked on real footage

These items need actual recorded gameplay (or a specific physical capture, like
a lens-calibration checkerboard shot) before they can be responsibly designed
or built — attempting them without that risks exactly the unvalidated-guesswork
failure mode several entries below already call out. Once footage/capture
exists, these are the first candidates to revisit (each points at its full
entry elsewhere in this file, not restated here):

- Trump inference via observed void contradictions beyond the first lead
  ("Trump determination beyond the first lead", item 3).
- Force-closed trick recovery via the trick winner's pile ("Recovering a
  force-closed trick's missing tiles").
- Automated retroactive revoke reinterpretation ("Automated retroactive
  reinterpretation of a late-discovered revoke").
- Confidence-tuned, multi-tier hand-outcome classification (needs real
  throw-in examples to calibrate against).
- `simgen`'s measured roughness/gloss and calibrated lens distortion (under
  "Standalone 3D simulation", "Deferred, explicitly out of scope for Phase 0").
- Real table/camera measurements for `tile_geometry.py`'s `TABLE_SIZE_M`/zone
  coordinates and `default_camera()`'s framing distance (under "Known rough
  edges to tune, not fixed yet").
- Phase 2-4's own numeric acceptance targets (per-tile accuracy, exact
  final-score match) — validated against real recorded hands, not synthetic
  data (see "Implementation notes for not-yet-built phases").
- Phase 5 robustness hardening — only in response to a specific, reproducible
  failure mode real validation footage actually shows, not speculatively.

## Standalone 3D simulation (`eye42.simgen`)

A Phase 0 skeleton (physics, a scripted-hand director reusing tested engine
logic, occlusion-correct rendering including a Blender/Cycles photoreal path,
YOLO-seg + JSON ground-truth export) and a Phase 1 hand/forearm occluder rig
are built. See RESEARCH.md's "Standalone 3D simulation" section for the
architecture and design rationale, and the modules themselves
(`simgen/render.py`, `simgen/hand.py`, `simgen/ground_truth.py`) for how each
piece works today.

**Deferred, explicitly out of scope for Phase 1** (see RESEARCH.md's
"Hand/manipulation fidelity" section for the full recommendation this is a
first slice of): the full ~8-clip keyframed gesture library (Phase 1 ships
one parameterized path); per-finger procedural noise beyond coarse per-frame
curl/spread jitter; contact-driven physics — tiles are still moved by
`physics.slide_toward`, the hand is posed near a tile's actual pre/post-
slide position, not causing its motion (`hand_parts()`'s output is the same
pose shape a future MuJoCo mocap-body attachment would consume, so this
isn't wasted work); mid-motion/dense frame sampling (`FrameSnapshot`s stay
settled-moments-only); skinning/deformable mesh; two hands/handedness/
sleeve variety; shuffle-phase hand modeling (RESEARCH.md explicitly licenses
low fidelity there).

**Known rough edges (hand rig):** occlusion uses one constant depth per box
part (matching the tile-tile approach), not a true per-pixel depth buffer —
accurate for a palm hovering over a tile (the common case given this
camera's angle) but can misorder a part extending horizontally toward the
camera across a nearer tile; bounded to roughly a tile's own size by using 2
segments per finger/forearm, named here as the approximation to revisit (a
per-pixel depth buffer) if it ever proves to matter. `RETR_EXTERNAL`-based
mask extraction won't split out a true enclosed hole (an occluder entirely
inside a tile's silhouette, touching no edge) — a separate, narrower,
unaddressed case. `_SKIN_COLOR` is an unmeasured placeholder tuned only for
contrast against this scene's table/lighting, not real skin tones.

**Deferred, explicitly out of scope for Phase 0** (see RESEARCH.md and
"Blocked on real footage" above): measured roughness/gloss calibrated
against real footage (pips currently share the body's flat `Roughness=0.15`,
no separate matte/gloss distinction); calibrated lens distortion (a real
checkerboard calibration capture doesn't exist yet; post-render sensor
noise is a plausible, uncalibrated placeholder shipped ahead of it); GPU-
rented bulk generation; retraining/evaluating the YOLOv8-seg model against
this new data source.

**Known rough edges to tune, not fixed yet:** `tile_geometry.py`'s
`TABLE_SIZE_M` and `trajectory.py`'s seat rack/won-pile zone coordinates are
placeholder guesses in absolute meters (no real table measurement or camera
calibration exists yet — see RESEARCH.md's "calibration.json is a
homography, not a camera calibration"), though a regression test pins a
minimum margin between the table edge and every zone's worst-case reach so
this can't silently regress. `default_camera()`'s oblique *angle* is
verified against `calibration.json`'s real corner points, but its
distance/`focal_px` (how tightly it frames the table) is still untuned
against real footage.

## Not yet built

- **Phase 2 — perception pipeline, mostly built as separate tested modules,
  not yet assembled into one live camera-in/event-out entry point.** Built:
  homography rectification (`TableRectifier`), tile localization, 28-class
  identity classification (`OpenCVTileClassifier` — color-threshold pip
  counting, not the template-match/CNN originally envisioned; avoids
  Hough-circle pip counting's spinner-pin/blank failure modes), settle-
  time-debounced event segmentation with explicit trick-sweep handling and
  motion-trail player attribution (`EventSegmenter` — player attribution
  matters because turn order/fixed seat quadrants alone can't resolve who's
  next while the trick winner, and therefore trump, is still ambiguous), and
  a shuffle/reshuffle hand-boundary detector independent of "7 tricks
  completed" (`HandBoundaryDetector`). A second, tested touching-tile-
  cluster segmenter (`tile_segment.py`, a trained YOLOv8-seg ONNX model)
  exists for fanned/boneyard tiles merging into one contour, but isn't
  wired into `tile_detect.py` or `tools/live_view.py`. Not built:
  concession/redeal entered on a perception signal (mass tile motion, tiles
  flipping, trick-pile mixing) — `IrregularEndSignal` exists on the engine
  side but nothing in perception produces it yet; concessions need this
  because they happen earlier than the engine's own arithmetic certainty.
  What's left, concretely: wire `tile_segment.py` in for the touching-
  cluster case, build the concession/redeal perception signal, and assemble
  everything into one live pipeline entry point.
- **Phase 3 — speech layer, batch MVP built and validated against real
  recorded gameplay audio, not yet live.** Built: `speech.transcribe`
  (batch, not streaming — local `faster-whisper`, "base" model +
  `vad_filter=True`, no cloud API) and `speech.bid_parser`'s
  `BidCandidateScorer`/`TrumpCueResolver` (closed-vocabulary bid/pass/marks/
  trump-cue matching scored by bid-rotation plausibility, checked via a
  deep-copied trial call into the real `BiddingRound`, rather than accepting
  on a bare vocabulary match — ordinary table talk containing a number must
  not read as a bid, confirmed against real transcripts). Speaker
  attribution is turn-order rotation only (`BiddingRound.current_bidder`),
  never true diarization — a single shared table mic makes real diarization
  both hard and unnecessary here. `tools/transcribe_session.py` runs the
  whole pipeline against a session's WAV and prints a candidate bid/trump-cue
  sequence for a human to confirm against the video — it does not feed
  events into a live `HandState` (see "No shared clock/frame convention",
  below, for why that's not wired up yet). See RESEARCH.md's "Speech:
  real-audio Whisper validation" section for the transcript evidence this
  was built against, including one real false-positive shape found and
  fixed during validation. Not built: real-time/streaming transcription
  (deliberately deferred — batch is the right MVP per that same research:
  real-time re-segmentation makes Whisper's worst failure mode, poor
  accuracy on short isolated utterances, worse, for no benefit this project
  needs); the engine's splash/plunge-from-observed-leader cross-check
  already works automatically once a real `TrumpCalled`/`TilePlayed` event
  reaches `HandState` (see `engine.hand._maybe_infer_splash_or_plunge`), so
  nothing extra is needed on the speech side for that; point-bid ("thirty",
  ...) vocabulary matching is unvalidated against real speech (no example
  turned up in the audio sampled so far) unlike marks bids, which do have
  real transcript evidence.
- **Phase 4 — live dashboard**: switch from record-then-batch to a live
  polling loop; a minimal web dashboard for current bid/trump/trick/score and
  multi-hand marks tracking. The only confirmation channel that doesn't
  interrupt play is post-hoc (in a replay/review UI) — the live view should
  show state as **provisional** wherever it depends on something not yet
  confirmed (trump not yet locked, a hand-end classification pending the next
  hand's dealer), not claim live certainty it can't back up in the moment.
  `tools/live_view.py` already serves a Flask page today, but its own
  docstring disclaims it as a test-session observability tool, not this
  planned product dashboard — worth checking for reusable pieces, not a
  substitute for building this.
- **Phase 4b remainder — probability visualization**: the engine
  (`engine/probability.py`) is built and tested; the display layer itself
  isn't — a two-team horizontal bar (bid-team % vs. defense %) updating per
  trick, a sparkline across the 7 tricks, and expected remaining defense
  count-points as a secondary stat. Slots into the Phase 4 dashboard once
  that exists.
- **Phase 5 (stretch, only if needed) — robustness hardening**: a heavier
  detector already exists and is tested (`tile_segment.py`'s YOLOv8-seg
  model, for the touching-tile-cluster failure mode) but isn't wired into
  the live pipeline (see Phase 2) — wire it in only in response to a
  specific, reproducible failure mode real validation footage actually
  shows, not speculatively; multi-camera angles have no code yet at all.

## Implementation notes for not-yet-built phases

- Phase 2's rectification/localization: OpenCV. Phase 3's STT: local Whisper
  (`faster-whisper`), no cloud API, so recordings never leave the device.
  Phase 4's server: plain HTML/JS or a tiny local Flask/FastAPI process.
- Numeric acceptance targets once these phases exist, not just "looks
  right": per-tile identity accuracy ≥98% over ≥200 observed tiles (before
  any repair/reconciliation), exact final-score match on ≥3 full real hands,
  ≤1 human confirmation prompt per hand on average. Per-phase validation:
  Phase 2 against 2-3 recorded hands (one set, one informal ending) checking
  the accuracy target, no duplicate/missed sweep events, and correct
  concession-vs-redeal routing; Phase 3 against hands with real background
  chatter (and a splash/plunge hand if captured), checking bid/trump/leader
  attribution and that bid-rotation-plausibility scoring suppresses
  table-talk false positives; Phase 4 by playing one full live game and
  comparing the dashboard's final tally to the players' manual one.

## Known gaps / ideas under consideration

- **Trump determination beyond the first lead.** The real priority order, as
  actually played: (1) an explicit verbal call is always primary and
  permanent — it must never be second-guessed by what the caller
  subsequently plays; that's evidence of a misplay to log, never grounds to
  reinterpret trump. (2) Absent a call, the led tile's high end, unless a
  spoken cue names a different end or number outright (e.g. leading 4-5 while
  saying "4s are trump," not just "low"/"high" — the low/high phrasing is
  only a sometimes-used convention, almost always at the hand's very start).
  (3) Trump can also be learned or confirmed mid-hand from an actual observed
  off-suit play. (4) Under splash/plunge, the bidder's partner calls trump —
  already correctly modeled. Only (1) and (4) are built.
  - **(2) is deliberately not generalized to every trick, not just the
    first — this itself IS the folklore-generalization to avoid, not a
    separate bug fix from it.** `hand.py`'s `_pending_cue` today is only
    consumed on the hand's very first lead; that reads like an arbitrary
    restriction worth "fixing," but the restriction is doing real work: the
    cue convention is rare beyond a hand's opening lead, and `TrumpCueHeard`
    carries no timestamp, so generalizing consumption to every trick's lead
    creates a real speech-lags-video misattribution risk at every trick
    boundary instead of just once (a cue meant for trick N's lead could
    arrive after trick N's lead was already processed, get silently held,
    and then get wrongly applied to trick N+1's led tile). Don't build this
    without a timing/ordering guard, and don't build the guard speculatively
    either — both need real transcript-vs-video lag data to design against.
    If/when this is revisited, `_ensure_trick_started`'s `trick.trump = ...`
    resync (currently done ad hoc only in the first-lead branch) needs to be
    carried to every consumption site, and it needs real test coverage
    (`hear_trump_cue`/`_pending_cue` routing has none today) — cue heard on
    trick 3's lead is consumed; a cue heard after trump is already confirmed
    is dropped and doesn't leak forward; a cue meant for trick N doesn't
    bleed into trick N+1.
  - **(3)'s non-oracle mechanism, sketched for whoever designs it once
    footage exists — deliberately not built now.** A live, camera-free
    mechanism is possible without any lookup into a player's remaining
    concealed hand: if a player fails to follow led suit S under a live
    (not yet confirmed) hypothesis H, and that same player later plays a
    tile that computes as suit S under H, that's a hard contradiction —
    they can't be void in S under H and also hold/play S under H, so H must
    be wrong. No folklore weighting needed, just the same publicly-observed
    play stream every other mechanism here already uses.
    `TrumpHypothesisTracker.observe_void` is already a documented no-op stub
    for exactly this; `observe_contradiction`'s implementation only touches
    `self.weights` and needs no per-player hand data, so it's a reusable
    primitive for this signal. What's missing is per-hypothesis void
    tracking (a player can be void in suit S under candidate H1 but not H2,
    since suit membership depends on trump) — real state-machine
    complexity, not a trivial wire-up: it has to interact correctly with
    `observe_contradiction`'s weight-reopening fallback
    (`_last_weights_before_empty`) — when a candidate's weight collapses and
    later gets restored from a snapshot, its void state needs a defined rule
    for whether it restores alongside the weight or resets. Even though
    this doesn't strictly need a camera to test (synthetic event sequences
    exercise it fine), hold off until real footage exists anyway — reasoning
    about a live state-machine change here is genuinely hard to validate
    without real hands to check it against, same as the rest of this entry.
    Needs its own design pass and plan-critic review before building, not a
    bundled addition to anything else.
- **Recovering a force-closed trick's missing tiles.** When a trick
  force-closes short (fewer than 4 plays observed), the hand is marked
  `disputed` and its count is not trusted for the rest of the hand's
  arithmetic — but the trick's own real data is kept, not discarded. Real
  play offers a plausible future recovery signal: the team that wins a trick
  physically drags its 4 tiles into a pile near them, which a camera could in
  principle inspect after the fact. Exception to design around: on a hand
  won with a 2-marks bid, the bidding team may optionally show only their
  last two won tricks stacked, obscuring earlier ones — so even a working
  pile-inspection signal won't always recover everything. No reconciliation
  mechanism is built; this is only a note for whoever designs one once a
  real pile-inspection signal exists.
- **Confidence field on speech-sourced events — do not add yet.**
  `BidMade`/`Passed`/`TrumpCalled`/`TrumpCueHeard` have no `confidence`
  field; `TilePlayed.confidence` does. `speech/bid_parser.py`'s
  `Utterance.confidence`/`BidCandidate.combined_score`
  (`confidence * plausibility`) is a working precedent for
  confidence-weighted speech scoring, so the idea isn't unprecedented — it
  just isn't wired into the engine layer, and shouldn't be guessed at before
  there's a concrete consumer.
- **No shared clock/frame convention between perception and speech** —
  `perception.tile_detect` timestamps with `frame_index: int` (a per-poll-loop
  counter, unrelated to true camera FPS, which itself varies under load);
  `speech.bid_parser`'s `Utterance` uses `start_time`/`end_time` in seconds
  relative to the transcribed WAV's own start, now actually produced (not
  just stubbed) by `speech.transcribe.transcribe_wav`. Needed before a
  played tile and a spoken bid can be ordered against each other and fed
  into the same live `HandState` — this is why `tools/transcribe_session.py`
  only prints a candidate sequence today rather than calling
  `HandState.ingest()` directly. `EventStore.log_event`'s existing wall-clock
  `time.time()` stamp on every ingested event is the likely anchor for both
  sides, rather than a new mechanism. One wrinkle for whoever designs this:
  `tools/live_view.py` records video and audio as two independently
  wall-clock-paced but separate files with no recorded shared T0, so there's
  an unquantified startup-latency skew between them to account for.
- **Engine-side reconciliation for `TileIdentityClassifier`'s ranked
  candidates.** `OpenCVTileClassifier.classify` returns a second candidate
  when exactly one tile half's merged-blob pip count is genuinely ambiguous
  between two readings (both halves ambiguous at once yields no second
  candidate — combining two independent guesses isn't a measurement).
  `EventSegmenter`'s settle window (`_resolve_identity`) only consumes each
  frame's single top candidate; there's no per-frame provenance today for
  threading a settled tile's alternates through to the `TilePlayed` it
  emits. A natural follow-up is engine-side
  reconciliation — when `HandState` logs a conflict for a tile already seen
  elsewhere, try the classifier's next-ranked candidate against tiles not
  yet accounted for before falling back to today's log-and-flag behavior —
  but that needs its own design pass first: how a settled tile's alternates
  survive the multi-frame vote, whether the *earlier* play (not just the
  conflicting new one) might be the actual misread, and how reinterpreting
  a tile's identity stays consistent with `_seen_tiles`/void-tracking/trump
  inference that already ran against the original reading. Not to be
  attempted speculatively — validate any design against real recorded
  footage (`tools/replay_video.py`) before landing it.
- **Self-caught in-the-moment retraction** (a `PlayRetracted` event) and
  rolling back trump-inference/void state for it — today a swapped-in tile
  just gets logged as its own irregularity and the hand keeps going, which
  isn't accurate but isn't a crash either, correctable by a human via the
  post-hoc confirmation channel.
- **Automated retroactive reinterpretation of a late-discovered revoke**
  (unwinding trick winners, replaying trump evidence, re-deriving the true
  void set) — currently logged as a flagged irregularity for a human to
  resolve; no auto-repair attempted. Considered the hardest open case; worth
  real footage before attempting a design.
- **Confidence-tuned, multi-tier hand-outcome classification** — currently a
  single misdeal-override rule; more tiers would need real throw-in examples
  to calibrate against.
- **`game.py`'s `observe_next_dealer` and `events.py`'s `TilesDealt`** are
  fully built, tested, and correct, but have zero production callers today —
  they're waiting on a perception signal (a seat number, a real deal
  observation) that doesn't exist yet. Kept as proven-but-dormant rather than
  cut; revisit once perception can produce that signal to see whether
  they're still the right shape.
- **`observe_next_dealer`'s unvalidated seat number** and **`trick.py`'s
  dead `entitled_leader` field** — no guard built for either, since nothing
  upstream can currently produce the input shape that would need one.
- **`tools/live_view.py` has no way to issue commands (e.g. a bid) from a
  background/automated process.** Its command input is an interactive
  `for line in sys.stdin` REPL loop, which hits EOF and exits immediately if
  stdin isn't a real terminal. Clean shutdown from a background process
  already works today (`SIGTERM`/`SIGINT` are handled and trigger the same
  cleanup as a normal REPL exit) — what's missing is a non-interactive way to
  issue REPL-style commands, e.g. a control file or socket read alongside
  `--no-repl`.

## Explicitly out of scope for now

Named so a future pass doesn't re-propose them without new information: a
real double-dummy solver; bid-plausibility importance weighting of sampled
deals in `probability.py`; any ML/MCTS/CFR-based play or bidding; antithetic
variates or exact-uniform-deal sampling via permanents for the Monte Carlo
estimator; calibration shrinkage, larger sample counts, or new tunable knobs
there; any display/UI work beyond what Phase 4 already scopes.
