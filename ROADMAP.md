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

## Where the in-progress Phase 2 work lives

In the `sqlite-telemetry-live-view` worktree (branch
`worktree-sqlite-telemetry-live-view`). Run `py -m pytest -q` from that
worktree to run the suite (`src/eye42/engine/tests`,
`src/eye42/perception/tests`, `src/eye42/simgen/tests`) — pass `-m "not slow"`
to skip the one real-physics-over-a-full-hand integration test.

## Standalone 3D simulation (`eye42.simgen`)

Phase 0 (a working skeleton, not the full fidelity investment) is built: a
headless MuJoCo physics scene (`simgen/physics.py`), a director that scripts a
full 7-trick hand by reusing `engine.trick`/`engine.probability`'s own tested
legality and playout logic (`simgen/director.py`, `simgen/trajectory.py`), a
pinhole-camera projection + occlusion-correct mask computation
(`simgen/render.py`) that fixes `synth_data.py`'s documented occluded-mask
bug using real 3D z-order, and a working headless Blender/Cycles photoreal
renderer (`render_photoreal`, via plain `bpy` — no `blenderproc` package
needed; it runs standalone in-process, no `blenderproc run` CLI wrapper
required, contrary to what was expected going in). `tools/gen_sim_dataset.py`
CLI ties it together. See RESEARCH.md's "Standalone 3D simulation" section
for the full design rationale.

Each tile's top face carries a pip/divider texture (`_tile_top_texture` in
`simgen/render.py`, reusing `synth_data.py`'s `pip_layout_fractions` as the
shared layout convention) so rendered tiles show the correct pip count and
divider line, not a flat glossy box. A six's pips render as 3 dots across
the length axis by 2 rows down the width axis, matching the Unicode
Standard's own Domino Tiles reference glyphs (verified directly against
them) — an earlier version rendered this transposed (2 columns of 3)
because `pip_layout_fractions`' `(u, v)` means (width-fraction, length-
fraction) per `synth_data.py`'s own stacked-halves convention, and this
texture's side-by-side halves need those axes swapped, not used as-is;
every other pip count (0-5) is unaffected by that swap. `render_photoreal()`
also applies
post-render sensor noise (`_add_sensor_noise`, randomized magnitude per
call) so an artificially noise-free image isn't itself a synthetic-data
tell — an unmeasured placeholder, not real calibration (see below).

`render_photoreal()`'s lighting was badly overexposed until it was fixed: a
300W area light with its emitting size left at Blender's unset 1×1m default,
1.5m above a ~1.3m scene, produced diffuse radiance roughly 8x scene-linear
"white" — a rendered tile body came back HSV saturation=9/value=253 (nearly
white) instead of a saturated green, which silently broke
`eye42.perception.tile_detect`'s real pip-counting classifier (verified by
hand: 0/5 tiles correctly identified against these renders before the fix,
5/7 after — the remaining 2 misses are specular-highlight ambiguity on a
blank/near-blank half, a real difficulty real photography has too, not a
simulation-only defect). Fixed by setting the light's `.size` and `.energy`
explicitly (tuned by rendering and measuring actual output HSV against
`tile_detect.py`'s own thresholds, not a formula alone) and setting
`scene.view_settings.view_transform` explicitly to `"Standard"` rather than
relying on Blender's version-dependent factory-template default. A
regression test (`test_photoreal_render_body_and_pip_colors_land_within_perceptions_own_thresholds`)
pins the rendered body value exactly against `tile_detect.py`'s real
`_TILE_HSV_HIGH` ceiling, and body saturation against a deliberately
tighter guard-band above `_PIP_HSV_HIGH`'s ceiling (not literally
`tile_detect.py`'s own, looser saturation floor) -- the test's own
docstring spells out which bound is which and why.

A Phase 1 hand+forearm occluder rig is also built (`simgen/hand.py`): 13
rigid boxes (2 forearm segments, palm, 2 thumb segments, 4 fingers × 2
segments each) placed by plain-numpy forward kinematics (`hand_parts`), not
a bpy armature/skinned mesh — deliberate, so `render_ground_truth` stays
usable without the `sim-render` extra (a skinned mesh's deformed vertices
can't be projected without `bpy`). `place_hand` gives one parameterized
reach-place-retract path per play/sweep, sampled at a randomized late phase
(released-through-retracting, matching that `FrameSnapshot`s are post-settle
moments, not mid-slide); `tools/gen_sim_dataset.py`'s `--scenes` track poses
a hand at a random table location with independent probability instead
(unconstrained domain randomization, no play/sweep to attach to). Hand/
forearm parts participate in the SAME occlusion computation
`render_ground_truth` uses for tile-tile occlusion — one global depth sort
over tiles and occluder parts together, each part's own projected convex
hull painted individually (not merged into one whole-hand blob), so gaps
between fingers survive as real gaps in a tile's `visible_fraction`/mask,
matching real footage. `render_debug_preview` (the default renderer
`gen_sim_dataset.py` uses) and `render_photoreal` both draw occluders
through the same code path, so a generated image and its ground-truth
labels can never disagree about whether a hand is present.

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
part (matching the pre-existing tile-tile approach), not a true per-pixel
depth buffer — accurate for a palm hovering over a tile (the common case
given this camera's angle) but can misorder a part extending horizontally
toward the camera across a nearer tile; bounded to roughly a tile's own
size by using 2 segments per finger/forearm, named here as the approximation
to revisit (a per-pixel depth buffer) if it ever proves to matter.
`ground_truth.py`'s `_polygons_from_mask` now emits one YOLO-seg label line
per disjoint visible fragment of a tile (matching ultralytics' own
mask-to-label converter's convention of one line per contour, same class —
not a novel scheme), so a tile split into two visible blobs by a finger no
longer silently drops the smaller piece's genuinely-visible pixels;
confirmed against a real simulated hand (14 of 36 frames in one hand
produced at least one fragment split). This means `gen_sim_dataset.py`'s
label semantics now diverge from `tools/gen_synthetic_tiles.py`'s own
`_polygon_from_mask` (deliberately not touched — that compositor's own
per-tile mask is an unsubtracted solid paste region and structurally can't
produce disjoint fragments in the first place, so it stays single-line-
per-tile): if the two data sources are ever combined into one training run,
whatever consumes them needs to know "one label line" doesn't mean "one
tile instance" for data from this module. `RETR_EXTERNAL` still won't
split out a true enclosed hole (an occluder entirely inside a tile's
silhouette, touching no edge) — a separate, narrower, still-unaddressed
case. `_SKIN_COLOR` is an unmeasured placeholder tuned only for contrast
against this scene's table/lighting, not real skin tones.

**Deferred, explicitly out of scope for Phase 0** (see RESEARCH.md and the
plan this was built from): measured roughness/gloss calibrated against real
footage (pips currently share the body's flat `Roughness=0.15`, no separate
matte/gloss distinction); calibrated lens distortion (RESEARCH.md's Tier 3
groups this with sensor noise as needing a real checkerboard calibration
capture, which doesn't exist yet — only that capture-blocked half is still
deferred; the sensor-noise half above is a plausible, uncalibrated
placeholder shipped ahead of it); GPU-rented bulk generation;
retraining/evaluating the YOLOv8-seg model against this new data source.

**Known rough edges to tune, not fixed yet:** `tile_geometry.py`'s
`TABLE_SIZE_M` and `trajectory.py`'s seat rack/won-pile zone coordinates are
still placeholder guesses in absolute meters (no real table measurement or
camera calibration exists yet — see RESEARCH.md's "calibration.json is a
homography, not a camera calibration"), though a regression test now pins a
minimum margin between the table edge and every zone's worst-case reach so
this can't silently regress. `default_camera()`'s oblique *angle* is not a
guess — its near/far edge ratio is verified against `calibration.json`'s real
corner points — but its distance/`focal_px` (how tightly it frames the table)
is still untuned against real footage.

## Not yet built

- **Phase 2 — perception pipeline** (camera → tile/event stream): one-time
  4-corner homography rectification; per-tile crop → 28-class identity
  classifier (small template-match or CNN, trained on the real physical set —
  not Hough-circle pip counting, which fails on the spinner pin and blanks);
  player attribution from observed hand-to-table motion (not turn order or
  fixed seat quadrants alone — the trick winner, and therefore whose turn is
  next, is sometimes undetermined while trump is ambiguous); settle-time-
  debounced event segmentation with explicit trick-sweep handling; a
  shuffle/reshuffle detector for hand/game boundaries, independent of "7
  tricks completed" (won't reliably happen in a set/concession hand);
  concession/redeal entered on a perception signal (mass tile motion, tiles
  flipping, trick-pile mixing), not on the engine's own arithmetic lock,
  since real concessions happen earlier than mathematical certainty.
- **Phase 3 — speech layer** (bidding/trump from audio): continuous local VAD
  + Whisper STT (no cloud API) against a closed vocabulary, scored by
  bid-rotation plausibility × ASR confidence rather than accepting on a bare
  vocabulary match (ordinary table talk containing a number must not read as
  a bid); weak trump cues feed the engine's `TrumpHypothesisTracker` as
  evidence, never as a direct, unconditional trump-set; speaker attribution
  via turn-order rotation, with observed-leader-is-partner treated as
  positive evidence for a splash/plunge bid, not a mismatch to flag.
- **Phase 4 — live dashboard**: switch from record-then-batch to a live
  polling loop; a minimal web dashboard for current bid/trump/trick/score and
  multi-hand marks tracking. The only confirmation channel that doesn't
  interrupt play is post-hoc (in a replay/review UI) — the live view should
  show state as **provisional** wherever it depends on something not yet
  confirmed (trump not yet locked, a hand-end classification pending the next
  hand's dealer), not claim live certainty it can't back up in the moment.
- **Phase 4b remainder — probability visualization**: the engine
  (`engine/probability.py`) is built and tested; the display layer itself
  isn't — a two-team horizontal bar (bid-team % vs. defense %) updating per
  trick, a sparkline across the 7 tricks, and expected remaining defense
  count-points as a secondary stat. Slots into the Phase 4 dashboard once
  that exists.
- **Phase 5 (stretch, only if needed) — robustness hardening**: a heavier
  detector (e.g. YOLO) or multi-camera angles, but only in response to a
  specific, reproducible failure mode that actually shows up in real
  validation footage — not speculatively.

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

- **Splash/plunge is currently unreachable through live bidding.** The
  correct mechanism — infer splash/plunge from who actually leads the first
  trick under a MARKS contract — needs new state (who led trick 1) and its
  own design pass; not yet built. `Contract.trump_caller` already correctly
  resolves to the bidder's partner for SPLASH/PLUNGE, so once this lands, no
  further change should be needed there.
- **Revoke detection has no live path today.** In principle it doesn't need
  any oracle knowledge: by a hand's end, every seat's original holding is
  reconstructable from the union of what they actually played, so a genuine
  revoke could be checked retroactively. Not built.
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
- **Confidence field on speech-sourced events — do not add yet, and note a
  related dead-code question first.** `BidMade`/`Passed`/`TrumpCalled`/
  `TrumpCueHeard` have no `confidence` field; `TilePlayed.confidence` does.
  But `TilePlayed.confidence` is itself currently dead — there are zero
  reads of it anywhere in `engine/` today — and `TrumpCalled` (the event
  type) is
  entirely unconstructed anywhere (`call_trump()` takes raw `(caller,
  trump)` args, not this event). Adding more confidence fields with no
  consumer on top of one that's already unconsumed would be pure unforced
  dead weight. `speech/bid_parser.py`'s `Utterance.confidence`/
  `BidCandidate.combined_score` (`confidence * plausibility`) is a working
  precedent for confidence-weighted speech scoring, so the idea isn't
  unprecedented — it just isn't wired into the engine layer, and shouldn't
  be guessed at before there's a concrete consumer. Separate, smaller
  decision worth making on its own: cut `TilePlayed.confidence` and
  `TrumpCalled` now that both are fully dead, or keep them dormant the same
  way `observe_next_dealer`/`TilesDealt` were kept — not resolved here.
- **No shared clock/frame convention between perception and speech stubs** —
  `perception.tile_detect` timestamps with `frame_index: int`, `speech.
  bid_parser` with `start_time`/`end_time` in seconds. Needed before a played
  tile and a spoken bid can be ordered against each other; not yet defined.
- **Speech bid vocabulary covers point bids only** — mark bids ("two marks,"
  "four marks," `BidKind.MARKS`) have no vocabulary entry in
  `speech.bid_parser` yet, even though the engine fully supports them.
- **`perception.tile_detect`'s `TileIdentityClassifier.classify` needs a real
  implementation** — its interface already returns ranked candidates, not a
  single best guess, matching `TrumpHypothesisTracker`'s weighted-candidate
  convention.
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
  cut; revisit once Phase 2 exists to see whether they're still the right
  shape.
- **`observe_next_dealer`'s unvalidated seat number** and **`trick.py`'s
  dead `entitled_leader` field** — no guard built for either, since nothing
  upstream can currently produce the input shape that would need one.
- **`tools/live_view.py`'s REPL can't be launched from an automated/background
  process directly** — its input loop is `for line in sys.stdin`, which hits
  EOF and exits immediately if stdin isn't a real terminal (e.g. launched via
  a background shell call), with no error, just a silent near-instant exit.
  Workaround today: a small supervisor script that spawns it via
  `subprocess.Popen(..., stdin=subprocess.PIPE)` and keeps that pipe open by
  polling a control file for a `"quit"` sentinel, writing `"quit\n"` to the
  child's stdin to trigger a clean shutdown rather than a force-kill (a
  force-kill skips `cv2.VideoWriter`'s cleanup on Windows, leaving the
  in-progress video segment with a missing moov atom — unplayable, though
  its audio track is unaffected). Not built: an actual non-interactive launch
  mode (e.g. a `--no-repl` flag) that wouldn't need this workaround.

## Explicitly out of scope for now

Named so a future pass doesn't re-propose them without new information: a
real double-dummy solver; bid-plausibility importance weighting of sampled
deals in `probability.py`; any ML/MCTS/CFR-based play or bidding; antithetic
variates or exact-uniform-deal sampling via permanents for the Monte Carlo
estimator; calibration shrinkage, larger sample counts, or new tunable knobs
there; any display/UI work beyond what Phase 4 already scopes.
