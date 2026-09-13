# Research: separating touching tiles and rejecting hands in TileLocalizer

Investigation into one open problem in `src/eye42/perception/tile_detect.py`'s
`TileLocalizer`: a color-mask + shape-filter pipeline finds isolated tiles fine, but (1)
tiles touching edge-to-edge (a boneyard pile, a fanned hand of tiles) merge into one
contour instead of being read as separate tiles, and (2) a hand/forearm in frame
sometimes passes the same green HSV mask and needs to be rejected, not misread as tiles.

Two real-footage fixtures back this investigation:
`src/eye42/perception/tests/fixtures/real_footage_touching_cluster.jpg` (a genuine
3-4-tile touching cluster plus several correctly-isolated tiles) and
`real_footage_hands_in_frame.jpg` (two hand/shadow blobs that pass the tile-color mask,
plus one genuine isolated tile).

## Attempted and falsified

**Per-frame "canonical tile area" comparison.** Compared each contour's area to the
smallest same-frame single-tile-shaped contour, flagging oversized ones as a cluster.
Broke two ways: when the smallest "single tile" in frame was itself a partial tile cut
off at a crop edge, the reference was too small and flagged every genuine tile as an
oversized cluster; separately, against the hands fixture it flagged two hand/shadow
blobs as clusters -- the mask already picks up large green-ish regions near hands that
were previously excluded only by hitting a fixed area cap, not by any real skin-tone
distinction, and one of those blobs had an aspect ratio (1.49) already inside the
*normal* single-tile range, so tightening thresholds wouldn't have fixed it.

**Geometric shape statistics on the merged blob itself** -- convex-hull solidity,
convexity-defect depth (normalized by short side), `cv2.approxPolyDP` vertex count --
do NOT separate a genuine multi-tile cluster from a hand/shadow false positive:

| | solidity | defect depth / short side |
|---|---|---|
| single tiles | 0.95-0.98 | <= 0.05 |
| genuine 2-4-tile clusters | 0.72-0.78 | up to 0.56 |
| hand/shadow blobs | 0.85-0.91 | up to 0.35 |

Hand blobs sit *between* single tiles and genuine clusters on every one of these axes,
so no single-feature threshold works.

**Marker-controlled watershed seeded by distance-transform peaks** (the standard
"touching coins/cells" recipe). Over-segments badly on real footage: the 3-4-tile
cluster (area ~26840) produced 7 peaks instead of ~3-4; a 2-tile merge (area ~18620)
produced 5. Resulting fragments were inconsistent quality (fill ratios 0.42-0.99, some
tiny slivers). Critically, the natural validation step -- checking each fragment's
aspect ratio against the normal single-tile band (1.4-3.2) -- does NOT reject hands
either: the same pipeline run on a hand/shadow blob produced 2 watershed segments with
aspect ratios 2.27 and 2.87, both comfortably inside the "normal tile" band. A consumer
trusting that gate would misread a hand as two real tiles.

**Internal pip-divider-stripe periodicity** (each tile has a bright white divider line;
hypothesis: N touching tiles show N periodic brightness peaks along the cluster's long
axis). Failed on a wrong assumption: real touching tiles in a pile/hand-fan sit at
varied individual angles, not collinear with the merged blob's own overall long axis, so
there's no consistent spatial period once the blob is rotated to one global axis (9-11
irregularly-spaced peaks on the real cluster, not ~3-4 evenly-spaced ones). Also
confounded by each tile's specular-highlight streak, brighter than the true divider line
at this footage's resolution/compression.

**Motion-based discrimination (hands move, resting tile piles don't) -- untested, not
falsified at the time.** Originally blocked: none of the three files in
`eye42_game1_videos/` opened (`moov atom not found` for all three). **Since fixed --
see "Video recovery" below.**

**Net assessment of everything above:** every technique that reduces to "does this
blob's shape/geometry look tile-like" (by whatever metric) keeps landing on hand/shadow
blobs too, because they're similar enough in scale and rough rectangularity under this
camera's resolution/compression to any single-frame shape-based test.

## External research (four independent lines, run in parallel)

**1. Template/chamfer matching, concave-point splitting, and modern segmentation
models.** Ranked recommendation: a small instance-segmentation model
(YOLOv8-seg or similar) trained purely on **synthetic data** -- rendered green
rectangles with pip-divider patterns, randomized position/rotation/overlap, composited
onto backgrounds including real hand photos as explicit negatives ("domain
randomization"). This exact pattern is documented working for closely analogous
problems: multiple independent playing-card-detection projects (rigid rectangles on a
tabletop, needing localization even when overlapping) use purely synthetic training
data successfully, and a separate paper (HaDR, arXiv 2304.05826) documents
domain-randomized synthetic *hand* segmentation for industrial bin-picking clutter --
together these cover both halves of this problem. Second-ranked, lower-effort option to
try first or in parallel: template/chamfer matching or a constrained Generalized Hough
Transform against the known tile rectangle (fixed aspect ratio, roughly known size) --
a fundamentally different test than blob-shape statistics because it's local
(per-candidate-location, template-driven) rather than global-blob-shape-driven; a hand
should score poorly against a rigid-rectangle template fit even where its whole-blob
aspect ratio looks plausible. Concave-point-detection + shape-fitting (from the
overlapping-cell/particle literature) was deprioritized -- most implementation-heavy
option, and its hard step (grouping arcs across 3-4 touching objects at varied angles)
is the same kind of problem that broke watershed and pip-periodicity here. SAM2/FastSAM
held in reserve as a fallback, not a first move -- no documented precedent for this exact
same-color-edge-to-edge-touching case, and that's a known weak spot for boundary-driven
segmentation models generally.

**2. Prior art search (domino/mahjong/card/coin-counting projects) and dedicated hand
detection.** No domino-specific or board-game-specific project has published a solved
touching-tile splitter -- confirmed genuine gap, not a wheel being reinvented. But the
hand-rejection half has a **directly analogous, already-solved precedent**: chess-vision
hobby projects hit the identical problem (small rigid pieces + a hand periodically
entering frame) and solved it by running a **dedicated hand detector** (a custom-trained
YOLO hand model, 97.5% precision / 96.5% recall) to gate/mask the piece-detection
pipeline, rather than trying to make the piece classifier hand-aware. The off-the-shelf
equivalent is **MediaPipe Hands** (free, pip-installable, real-time on CPU, returns hand
bounding region + 21 landmarks, no dependency on skin color/lighting since it's a
trained CNN not a color threshold). Integration path: run it each frame (or every
2nd-3rd), get each detected hand's bounding box, and exclude/suppress any tile contour
overlapping that region above some threshold -- sidesteps needing a color/shape-based
hand-vs-tile discriminator at all. One flagged, untested risk: MediaPipe Hands is
trained mostly on frontal/egocentric views; a hand resting flat viewed from directly
overhead, fingers possibly curled, is a harder case for it -- worth a five-minute
real-footage sanity check before committing. If overhead recall is weak, a small
custom-trained YOLO hand/forearm detector (a few hundred labeled frames) is the
documented fallback, per the chess projects.

**3. Hardware/lighting changes.** Two concrete, complementary recommendations:
- **A purpose-built short-range depth camera** (Intel RealSense D405 or Luxonis
  OAK-D-SR, ~$140-250), used purely for a height mask alongside the existing overhead
  RGB camera. A domino is a rigid ~1cm object sitting flush on a known table plane; a
  hand is several cm to inches thick and moves freely in Z -- a trivial height-band
  filter ("keep blobs 0.5-1.5cm above table plane, reject anything taller") separates
  hands from tiles independent of color entirely. Close to a solved problem in the
  tabletop/surface-computing literature. Whether it also resolves the touching-tile seam
  itself is a "bench-test and see" bonus, not guaranteed -- consumer depth sensors often
  can't resolve the sub-mm gap between two flush tiles.
- **A raking/grazing LED light** (~$15-30, one or two strips at table height, ideally at
  ~90° to each other to catch seams at all orientations) for the seam-detection half
  specifically. This is a core, decades-old machine-vision technique (dark-field/low-angle
  illumination) used industrially to reveal seams/edges on otherwise-flat, evenly-colored
  surfaces -- exactly the domino-seam geometry. Minimally invasive (foldable/removable
  strip), does little for hand-rejection (that's what the depth camera is for).
  Deprioritized: IR/UV tile marking (requires modifying all 28 physical tiles, most
  invasive option for the narrowest added coverage) and a second oblique camera angle
  (trades one hard vision problem -- touching tiles -- for another comparably hard one --
  cross-view correspondence/occlusion reasoning -- where a depth camera gives the same
  "third dimension" information far more directly with a mature SDK).

## Recommendation

This is solvable, but not with a quick heuristic -- every single-frame, shape-only test
tried so far fails for the same structural reason (hand/shadow blobs and genuine tile
clusters overlap on every shape statistic at this camera's resolution/compression). The
fixes that look real all work by getting a genuinely different signal instead of a
smarter shape statistic:

1. **Cheapest next step, software-only:** integrate MediaPipe Hands as an independent
   hand-region mask, gating tile detection -- well-precedented (chess-vision projects),
   low effort, no new hardware. Verify overhead-view recall against real footage first
   (five-minute check) before building the exclusion logic around it.
2. **Most reliable fix, needs hardware:** add a short-range depth camera (~$140-250) for
   hand rejection via height thresholding, optionally paired with a raking LED light
   (~$15-30) for the touching-tile seam.
3. **For splitting touching tiles specifically, if 1-2 aren't enough:** a small
   synthetic-data-trained YOLOv8-seg model -- more implementation investment (a data
   generation/compositing pipeline, a training run) but the only approach that solves the
   splitting problem itself rather than avoiding it.

This item was explicitly "not required for ordinary trick-by-trick play." Treat it as
its own scoped project (plan it properly, likely across a couple of sessions) rather
than something to bolt onto the existing perception module in passing.

## Status

Item 1 (MediaPipe hand-gating) is implemented: `src/eye42/perception/hand_detect.py`
(`HandRegion`, `HandDetector`, `HandGate`), a `TableRectifier.project_points` addition
in `tile_detect.py`, and `HandGate` wired into `tools/live_view.py`'s
`load_perception()`/`Session.on_frame`. See the approved plan at
`C:\Users\jorda\.claude\plans\reactive-waddling-axolotl.md` for the full design
(coordinate-space handling, the motion-gated invocation strategy, the exclusion-margin
sizing rationale, and the opencv-headless/mediapipe coexistence spike that resolved
successfully in this environment before the code was written). The manual live-camera
checklist item in that plan (confirming no spurious plays fire with a real hand in
frame) still needs to be done against physical hardware.

Item 3 (synthetic-data YOLOv8-seg splitter) tooling is built and has been run once,
end-to-end, as a deliberately small proof-of-concept (120 synthetic images, 5 epochs,
YOLOv8n-seg, CPU-only -- no GPU available in this environment): `gen_synthetic_tiles.py`
produced a valid YOLO-seg dataset, `train_tile_segmenter.py` trained and exported to
ONNX with the exact output shapes `tile_segment.py`'s postprocessing expects
(`(1, 37, 8400)` box/mask-coefficient tensor, `(1, 32, 160, 160)` mask protos), and
`tile_segment.py`'s custom decode logic was verified to numerically match ultralytics'
own reference inference path exactly (same confidence values on the same image) -- so
the full pipeline mechanics (data generation -> training -> export -> inference,
output-compatible with `TileLocalizer.find_tiles()`) are confirmed working.

**What this proof-of-concept run did NOT establish: a usable model.** At this scale
(108 training images, 5 epochs) the model's confidence outputs are real but
uncalibrated -- max confidence across all anchors was ~0.05-0.06 even on its own
training images, well below any reasonable detection threshold, though the *relative
ranking* was good enough to produce a non-trivial mAP50 (~0.48) during training's own
validation. On the real `real_footage_touching_cluster.jpg` fixture it detected nothing
at all -- a genuine sim-to-real gap (training data was 100% synthetic, no real-photo
lighting/compression augmentation was applied), not a bug. The trained model/dataset
artifacts from this run were deleted after validating the pipeline (they were a
throwaway proof-of-concept, not intended to be the real deployed model, and a
non-functional model file left in place would just make `test_tile_segment.py` fail
instead of correctly skipping). A real, usable model needs the originally-scoped
low-thousands-of-images dataset, more epochs, and ideally either a GPU or patience for
a much longer CPU run -- unchanged from the original estimate, now with a concrete data
point behind it rather than just a guess.

## Synthetic scene realism: overlap/stacking was miscalibrated against real footage

`synth_data.py`'s original `composite_scene` rejected any tile placement overlapping a
previous one by more than ~5% of its area (retrying up to 40 times, falling back to the
least-bad attempt). A domain review of the actual generated output called this
"unnatural" -- and a careful re-review of real footage (many more frames than the two
fixtures, including shuffle-in-progress and boneyard/trick-pile moments specifically)
confirmed the calibration was backwards:

- Real dominoes **do** visually overlap, not just touch -- e.g.
  `eye42_game1_frames/game1_boundary/1789265460.454.jpg`'s trick-pile cluster, and a
  closer look at `real_footage_touching_cluster.jpg`'s own 3-tile knot, show one tile
  unambiguously lying over another with 30-50%+ of its area covered.
- Overlap is flat (tiles stay lying flat on the table, one crossing diagonally over
  another), not steep-angle leaning -- consistent with ~1cm-thick tiles.
- Density varies by context: a shuffle-in-progress (`1789265304.194.jpg`) is looser,
  tiles mostly separate across ~4-5 tile-widths with occasional light touching; a
  settled trick-pile or boneyard remainder is much denser, packed into a ~1.5-2.5x
  tile-length footprint with heavy mutual overlap common.
- Orientation is fully random in every case observed -- no angle alignment tendency.

External research on how to generate this (physics-based "drop and settle" simulation
via pymunk/PyBullet vs. simpler compositing) concluded **full physics is not warranted**
for this fix specifically: the defect wasn't missing physical accuracy, it was two
concrete bugs -- overlap was explicitly forbidden, and there was no z-order occlusion
(a later-placed tile should visually cover, and its mask should be clipped from, an
earlier one's recorded mask -- `composite_scene` didn't do this). Production
segmentation-training literature (the "Cut, Paste and Learn" line of work) trains real
detectors using exactly this kind of non-physics cut/paste/occlude compositing, since a
detector learns from local occlusion appearance, not global scene physics. Physics
(pymunk for 2D lateral settling, or PyBullet for true 3D) remains a legitimate escalation
if the plain fix still looks wrong specifically in *how* tiles rest against each other,
but is not the first move. **Not yet implemented as of this writing** -- the concrete
fix (allow overlap per the density evidence above, add z-order occlusion-clipping to
`SyntheticTileInstance.mask`) is scoped but was superseded by the motion-tracking pivot
below before being applied.

## Pivot: tracking dominoes through motion, not just static clusters

Mid-session, the goal was reframed: beyond splitting a static touching-tile cluster, can
the pipeline identify/track a domino through active motion (a shuffle, a slide, the act
of a tile being played), reviewing actual video to ground this rather than guessing.

**Physical constraint, not a software limitation:** a domino's pip pattern is genuinely
unrecoverable from a motion-blurred frame at typical webcam shutter speeds -- this is an
optics fact, not something more code or better training data changes. Confirmed the
scoped goal is: track a tile's position/identity continuously through motion (so it's
never "lost and rediscovered" as a new object once it settles), and only commit its pip
value once it's actually still enough to read -- plus the same for the specific motion
of a tile being played, not just generic shuffling.

**External research on physics-based motion simulation for this** (PyBullet/MuJoCo for
tile dynamics, directional-blur compositing for realistic training frames) found:
domino chain-reaction sims are a trivial, well-trodden physics-engine demo, but a
*shuffle* (many tiles in sustained simultaneous contact, plus an external "hand" actor)
is a fundamentally different, harder simulation regime with no existing domino-specific
example to adapt -- closer to a granular-flow/bin-picking manipulation problem. More
importantly, the research's own recommendation was skeptical of the physics-simulation
premise as a first move: **sports multi-object-tracking prior art (tracking-by-detection
+ Kalman filter/appearance re-identification) solves "maintain identity through a
blurred/occluded moment" without any synthetic motion training data at all** -- identity
survives via track continuity across frames, not by successfully classifying the single
blurriest frame. That's a much smaller, cheaper thing to try first against real footage
directly than a physics-and-render pipeline, and this codebase already has a relevant
precedent to build from: `EventSegmenter`'s existing motion-history machinery (built for
seat attribution, see `tile_detect.py`).

**Real motion video was needed to pursue this at all, and was blocked, then unblocked --
see "Video recovery" below.**

## Motion-tracking design research (nine questions, external + real-footage)

With the video recovered (below), nine questions were researched in parallel to answer
before designing anything: footage motion characteristics, 2D-vs-3D feasibility,
tracking-algorithm choice, the physics engine's actual job, play-event detection,
multi-object identity-switch risk + testing strategy, real-time performance/pipeline
integration, generalization beyond this one session's setup, and Texas-42-specific
motion edge cases. Findings below; a proposed design follows.

**Footage reality, contradicting the earlier "motion-blur" framing above.** Both
`game1_boundary.mp4` and `game1_final.mp4` are a clean 1920x1080 @ 25fps constant. Sampled
bursts show plays/draws are unhurried and sharp -- no visible motion-blur streaking even
during active manipulation, at this camera's frame rate and lighting. The "pip pattern is
unrecoverable from blur" framing above turns out to be less of a live constraint than
assumed; the real difficulty is occlusion (a hand or another tile covering the pips), not
optical blur. Two real off-plane behaviors were confirmed, though, that break a pure flat-
table assumption: (1) a shuffle produces genuine stacking -- tiles resting on top of each
other, not just edge-adjacent, 30-50%+ area overlap in the camera projection, both hands
of multiple players kneading a 30-40 tile pile at once; (2) dealt hands are stood on edge
in front of each player -- from directly overhead these read as thin vertical strips, a
different silhouette than a flat tile entirely, not just an in-plane rotation.

**A clean single-tile "play" event was never actually isolated**, across two independent
footage-analysis passes. Both searched for it by sampling the highest-motion windows in
the video, and both passes landed on shuffles every time -- shuffles dominate raw motion
magnitude, so that search strategy structurally can't find a play (a much smaller, more
localized motion). This means several Texas-42-specific questions below are genuinely
open, not merely deprioritized: whether a double/spinner tile's 90-degree placement looks
different from a normal tile's slide, whether boundary/edge-of-rectified-plane plays
introduce homography distortion, and whether a player nudging/straightening an already-
played tile risks being double-counted as a second play. Finding an actual play event (by
searching for moderate, spatially-isolated motion in one table quadrant rather than
sorting by raw motion magnitude) is unfinished, necessary groundwork, not a nice-to-have.

**2D vs. 3D.** Monocular height/pose recovery from a single fixed overhead camera with no
depth/stereo is fundamentally ill-posed -- shadow-based and reference-plane height cues
exist but are fragile and coarse, not metrically precise, and break entirely once a tile
leaves a calibrated plane (e.g., lifted into a hand). The closest real precedent for this
exact problem shape (flat rigid game pieces, fixed overhead camera, occlusion during play)
is TCG-AR, a production real-time trading-card-tracking system -- it solves detection,
orientation, and identity entirely with 2D compositing for synthetic data and 2D tracking,
no 3D physics or rendering anywhere. No source found evidence that 3D rendering fidelity
measurably improves a 2D-consuming model's real-world accuracy over 2D compositing; the
cut-paste-learn literature's finding is that placement/scale/blending realism matters, not
3D dynamics. The off-plane footage behaviors above (stacking, edge-standing hands) don't
overturn this -- they're better handled as distinct *states* (in-hand vs. on-table) with a
transition between them (the play event) than as continuous 3D poses to reconstruct.
Recommendation: build 2D, on the existing rectified plane, with a hand/interaction model
for the in-hand state -- not full 3D physics + rendering. The one legitimate trigger for
revisiting 3D: a future goal of *replaying* physically plausible tumbling/sliding motion
for visualization, which 2D compositing structurally cannot produce.

**Superseded, see "Standalone 3D simulation" below.** The "no evidence 3D rendering
fidelity beats 2D compositing" claim above does not hold for glossy, texture-less objects
like these tiles -- the BOP Challenge 2020 benchmark found switching from flat-shaded
synthetic renders to ray-traced PBR rendering swung detection accuracy from 6.1% to 64.0%
on reflective, texture-less objects, a case much closer to this project's tiles than the
TCG-AR/cut-paste-learn precedents cited above (whose objects weren't glossy). Separately,
the project's goal changed mid-session from "extract 3D signal from the real camera" (where
the conclusion above still holds -- monocular pose inference from this camera is genuinely
not viable, see "3D vs 2D" further below) to "build a standalone 3D simulation, independent
of the real camera, for training/validation data and eventually interactive/VR use" -- a
different goal the paragraph above never evaluated.

**Tracking algorithm.** Standard MOT (SORT/DeepSORT/ByteTrack) is built for visually
distinguishable objects with mostly-divergent trajectories; domino tiles are the opposite
case -- visually near-identical, and the one distinguishing feature (pip pattern) is
exactly what's unreadable during occlusion. Appearance-based re-identification doesn't
help here; it would just be indirectly re-deriving pip counts at the moment they're least
readable. The closest real prior art (chess-piece-tracking robotics, poker/mahjong vision
systems) doesn't track through motion at all -- it detects stable board/hand state before
and after, and infers the move/play from the diff. That's architecturally identical to
`EventSegmenter`'s existing settle-time debounce. Recommendation: don't build general MOT.
Use a lightweight motion tracker (simple centroid/Kalman, constant-velocity) only to
bridge one hand-driven motion event from before-state to after-state, and let the existing
settle-based pip-reclassification remain the sole authority on identity once a tile stops
moving.

**The physics engine's actual job.** Three distinct roles exist and were being conflated:
(a) generating synthetic training data with plausible pile/resting configurations, (b)
serving as a real-time motion prior during live tracking, (c) joint physics-informed state
estimation. Only (a) has real payoff here, and only for *static resting realism* (the
overlap/stacking fix already scoped below) -- not for simulating in-flight collision
dynamics or blur, where no evidence favors true rigid-body sim over simpler scripted
motion. For (b), a plain Kalman filter (already the standard, proven MOT primitive) is the
right tool for bridging brief occlusion; true rigid-body simulation adds friction/
restitution parameter-estimation burden with no evidence of better short-horizon
prediction. (c) is research-grade and not appropriate at this scale. Recommendation:
pymunk (2D, not PyBullet/MuJoCo/3D) if a physics engine is used at all, scoped narrowly to
generating better synthetic pile configurations -- not for live tracking or motion
simulation.

**Superseded, see "Standalone 3D simulation" below.** This recommendation was scoped for
physics as a narrow helper to the 2D pipeline. Under the project's later-adopted goal (a
standalone 3D simulation as its own deliverable), a full 3D physics engine (PyBullet, or
MuJoCo >=3.12) is the right choice -- not because live tracking or blur simulation need it
(they still don't), but because the simulation itself is now the goal, not just a synthetic
2D image source.

**Play-event detection.** The tabletop-game vision literature (chess, poker, mahjong)
converges on "stable state -> motion/occlusion -> stable state, diff the two" as the
standard pattern -- which `EventSegmenter`'s settle-time debounce already implements.
Hand-shape grasp/release classification (open vs. closed hand) was considered and is
likely unreliable at this overhead, oblique, frequently-finger-occluded camera angle, and
would be redundant with what settle-time debounce already encodes. The one cheap addition
worth considering: tag a confirmed play with whether a hand region was present near its
position just before settling (data already computed by `HandGate`), to distinguish "a
hand placed this" from "this tile was merely uncovered by something else moving" -- no new
model or velocity math needed.

**Multi-object identity-switch risk and how to test any of this.** A 28-tile shuffle,
where most tiles move simultaneously with heavy mutual occlusion, is close to a
worst-case scenario for any motion tracker -- Kalman-based re-association degrades as
occlusion duration and simultaneous-mover count increase, and there's no real prior art
for reliably tracking identity through totally chaotic near-identical-object motion
without physical markers. This is confirmed as the *normal* case for a shuffle by the
Texas-42 footage review below (all four players' hands over the pile at once), not an edge
case to design around later. Recommendation: don't attempt to track identity through the
shuffle at all -- treat it as an opaque event (motion detected, then a new stable table
state), and scope active motion-tracking to hand-to-table plays only (2-7 tiles, one
principal mover, far less simultaneous occlusion), which is also all the game engine
actually needs identity for. Testing: skip formal MOT metrics (MOTA/IDF1) as
disproportionate to a hobby project's scale; use scripted synthetic motion sequences with
known ground truth for deterministic tracker-logic unit tests, plus a small number of
real-footage clips checked only for coarse before/after outcomes ("tile last seen at
position X in a hand ends up played at position Y"), not full per-frame trajectory
annotation.

**Real-time performance and where this fits in the existing pipeline.** The real
footage is 1920x1080 @ 25fps. YOLOv8n-seg ONNX inference at a typical 640px input runs
roughly 80-200ms/frame on CPU (5-12fps) standalone, before the existing rectify/mask/
contour/classify/hand-gate stages -- not real-time at full frame rate without gating.
Recommendation: mirror `HandGate`'s existing motion-gated pattern (only run expensive
inference on motion-flagged frames, or only on ROI crops where the contour pass already
flagged a plausible merged cluster) rather than running it on every frame. A Kalman
tracker's own per-frame cost is trivial by comparison. Architecturally, a tracker slots
into `Session.on_frame` as one more additive stage (after `hand_gate.filter`, before
`segmenter.feed`), the same insertion pattern `HandGate` itself used -- no restructuring
of `observe_frame()`'s single-frame call shape needed, and no regression risk to the 148
existing tests, which don't touch this insertion point. Strong recommendation: prototype
tracking/segmentation entirely offline against the recovered `.mp4` files first (batch,
no real-time constraint) before wiring either into the live `tools/live_view.py` path --
this is also a good fit for Railway's available CPU capacity (up to 24 vCPU/replica on
the Pro plan, no GPU offered anywhere on the platform) for a long batch run, rather than
the real-time-on-a-laptop question at all.

**Generalization beyond this one session's setup.** `TileLocalizer`'s fixed HSV band and
`synth_data.py`'s fixed pure-green/pure-white synthetic colors and implied fixed lighting
are both overfit to this one session's camera/room/table -- a well-known fragility for
color-threshold detection, and domain-randomization literature is explicit that lighting/
color-temperature/exposure randomization in the synthetic generator is what actually
produces real-world generalization, not present here yet. Manual 4-corner calibration
needing to be redone per session is normal/accepted practice at this scale, not worth
automating yet. Recommendation: add lighting/color/exposure randomization to
`synth_data.py`'s generator before the next full-scale training run (cheap now, expensive
to redo after discovering it doesn't generalize), and validate against footage from at
least one more physically distinct setup before considering the perception pipeline done.

**Texas-42-specific motion edge cases.** Cross-player multi-hand occlusion during a
shuffle is confirmed real and is the *default* case, not an edge case (all 4 players'
hands over/near the pile simultaneously) -- consistent with the identity-switch finding
above, and not a risk to the existing seat-attribution mechanism, which already reasons
via per-tile motion-history trails (not a "closest hand" heuristic) and already excludes
shuffles from attribution via its sweep-motion flag. Double/spinner-tile rotation,
boundary-of-rectified-plane plays, and post-placement re-adjustment (risk of double-
counting a play) remain **unanswered** -- no isolated single-tile play event was found in
either footage pass (see above). These need a real answer before the play-event design is
finalized, not an assumption either way.

## Video recovery

All three files in `eye42_game1_videos/` failed to open (`moov atom not found`) --
previously documented (wrongly) as only affecting the one file already named
"corrupted." Diagnosed and fully repaired this session, not just described as a future
task:

- **Diagnosis:** a raw box-level scan showed all three files have a fully intact `mdat`
  box spanning virtually the entire file (397-844MB) -- the actual encoded video data
  was never lost, only the `moov` index (sample table, timing, codec config) was never
  written, consistent with the process being force-killed before `cv2.VideoWriter`
  finalized the file (see `ROADMAP.md`'s note on `tools/live_view.py`'s REPL/relaunch
  gotcha).
- **Codec:** the raw stream is MPEG-4 Part 2 (`cv2.VideoWriter`'s `mp4v` fourcc), not
  H.264 -- confirmed via the `000001b3` GOV-header start code at the start of `mdat`.
  Unlike H.264, ffmpeg can demux a raw MPEG-4 Part 2 elementary stream directly with no
  container index at all, IF the stream includes its own VOL header (width/height/etc.).
  It didn't -- this encoder's output stores that config only in the container's `esds`
  box, not repeated in-band in the bitstream, so the raw `mdat` payload alone was not
  self-describing.
- **Fix:** generated a small fresh reference video with the identical encoder
  (`cv2.VideoWriter`, `mp4v`, same resolution) on this machine, which produces a normal,
  valid `moov`/`esds`; extracted the 47-byte VOL header (`000001b0...`) from that
  reference's `esds` decoder-specific-info, and prepended those exact bytes to each
  broken file's raw `mdat` payload before remuxing with `ffmpeg -f m4v -c copy`. All
  three files opened cleanly afterward with correct duration/resolution/frame count, and
  a decoded frame was visually confirmed to show real, correct session footage.
- **Result:** `game1_boundary.mp4` (14,027 frames, 561.08s), `game1_final.mp4` (15,447
  frames, 617.88s), `game1_boundary.pre-restart-corrupted.mp4` (37,870 frames, ~25min) --
  all now open and seek correctly via `cv2.VideoCapture`/PyAV. Originals preserved
  (unmodified) under `eye42_game1_videos/broken_originals/` rather than overwritten.
  `imageio-ffmpeg` (pip-installed, bundles a static `ffmpeg` binary) was used for the
  repair; no system-wide ffmpeg install was needed or made.

## Audio recording bug found and fixed (`AudioRecorder`)

While cross-checking the repaired videos' duration against their paired `.wav`
recordings, found a real, reproducible discrepancy: `game1_boundary` video runs 561.08s
but its audio is only 506.7s (54.4s / 9.7% short); `game1_final` video runs 617.88s but
its audio is only 515.6s (102.3s / 16.6% short). Both audio files are otherwise valid,
readable WAVs -- not corrupted, just short.

**Root cause, found by reading `tools/live_view.py`, not assumed:**
`AudioRecorder._on_audio` (the `sounddevice` input callback) completely ignored the
`status` argument the library passes specifically to report a dropped/overflowed
callback, and unlike `VideoRecorder` (which already has a measured-necessary wall-clock
catch-up mechanism for exactly this kind of thread contention under perception load),
had no equivalent compensation -- a dropped callback just silently shrank the WAV file
with no record of it happening. Ruled out simpler explanations first: `start_consumer`
spawns a non-blocking thread, so video/audio recorder construction happens within
milliseconds of each other, not tens of seconds apart, and the shutdown order
(`camera.stop()` -> `recorder.close()` -> `audio_recorder.close()`) would if anything
make audio slightly longer at the tail, not dramatically shorter -- neither explains a
54-102s deficit. The size and direction of the deficit (larger on the longer session)
matches cumulative small dropped-buffer loss under sustained contention, the same
category of problem `VideoRecorder`'s own docstring already documents as measured, real,
and significant enough to require a dedicated fix.

**Fixed:** `AudioRecorder` now tracks wall-clock elapsed time against samples actually
written, padding with silence whenever a gap opens up (mirroring `VideoRecorder`'s own
padding-to-catch-up design) so a session's WAV duration keeps matching real elapsed time
regardless of dropped callbacks, and logs a count of flagged callbacks on `close()`
rather than staying silent. Verified with a real-time-paced simulated dropped callback:
written duration tracked real elapsed time to within normal scheduling jitter. Note: the
audio already lost in the two existing recordings is **not recoverable** -- those
samples were genuinely never written, unlike the video's `moov`/VOL indexing problem
where all the real data was intact. This fix only prevents the same loss in future
sessions; it doesn't reconstruct what's already missing, and because the loss was likely
many small drops scattered through the session rather than one contiguous gap, no single
global time-shift will cleanly re-sync the existing two files to their video throughout.

## Real play events, finally found -- and two prior findings corrected

Two prior passes (above) failed to find an isolated single-tile play by sorting frames by
raw motion magnitude -- that strategy structurally can't work, because Texas 42 is
trick-taking (tiles are tossed loosely to a central trick area, never a connected line --
see "Domino object physical properties" below), so the correct spatial prior is the
table's centre, and shuffles have to be separated from plays by *extent and duration*, not
magnitude. Re-searching on that basis (masked motion inside/outside the calibration quad,
event = burst bracketed by quiet, adjudicated by eye) found nine confirmed plays and one
complete four-tile trick traced start to finish (`game1_boundary.mp4` f3190-f3735).

**Two things this session previously got wrong, now corrected by direct measurement:**

- **The footage is not really 25fps of information.** Both files are 25fps CFR containers
  holding runs of 2-3 duplicate frames; real content rate is ~8-13fps, so a play is only
  5-9 genuinely distinct images, not 15-17. Any timing-sensitive design (a Kalman filter's
  frame-to-frame velocity assumption, an "N frames of settle" debounce threshold) needs to
  reason in content-frames, not container-frames. Dropping duplicates before processing
  also cuts decode/inference cost by roughly 60% for free.
- **Motion blur is real, contradicting the earlier "no motion blur" finding.** Measured
  directly (Laplacian variance on a tracked tile patch): a 4.2x sharpness drop at peak
  slide velocity, visually confirmed as unreadable pip smearing for 2-4 content frames.
  The earlier finding was very likely an artifact of having sampled duplicate frames,
  which are by construction sharp copies of a settled instant. Blur self-resolves within
  about 0.2s, so it doesn't change the settle-based design's soundness, but it does mean
  a "just read pips mid-motion because there's no blur" shortcut isn't available.

**Two play styles, with very different occlusion profiles:**

- **Slide** (the dominant style, 6 of 9 confirmed plays): the tile is pushed flat across
  the table, fingers trailing, tile leading -- **fully visible for the entire transit**,
  with substantial incidental in-plane rotation (~90 degrees over a 0.4s slide in one
  case). This is a nearly-free tracking channel a design expecting the tile to be hidden
  during a play would underuse.
- **Carry-and-set** (2 of 9): the tile is pinched edge-on, carried through the air, then
  dropped/set flat -- briefly (~25% of the event) fully face-occluded while airborne.

A double (0-0 blank) was found being played (`game1_boundary.mp4` f3662-f3677) with an
unremarkable motion profile matching non-doubles -- consistent with the "no spinner rule in
42" finding below. A pipped double (e.g. 6-6) being played was not found or specifically
searched for exhaustively.

**The decisive finding for play-detection design:** a genuine tile nudge/re-adjustment
(no play occurred) was found (`game1_final.mp4` f11716-f11829) with **longer duration and
roughly 2x the motion magnitude of two genuine confirmed plays**. No threshold on motion
size or duration can separate a real play from a meaningless nudge -- only a before/after
full-table-state diff can. This confirms `EventSegmenter`'s existing settle-debounce
architecture isn't just the cheaper choice, it's the only one the footage supports.

**A genuine, previously-unknown gap:** the calibration quad only covers the central trick
area. Racks, won-trick piles, and tiles from the deal/draw regularly land entirely outside
it (e.g. a tile at boundary f5210, well past the W corner). Any design assuming "everything
game-relevant sits on the rectified plane" is wrong today; either widen the calibration or
model off-plane regions explicitly.

Reliable negative fixtures identified for future regression tests: shuffle (`final`
f6680-6960; `boundary` f1715-1980), trick collection/sweep (`final` f5900-5960), a
no-play nudge (`final` f11716-11829), hands-only-no-tiles (`boundary` f9869-9919).

## Domino object physical properties

Researched the physical object itself (dimensions, materials, manufacturing, and this
project's specific set) since several open questions (PnP feasibility, physics-sim
friction/restitution, pip-layout ground truth) depend on it.

**This project's real set:** green-bodied, white-pipped (confirmed from `tile_detect.py`'s
own HSV comment -- "this tile set's green" -- a real physical choice, not synthetic
invention), standard double-six dimensions and 2:1 long:short aspect (matching
`tile_detect.py`'s existing aspect filter). Close inspection of real fixtures found: pips
read as flat with no shading (consistent with printed, not drilled, pips), no visible
metal spinner pin on any double examined, the center divider is a clearly visible solid
white line, and the real pip grid layout matches `synth_data.py`'s `_GRID_POSITIONS`/
`_PIP_LAYOUTS` template well. **One real gap found and not yet modeled synthetically: the
tiles are glossy enough that specular highlights blow a tile's corner out to near-white,**
visually merging with the background and risking contour fragmentation -- `synth_data.py`'s
flat-color renderer doesn't produce this at all.

**Standard manufacturing facts** (double-six sets generally): ~28x56x10mm face and
thickness at typical scale (2:1 ratio holds across cheap-to-professional grades); melamine
plastic is the dominant material, glossy-finished; pip layout is universally the same
"dice-pattern" grid across manufacturers; a metal spinner rivet is common on
professional/tournament sets specifically to reduce wear when a double is spun -- but
**this is a manufacturing feature only, not a gameplay mechanic in Texas 42.**

**Important rules correction, confirmed by two independent sources (external rules
research and direct footage review):** Texas 42 is a **trick-taking** game -- tiles are
played to a central trick and collected, never connected into a line/layout the way block
dominoes are. The "double played perpendicular at a spinner" convention belongs to
block/layout games (Muggins, All Fives), **not to 42.** This invalidated an earlier
assumption in this document's motion-tracking research (a "spinner tile placement" question
was open; it's now resolved as inapplicable) and reframes what a "play" physically looks
like: tossed into a shared central area, not laid along a growing boundary line.

## Identifying a tile before it's fully at rest

The current classifier (`OpenCVTileClassifier._count_pips`) counts pip blobs and discards
their *positions* -- this turns out to be the single highest-leverage thing to fix,
independent of anything else in this section. Exhaustive enumeration over
`_PIP_LAYOUTS`'s 7 canonical layouts shows: knowing which of 7 specific grid cells are
occupied (not just how many) lets a partial/occluded view either certify a half's value
with proof, or -- critically -- **prove itself invalid** when the visible cells don't match
any of the 7 real layouts. A count is always in 0-6, so it's always "valid," so a
count-only read of an occluded tile is *always* silently wrong; a position-aware read is
usually *visibly* wrong instead. With one cell occluded, position-aware reading is
calculated to be about 7x less likely to be silently wrong than today's count-only
approach. A specific hazard confirmed: occluding the two "6-only" cells (ML+MR) turns a 6
into a false 4 -- the single most dangerous 2-cell occlusion case, and now a nameable,
checkable predicate rather than a vague worry.

**On aggregating classifications across a tracked tile's motion:** running the classifier
on every frame and keeping the highest-confidence read (an intuitive first idea) is the
specific variant the literature says performs *worst* -- it reliably selects the frame the
classifier was most overconfident on, not the most correct one, and this project's own
classifier already emits a hardcoded confident `1.0` that can be confidently wrong under
partial occlusion. The right version is confidence-weighted accumulation across frames,
explicitly tempered because consecutive video frames are correlated evidence, not
independent samples (this compounds with the frame-duplication finding above -- duplicate
frames are *zero* new evidence, not weak evidence, and must be skipped, not down-weighted).

**On closed-world elimination** (a standard double-six set has exactly 28 unique tiles, so
knowing 27 identifies the 28th by elimination): weaker than intuition suggests as an
*early-resolution* aid -- it saves relatively little until the endgame, and simulation
shows it barely accelerates a partial read's resolution mid-hand. It's much more valuable
as an **error-catcher**: silently rejecting an identification that names an already-played
tile, worth an estimated 25-35% relative reduction in wrong commits at near-zero cost. Must
be applied as a soft prior (down-weight, never hard-remove a tile from the candidate pool),
since a wrong earlier identification would otherwise cascade errors into everything after
it.

**Recommendation, in dependency order:** (1) make the classifier position-aware over the 7
grid cells, with an explicit valid-layout check -- highest leverage, no new model or
training data needed; (2) define "unoccluded enough" as a structural predicate (at least
one cell confirmed from each of 4 specific cell-groups, not a vague confidence threshold);
(3) accumulate per-track belief across content-frames only (never duplicates), tempered,
never by max-confidence; (4) apply closed-world elimination as a soft multiplicative prior
after the above, not before. This decouples "when did a play happen" (settle-debounce stays
authoritative for this) from "what tile was it" (can resolve earlier than settling, with
the debounce as a backstop, never worse than today).

## Standalone 3D simulation: from "extract 3D from the real camera" to "build a 3D world"

Mid-session the goal changed from "can 3D information be extracted from the single real
camera" (researched and answered above/below: no, not usefully -- see "3D vs 2D") to a
different, larger goal: **build a standalone, high-fidelity 3D physics + rendering
simulation of a Texas 42 game -- a game-engine/VR-like environment, functioning
independently, which eye42's perception program can observe and interact with** to train,
test, and experiment against, rather than relying on scarce real footage. This section is
the synthesis of that research thread.

### 3D vs 2D for extracting information from the *real* camera (settled, separate question)

Monocular 6-DoF pose estimation of a tile via known-geometry PnP (a real, different
technique from generic monocular depth, and worth investigating on its own merits) was
measured against this project's actual camera geometry: a tile's thickness (~10-12mm) is
the quantity that would need resolving, but the best achievable monocular depth precision
at this camera's distance/resolution -- even using published numbers from purpose-built
high-contrast fiducial markers, which are far easier to localize than a domino's rounded,
glossy corners -- is on the order of +/-15-25mm of noise. The signal is smaller than the
noise floor; this is fixed by focal length, distance and tile size, not fixable by better
software. A flat tile near-overhead is also a textbook case for PnP's two-fold pose
ambiguity. Every candidate value (occlusion z-order, rack-vs-table state, motion tracking)
either is already available for free from the existing timestamped observation record, is
better served by simple 2D features (a rack tile's aspect ratio is already far outside
`_MAX_TILE_ASPECT`), or is actively made worse by PnP's discontinuous failure under
occlusion versus a centroid tracker's graceful degradation. **If real 3D sensing is wanted
from the real camera, the answer is a depth camera** (Intel RealSense D405 / OAK-D-SR,
~$140-250, 1-5mm accuracy at this range -- comfortably under the 10mm signal needed), not
monocular pose math -- and it would also solve the long-standing, still-unsolved
hand-vs-tile HSV false-positive problem as a side effect. This conclusion is about the real
camera specifically and doesn't apply to the standalone-simulation goal below, since a
simulation's virtual camera/ground truth isn't limited by real-world sensor physics.

### Engine and tooling: two different goals need two different stacks

Research here split cleanly along the difference between "generate offline training/
validation data" and "build an interactive, VR-capable game" -- and arrived at genuinely
different tool recommendations for each, which is a real fork requiring a decision, not
something to resolve unilaterally:

**For offline, photorealistic training/validation data generation:** decouple physics from
rendering. Recommendation is **PyBullet (or MuJoCo >=3.12 as a second opinion) for physics,
headless and CPU-fine, orchestrated with BlenderProc2 driving Blender/Cycles for
rendering.** This follows the same architecture as Kubric (Google's synthetic-data
generator) and BOP-challenge dataset generation. Key reasons: Cycles is the only renderer
on the table that's both CPU-capable *and* physically based (Eevee has no CPU path at all,
per Blender's own developers), and rendering fidelity turns out to matter a lot here --
BOP Challenge 2020 found switching flat-shaded synthetic renders for ray-traced PBR
rendering swung detection accuracy from 6.1% to 64.0% on reflective, texture-less objects,
much closer to this project's glossy tiles than the 2D-compositing precedents cited
earlier in this document. BlenderProc2 specifically (not raw `bpy` or Kubric directly) is
recommended because it already solves camera-intrinsics-from-K-matrix and OpenCV-compatible
Brown-Conrady lens distortion, has BOP-format ground-truth export built in, and is
actively maintained (Kubric is pre-alpha and version-brittle).

**For an interactive, VR-capable game/simulation:** Blender has two hard, non-negotiable
ceilings here -- no runtime at all (the Blender Game Engine was removed in 2019; there's no
way to ship or play back an interactive scene) and no real VR game support (its OpenXR
addon is a look-around design-review viewer, not an interactive experience with grabbing/
game logic). There's no incremental migration path from a Blender-only pipeline to either
capability; it's a rewrite. If interactivity/VR is a real near-term goal, not a someday
maybe, **Unity is recommended over Unreal or Isaac Sim**: Unity Perception (first-party
synthetic-data/ground-truth tooling) and a genuine runtime Python bridge (ML-Agents' gRPC
channel, or a plain socket) both beat Unreal's offerings for this specific need, while
Unreal's real strengths (Nanite, Lumen, photoreal MetaHumans) are mostly wasted on 28
dominoes on a table; Isaac Sim was reconsidered seriously (not dismissed as "overscoped")
and ruled out because it's a robotics simulator, not a game engine -- it structurally can't
deliver a playable/VR experience regardless of its (real, and better than Unity's)
synthetic-data pipeline, and its hardware floor (RTX 4080 16GB minimum, 32GB RAM) is
steeper than Unity's.

**These two recommendations are not in conflict, they're for different deliverables.**
Blender/PyBullet is the better choice specifically for generating the best possible
training/validation imagery; Unity is the better choice specifically for a playable/VR
experience. Which to build first (or whether to build both, sequenced) is an open decision,
not resolved here.

### Hardware: no GPU is a procurement decision, not a fixed constraint

Direct inspection found the actual dev machine (a 4-core/8-thread ultrabook, 16GB RAM,
integrated graphics only, no PCIe slot) is a harder constraint than "no GPU" implies --
there's nowhere to add a card. The earlier "solo hobbyist, no GPU, CPU-only" framing this
session used for prior research passes was itself hardware-determined, not merit-based.
Renting is confirmed cheap: consumer GPU rental (RunPod, Vast.ai) runs roughly
$0.30-0.70/hour for a 4090-class card; a full 10,000-frame Tier-2-quality render batch
(see fidelity tiers below) costs on the order of $20-35 and a few hours to a day, versus
roughly $185-280 and 7-10 days on Railway's CPU-only pricing for the same job (Railway CPU
is both slower and more expensive than renting a GPU for this workload -- not a close
call). Buying a desktop (~$1,300-1,800 for an RTX 5070 Ti class machine) was researched and
would unlock Unity/interactive work plus independently unblock the already-scoped-but-never-
run full YOLOv8-seg training pass -- but the user has decided to rent rather than buy, for
now. Renting comfortably covers the offline Blender/PyBullet training-data path (the
recommended stack there runs identically on CPU and GPU, just flip a device flag); it does
not cover sustained interactive Unity/VR *development* work, which would want a persistent
rented GPU workstation rather than an ephemeral batch pod -- a distinction worth keeping in
mind if the Unity path is chosen, not something to resolve now.

**A real prerequisite, independent of which path is chosen:** `calibration.json` is a
4-point homography, not a camera calibration -- it has no focal length, no distortion
coefficients. A proper checkerboard/ChArUco calibration (`cv2.calibrateCamera`, ~2 hours)
is cheap now and becomes expensive to retrofit once scenes/ground truth already exist
against a guessed camera model. Worth doing early regardless of which simulation path is
chosen.

### Fidelity tiers

Three tiers were scoped, each unlocking a different real capability:

- **Tier 1 (geometrically correct, visually approximate):** physics-settled tile layouts
  and scripted plays, simple materials, no camera distortion/sensor model. ~4-8 weeks.
  Unlocks a **logic/geometry regression testbed** (rectification correctness, tile-count
  invariants, occlusion-order reasoning, settle-debounce behavior) but explicitly does
  **not** unlock training a detector that transfers to real footage -- confirmed by this
  project's own earlier proof-of-concept, which produced zero real-footage detections
  from exactly this kind of flat-shaded synthetic data.
- **Tier 2 (appearance-faithful):** proper glossy-plastic PBR materials matched to real
  footage, realistic lighting including the corner-highlight-blowout behavior found in real
  tiles, domain-randomized lighting/color/camera jitter. ~6-10 additional weeks. This is
  the tier the evidence says is actually needed for training data that transfers, given
  these are glossy objects (see the BOP T-LESS result above) -- and it's the tier at which
  retrying the failed proof-of-concept has a real chance of working.
- **Tier 3 (camera-faithful):** adds calibrated lens distortion, sensor noise, compression
  artifacts matched to the real capture chain. ~4-8 additional weeks. Unlocks genuine
  paired sim/real validation (a performance delta measured on synthetic footage predicting
  real-footage performance), not just training data.

A structural trick to cut total render cost roughly 10x: split output into **Track A**
(independent randomized training stills, no temporal continuity needed, ~10,000 of them)
and **Track B** (short curated 25fps validation clips of specific events -- a play, a
sweep, a shuffle -- only 2-5 minutes total needed). Track B is also the strongest concrete
argument for building this at all: it can manufacture on demand, with perfect labels, the
exact clean single-tile play event that two independent real-footage searches this session
initially failed to find.

### Hand/manipulation fidelity: re-examined against real footage, and the first answer was wrong

An initial pass recommended skipping dexterous hand simulation entirely (static canned
poses on a kinematic proxy), reasoning that nothing in the perception pipeline tracks
finger-level detail. **Re-examined directly against real footage per explicit pushback,
and the initial premise didn't survive contact with the evidence:** hand-tile contact is
not brief or occasional -- it's present in 94-100% of sampled frames during play and
shuffle phases across both videos, with contact-free gaps typically under a second.
Occlusion is not a simple whole-hand blob either: real footage shows tiles visible through
finger gaps, forearms (not just hands) occluding large tile groups for extended periods,
and occlusion patterns shifting meaningfully frame-to-frame (roughly every ~100ms) even
while the wrist stays stationary -- a static canned pose held for a multi-second
interaction has no analogue in the real data.

However, the *reason* a fuller solution is needed turned out to be narrower and cheaper
than full dexterous-grasping physics: the hand pose space itself is small (players' hands
stay flat/palm-down throughout; nobody ever fans or lifts tiles to eye level, unlike a card
game), and the real complexity lives in two specific places: (1) tiles genuinely need real
rigid-body physics driven by finger/fingertip contact -- flicked plays that land mid-air
and settle, multi-tile groups dragged and re-shuffled relative to each other under one
hand's fingertips -- which a scripted "kinematic proxy carries one tile A-to-B" model
cannot produce; (2) the forearm is a major, previously-unmodeled occluder, comparable in
frequency to the hand itself.

**Revised recommendation:** an articulated hand-plus-forearm rig (not a floating
hand-only proxy), with tiles as real rigid bodies pushed by fingertip/palm colliders
(contact-driven, not attach/release teleportation), animated from a small (~8-clip)
keyframed gesture library covering the actual observed repertoire (idle rest, forearm
transit, rack pinch-pick-carry-release, trick gather/drag/square-up, rack sort, draw,
flat-palm-on-pile, point/gesture) plus per-finger procedural noise on top to multiply that
small library into realistic variety at low authoring cost. Full dexterous grasp-solving
physics is still not needed -- the motions are stereotyped and the hand stays planar, so
this is an animation-plus-contact-physics problem, not a robotics-manipulation-policy
problem. The one place low fidelity is explicitly fine to accept: the pre-game shuffle,
which the perception pipeline already treats as an opaque event to reject rather than
parse, so any plausible chaos suffices there.

### Interaction API and sim/real boundary: batch producer, not a live Gym-style environment

Given there's no learning agent inside the simulation and no reward signal, an
OpenAI-Gym-style `reset()/step()` API is the wrong shape -- its three load-bearing concepts
(agent, action, reward) map onto nothing here. **Recommendation: a headless, seeded, batch
generator** that writes rendered frames plus a ground-truth JSON sidecar (per-tile
identity/pose/visible-fraction/occluder, per-hand landmarks, full game state) plus
YOLO-seg-format labels per session -- matching what `tools/gen_synthetic_tiles.py` and
`tools/train_tile_segmenter.py` already consume, so day-one output needs no new plumbing.
Build it around a small internal `SimSession` object (`reset/advance/render/truth`) so a
live interactive API or a Gym wrapper could be added later cheaply if a real need for one
appears, without needing it now.

**On keeping the perception pipeline honest across both real and simulated input** (the
specific concern that motivated this research thread): `tools/live_view.py`'s existing
`Camera` class interface is already narrow and hardware-agnostic (4 methods, no
real-hardware assumptions leak into `Session.on_frame`) -- a simulated frame source could
implement the same interface as a drop-in with no downstream changes, which is good news.
Two real problems already exist, though, and need fixing regardless of which simulation
path is chosen: (1) `synth_data.py`'s synthetic tile colors were chosen specifically to
pass the real detector's HSV thresholds -- this makes "the classifier works on synthetic
data" circular today, not a real validation signal; (2) synthetic frames are generated
already in rectified-plane space, skipping the raw-frame/rectification/hand-detection front
end entirely, so that whole stage of the pipeline currently has zero synthetic test
coverage. Going forward: a simulated camera's output should go through the *same*
calibration procedure a real camera requires (never hand the pipeline a perfect known
homography for free -- this is a recognized anti-pattern, sometimes called ground-truth/
calibration leakage), the renderer should deliberately replicate real-camera imperfections
(lens distortion, sensor noise, compression artifacts -- a domain-randomization finding,
not just an aesthetic one) rather than emit an idealized clean image, and real-footage
accuracy should be tracked as a permanent, separate, non-optional number from
synthetic-footage accuracy so a widening gap between them is caught rather than averaged
away.

### Driving realistic gameplay: the existing engine/ code does most of this for free

`eye42/engine/trick.py`'s `legal_plays()` and `eye42/engine/probability.py`'s existing
heuristic playout policy (already written and tested, originally for Monte-Carlo bid
estimation) can drive simulated players directly -- a thin "director" (~200-350 lines)
wrapping them to own the deck, run bidding, and emit the project's existing `events.py`
dataclasses is enough for realistic, legal Texas 42 gameplay, no new game AI needed. Play
*strength* doesn't matter for what the simulation needs to produce; what matters is
physical/behavioral variety (tempo, placement sloppiness, which hand a player uses, sweep
style) and the ability to deliberately manufacture rare cases on demand with perfect labels
(a renege, a concession vs. redeal boundary, a splash/plunge hand) that real footage can't
be relied on to ever produce.

### Open decisions this research surfaced, not resolved here

- Blender/PyBullet (best training data) vs. Unity (interactive/VR capable) -- build one,
  the other, or sequence both -- is a real fork requiring a decision. **Resolved for now:**
  the user chose "best possible training data that can simulate and replicate live play
  with high fidelity" -- the Blender/physics-engine path. Phase 0 of that (see below) is
  built; Unity/interactive-VR remains a possible future track, not started.
- Which fidelity tier to target first, and how much GPU rental spend that implies, given
  the user has decided to rent rather than buy hardware for now. Still open -- Phase 0
  below is CPU-only and doesn't need an answer to this yet.
- Whether to pursue the real-camera depth-sensor option (~$140-250) as a separate,
  independent track alongside the simulation work. Still open.

## Standalone 3D simulation, Phase 0: built and working

Implemented as `src/eye42/simgen/` -- see ROADMAP.md's "Standalone 3D simulation" section
for what's built, what's deferred, and known rough edges. Two things worth recording here
because they update earlier research in this same document, not just "what got built":

**PyBullet, the plan's original physics engine choice, could not actually be installed** --
no prebuilt wheel exists for this platform/Python version, and building from source needs
a C++ toolchain not worth installing for this. **MuJoCo >=3.12 (already named in this
document as an acceptable alternative) was used instead**, and worked well: real Windows
wheels, no instability with many thin tiles in contact once the push-force controller was
correctly tuned (an early version pushed all four trick tiles to the exact same point with
a force stronger than table friction could safely dissipate, producing a genuine MuJoCo
QACC-instability warning -- fixed by jittering trick/pile target points and switching to a
closed-loop, velocity-capped `slide_toward` control instead of a blind constant-force push).

**The BlenderProc integration risk flagged by plan review did not materialize the way
expected, and turned out better than planned.** Plain `bpy` (pip-installable, ~340MB)
imports and runs a full headless CPU Cycles render directly from a normal Python process
in under a second for a small scene -- no `blenderproc run` CLI wrapper, no re-executing
inside a separate Blender-managed interpreter needed at all. The `blenderproc` package
itself was never installed or needed; `render_photoreal` in `simgen/render.py` drives
`bpy` directly. This meaningfully de-risks the whole initiative: photoreal rendering is
available today, in-process, CPU-only, at a few seconds per frame for a modest scene --
not blocked on GPU rental or a separate render-farm architecture the way earlier research
worried it might be.
