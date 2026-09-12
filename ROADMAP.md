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
  Phase 2/4 storage: flat JSON per hand/game (SQLite later only if
  cross-game analytics are wanted). Phase 4's server: plain HTML/JS or a
  tiny local Flask/FastAPI process.
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
    footage exists — deliberately not built now.** The oracle mechanism
    removed earlier (`_check_trump_contradiction`) needed a lookup into a
    player's *remaining concealed hand* to tell "genuinely void in a suit"
    from "held a follower and revoked" — permanently impossible. But a
    different, purely logical mechanism needs no such lookup: if a player
    fails to follow led suit S under live hypothesis H (a candidate, not yet
    confirmed), and that same player later plays a tile that computes as
    suit S under H, that's a hard contradiction — they can't be void in S
    under H and also hold/play S under H, so H must be wrong. No folklore
    weighting needed, just the same publicly-observed play stream every
    other mechanism here already sees. `TrumpHypothesisTracker.
    observe_void` is already a documented no-op stub for exactly this;
    `observe_contradiction` is already a fully reusable, hand-agnostic
    primitive (verified: its own implementation only touches `self.weights`,
    no hand-knowledge coupling survives in it after the oracle-removal
    pass). What's missing is per-hypothesis void tracking (a player can be
    void in suit S under candidate H1 but not H2, since suit membership
    depends on trump) — real state-machine complexity, not a trivial wire-up:
    it has to interact correctly with `observe_contradiction`'s existing
    weight-reopening fallback (`_last_weights_before_empty`) — when a
    candidate's weight collapses and later gets restored from a snapshot,
    decide whether its void state restores alongside it or resets — exactly
    the class of edge that broke in subtle ways during this project's
    trump-hypothesis reversibility work. Needs its own design pass and
    plan-critic review before building, not a bundled addition to anything
    else.
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
  But `TilePlayed.confidence` is itself currently dead — every consumer of
  it was deleted in the oracle-removal pass, so there are zero reads of it
  anywhere in `engine/` today — and `TrumpCalled` (the event type) is
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

## Explicitly out of scope for now

Named so a future pass doesn't re-propose them without new information: a
real double-dummy solver; bid-plausibility importance weighting of sampled
deals in `probability.py`; any ML/MCTS/CFR-based play or bidding; antithetic
variates or exact-uniform-deal sampling via permanents for the Monte Carlo
estimator; calibration shrinkage, larger sample counts, or new tunable knobs
there; any display/UI work beyond what Phase 4 already scopes.
