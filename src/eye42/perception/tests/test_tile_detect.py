from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eye42.engine.tiles import Tile
from eye42.perception.synth_data import BODY_COLOR as _BODY_COLOR
from eye42.perception.synth_data import draw_tile_crop as _make_tile_crop
from eye42.perception.synth_data import draw_tile_half as _make_half
from eye42.perception.tile_detect import (
    _SWEEP_SUSTAINED_FRAMES,
    EventSegmenter,
    HandBoundaryDetector,
    OpenCVTileClassifier,
    TableRectifier,
    TileLocalizer,
    TileObservation,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# TableRectifier
# ---------------------------------------------------------------------------

def test_rectifier_produces_requested_output_size():
    corners = [(10, 10), (110, 10), (110, 110), (10, 110)]
    rectifier = TableRectifier(corners, (100, 100))
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    rectified = rectifier.rectify(frame)
    assert rectified.shape[:2] == (100, 100)


def test_project_points_round_trips_a_known_rectangle():
    corners = [(10, 10), (110, 10), (110, 110), (10, 110)]
    rectifier = TableRectifier(corners, (100, 100))

    projected = rectifier.project_points(corners)

    expected = [(0, 0), (99, 0), (99, 99), (0, 99)]
    for (px, py), (ex, ey) in zip(projected, expected):
        assert abs(px - ex) < 1e-3 and abs(py - ey) < 1e-3


# ---------------------------------------------------------------------------
# TileLocalizer
# ---------------------------------------------------------------------------

def test_localizer_finds_a_rotated_green_tile_on_a_white_table():
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    box = cv2.boxPoints(((150.0, 150.0), (60.0, 30.0), 25.0)).astype(np.int32)
    cv2.fillPoly(frame, [box], _BODY_COLOR)

    regions = TileLocalizer().find_tiles(frame)

    assert len(regions) == 1
    cx, cy = regions[0].center
    assert abs(cx - 150) < 5
    assert abs(cy - 150) < 5


def test_localizer_ignores_a_non_tile_shaped_green_blob():
    """A roughly square blob (a green mug, say) shouldn't pass the domino
    aspect-ratio filter the way an actual tile-shaped region does."""
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 40, _BODY_COLOR, -1)

    regions = TileLocalizer().find_tiles(frame)

    assert regions == []


def test_localizer_on_real_footage_still_finds_isolated_tiles_around_a_cluster():
    """``fixtures/real_footage_touching_cluster.jpg`` is a rectified crop from
    real game footage: several isolated tiles around a knot of 3-4 tiles
    touching edge-to-edge (a boneyard-pile-like cluster). Isolated tiles must
    still be found correctly even with a touching cluster elsewhere in frame."""
    crop = cv2.imread(str(_FIXTURES_DIR / "real_footage_touching_cluster.jpg"))
    regions = TileLocalizer().find_tiles(crop)

    isolated_tile_center = (434.5, 425.0)
    match = min(regions, key=lambda r: np.hypot(r.center[0] - isolated_tile_center[0], r.center[1] - isolated_tile_center[1]))
    assert np.hypot(match.center[0] - isolated_tile_center[0], match.center[1] - isolated_tile_center[1]) < 5


def test_localizer_on_real_footage_cannot_yet_separate_a_touching_cluster():
    """Same fixture as above. Documents a known limitation (see ROADMAP): a
    knot of 3+ tiles touching edge-to-edge merges into one contour that either
    fails the area/aspect filters and vanishes entirely, or -- as with the two
    touching tiles here -- passes the filters as one bogus double-wide region.
    Neither failure mode is a per-tile detection, so this asserts today's
    actual behavior rather than the eventually-correct one, to catch
    regressions until TileLocalizer can split touching clusters."""
    crop = cv2.imread(str(_FIXTURES_DIR / "real_footage_touching_cluster.jpg"))
    regions = TileLocalizer().find_tiles(crop)

    # The 3-tile touching knot sits roughly in this box; no region's center
    # should fall inside it -- it produces no detections at all.
    untouched_cluster_box = (40, 200, 300, 350)  # x0, y0, x1, y1
    x0, y0, x1, y1 = untouched_cluster_box
    assert not any(x0 <= r.center[0] <= x1 and y0 <= r.center[1] <= y1 for r in regions)

    # The two touching tiles near (336, 147) collapse into one region whose
    # long side is far past a single tile's -- a bogus merged detection, not
    # two correct ones.
    merged = min(regions, key=lambda r: np.hypot(r.center[0] - 336, r.center[1] - 147))
    assert max(merged.rect_size) > 200


def test_localizer_ignores_skin_tone_on_real_footage_with_hands_in_frame():
    """``fixtures/real_footage_hands_in_frame.jpg`` is a rectified crop from
    real game footage with two players' hands/fingers occupying most of the
    frame, alongside one genuine tile. Skin tone falls well outside the
    tile-green HSV band in theory; this confirms it in practice rather than
    leaving the claim unverified against a real camera/lighting setup."""
    crop = cv2.imread(str(_FIXTURES_DIR / "real_footage_hands_in_frame.jpg"))
    regions = TileLocalizer().find_tiles(crop)

    assert len(regions) == 1
    cx, cy = regions[0].center
    assert abs(cx - 590) < 5 and abs(cy - 273) < 5


# ---------------------------------------------------------------------------
# OpenCVTileClassifier (pip counting)
# ---------------------------------------------------------------------------

def test_classifier_reads_pip_counts_from_a_synthetic_crop():
    six_pips = [(30, 30), (30, 60), (30, 90), (90, 30), (90, 60), (90, 90)]
    crop = _make_tile_crop(six_pips, [])

    result = OpenCVTileClassifier().classify(crop)

    assert result
    tile, confidence = result[0]
    assert tile == Tile.of(6, 0)
    assert confidence > 0


def test_classifier_reads_a_blank_half_as_zero_not_a_failed_detection():
    crop = _make_tile_crop([(30, 30), (30, 90), (90, 30)], [])

    tile, _ = OpenCVTileClassifier().classify(crop)[0]

    assert tile == Tile.of(3, 0)


def test_classifier_does_not_count_a_dark_non_white_pin_as_a_pip():
    """The known Hough-circle failure mode this project explicitly avoids:
    a spinner-pin-like blob that's the wrong color must not read as a pip."""
    top = _make_half([(30, 30), (30, 90), (90, 30)])
    cv2.circle(top, (60, 60), 6, (20, 20, 20), -1)  # dark pin, dead center
    bottom = _make_half([])
    crop = np.vstack([top, bottom])

    tile, _ = OpenCVTileClassifier().classify(crop)[0]

    assert tile == Tile.of(3, 0)


# ---------------------------------------------------------------------------
# EventSegmenter -- settle-time debounce
# ---------------------------------------------------------------------------

def _settle_on_empty_table(segmenter, frame):
    """Fast-forwards a fresh (or just-reset) segmenter past its baseline-
    capture phase with nothing on the table, so the calls that follow
    exercise ordinary play debounce rather than baseline capture."""
    for _ in range(segmenter.settle_frames):
        segmenter.feed(frame, [])


def test_segmenter_only_confirms_a_play_after_settle_frames():
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    observation = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    assert segmenter.feed(frame, observation) == []
    assert segmenter.feed(frame, observation) == []
    played = segmenter.feed(frame, observation)

    assert len(played) == 1
    assert played[0].tile == Tile.of(4, 2)


def test_segmenter_resolves_a_settling_tile_by_plurality_vote_not_last_frame():
    """A single misclassified frame landing on exactly the confirming frame must
    not corrupt the reported identity -- the majority of the settle window's
    frames agreeing on the real tile must win instead."""
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    real_tile = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]
    misread = [TileObservation(tile=Tile.of(1, 1), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    segmenter.feed(frame, real_tile)
    segmenter.feed(frame, real_tile)
    played = segmenter.feed(frame, misread)  # the confirming frame is the noisy one

    assert len(played) == 1
    assert played[0].tile == Tile.of(4, 2)  # majority (2 of 3 frames), not the last frame


def test_segmenter_reports_the_mean_confidence_of_the_winning_tiles_votes():
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    confidences = [1.0, 0.6, 0.4]  # deliberately distinct from their mean, so this test
    # can't pass under the old last-frame-wins behavior (which would report 0.4) by
    # coincidence -- it must actually exercise the mean-of-votes computation.

    played: list = []
    for conf in confidences:
        obs = [TileObservation(tile=Tile.of(4, 2), confidence=conf, position=(10.0, 10.0), frame_index=0)]
        played += segmenter.feed(frame, obs)

    assert len(played) == 1
    assert played[0].confidence == pytest.approx(sum(confidences) / len(confidences))


def test_segmenter_drops_a_candidate_that_vanishes_before_settling():
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    observation = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    segmenter.feed(frame, observation)
    segmenter.feed(frame, [])  # tile briefly not observed -- candidate dropped
    played = segmenter.feed(frame, observation)

    assert played == []  # a fresh candidate again, not yet settled


def test_segmenter_reset_forgets_confirmed_positions():
    segmenter = EventSegmenter(settle_frames=2)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    observation = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    segmenter.feed(frame, observation)
    assert len(segmenter.feed(frame, observation)) == 1  # settled
    assert segmenter.feed(frame, observation) == []  # already confirmed at this position -- ignored

    segmenter.reset()
    _settle_on_empty_table(segmenter, frame)
    segmenter.feed(frame, observation)
    played = segmenter.feed(frame, observation)
    assert len(played) == 1  # a new hand can reuse the same table position


def test_segmenter_never_reports_a_tile_present_since_before_settling():
    """The exact bug found against live footage: a tile sitting in a
    player's hand from the moment a new hand starts must never be reported
    as a play, no matter how long it then sits stable on the table."""
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    hand_tile = [TileObservation(tile=Tile.of(5, 3), confidence=1.0, position=(40.0, 40.0), frame_index=0)]

    played: list = []
    for _ in range(10):
        played += segmenter.feed(frame, hand_tile)

    assert played == []


def test_segmenter_reports_a_genuinely_new_position_after_baseline_settles():
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    hand_tile = [TileObservation(tile=Tile.of(5, 3), confidence=1.0, position=(40.0, 40.0), frame_index=0)]
    for _ in range(3):
        segmenter.feed(frame, hand_tile)  # baseline settles with the hand tile present

    new_play = TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)
    played: list = []
    for _ in range(3):
        played += segmenter.feed(frame, hand_tile + [new_play])

    assert len(played) == 1
    assert played[0].tile == Tile.of(4, 2)


def _sweep_frames(shape=(50, 50, 3)):
    """Two frames whose pixel difference exceeds _sweep_in_progress's 15%-of-
    table motion threshold -- alternating between them drives sustained
    "sweeping" for as many feed() calls as needed."""
    quiet = np.zeros(shape, dtype=np.uint8)
    busy = np.zeros(shape, dtype=np.uint8)
    busy[: shape[0] // 2, :, :] = 255
    return quiet, busy


def test_segmenter_reuses_a_position_for_a_later_trick_after_a_sustained_sweep():
    """A later trick's play landing at the exact same table position an
    earlier trick's play did (players can toss a trick anywhere on the
    table, not one fixed zone) must still be reported, once the sweep
    between the two tricks has been sustained long enough to prune the
    earlier position out of _confirmed_positions."""
    segmenter = EventSegmenter(settle_frames=3)
    quiet, busy = _sweep_frames()
    _settle_on_empty_table(segmenter, quiet)
    trick_one = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    for _ in range(3):
        played = segmenter.feed(quiet, trick_one)
    assert len(played) == 1  # trick one's play confirmed

    for i in range(_SWEEP_SUSTAINED_FRAMES):
        segmenter.feed(busy if i % 2 == 0 else quiet, [])  # sweep clears the trick area

    trick_two = [TileObservation(tile=Tile.of(6, 6), confidence=1.0, position=(10.0, 10.0), frame_index=0)]
    played = []
    for _ in range(3):
        played += segmenter.feed(quiet, trick_two)

    assert len(played) == 1
    assert played[0].tile == Tile.of(6, 6)


def test_segmenter_never_confirms_a_play_while_a_sweep_is_in_progress():
    """A sustained sweep resets _confirmed_positions (see the earlier reuse
    test), which un-confirms an already-played tile that's still physically
    sitting on the table a few frames from being picked up. If that
    still-present tile were allowed to cross the settle threshold mid-sweep,
    it would get re-reported as a brand-new play of the tile it already
    was. No play may ever be confirmed while sweeping is True, however long
    the sweep runs."""
    segmenter = EventSegmenter(settle_frames=3)
    quiet, busy = _sweep_frames()
    _settle_on_empty_table(segmenter, quiet)
    trick_one = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    for _ in range(3):
        played = segmenter.feed(quiet, trick_one)
    assert len(played) == 1  # trick one's play confirmed

    # The tile never actually leaves (unlike the reuse test) -- it's still
    # sitting at the same position while a sustained sweep (elsewhere, or
    # falsely triggered) resets _confirmed_positions and keeps running well
    # past the settle threshold.
    played = []
    for i in range(_SWEEP_SUSTAINED_FRAMES + 10):
        played += segmenter.feed(busy if i % 2 == 0 else quiet, trick_one)

    assert played == []


def test_segmenter_still_dedupes_same_position_without_a_sweep_between_plays():
    """Without a sustained sweep in between, a repeat observation at an
    already-confirmed position must stay suppressed -- the sweep-triggered
    prune must not weaken ordinary same-hand dedup."""
    segmenter = EventSegmenter(settle_frames=3)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    _settle_on_empty_table(segmenter, frame)
    observation = [TileObservation(tile=Tile.of(4, 2), confidence=1.0, position=(10.0, 10.0), frame_index=0)]

    for _ in range(3):
        played = segmenter.feed(frame, observation)
    assert len(played) == 1

    assert segmenter.feed(frame, observation) == []  # no sweep occurred -- still deduped


# ---------------------------------------------------------------------------
# HandBoundaryDetector -- edge-triggered reshuffle detection
# ---------------------------------------------------------------------------

def _motion_mask(fraction: float, shape=(50, 50)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[: int(shape[0] * fraction), :] = 1
    return mask


def test_hand_boundary_fires_once_for_a_sustained_shuffle_not_every_frame():
    detector = HandBoundaryDetector()
    heavy_motion = _motion_mask(0.9)  # well above the detector's own threshold

    results = [detector.is_new_hand_starting(heavy_motion) for _ in range(30)]

    assert results.count(True) == 1  # fires once despite motion staying high for many frames


def test_hand_boundary_can_fire_again_after_motion_subsides():
    detector = HandBoundaryDetector()
    heavy_motion = _motion_mask(0.9)
    no_motion = _motion_mask(0.0)

    for _ in range(25):
        detector.is_new_hand_starting(heavy_motion)  # first shuffle: fires once, already covered above
    for _ in range(5):
        detector.is_new_hand_starting(no_motion)  # shuffle ends
    results = [detector.is_new_hand_starting(heavy_motion) for _ in range(25)]  # a second shuffle

    assert results.count(True) == 1
