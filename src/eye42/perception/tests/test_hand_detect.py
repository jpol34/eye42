from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eye42.engine.tiles import Tile
from eye42.perception.hand_detect import HandGate, HandRegion
from eye42.perception.tile_detect import TableRectifier, TileObservation

mediapipe = pytest.importorskip("mediapipe")

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _FakeDetector:
    """Test double for HandDetector -- returns a fixed set of regions
    instead of running real MediaPipe inference, so tests using it are fast,
    deterministic, and don't need the optional mediapipe dependency."""

    def __init__(self, regions):
        self._regions = regions
        self.call_count = 0

    def detect(self, frame_bgr):
        self.call_count += 1
        return self._regions


# ---------------------------------------------------------------------------
# HandDetector (real MediaPipe inference)
# ---------------------------------------------------------------------------

def test_hand_detector_finds_hands_in_real_footage():
    from eye42.perception.hand_detect import HandDetector

    frame = cv2.imread(str(_FIXTURES_DIR / "real_footage_raw_frame_with_hands.jpg"))
    assert frame is not None

    regions = HandDetector().detect(frame)

    assert len(regions) >= 2
    assert all(r.confidence > 0.5 for r in regions)


# ---------------------------------------------------------------------------
# HandGate -- suppression logic (fake detector, no real inference)
# ---------------------------------------------------------------------------

def test_hand_gate_suppresses_observations_under_a_hand():
    """This is the test that actually proves suppression works -- the real
    footage fixture's hand blobs are already excluded today by
    TileLocalizer's own area cap alone, so it can't demonstrate a
    hand-caused false positive slipping through; this synthetic case can."""
    rectifier = TableRectifier([(0, 0), (100, 0), (100, 100), (0, 100)], (100, 100))
    hand = HandRegion(x0=10, y0=10, x1=30, y1=30, confidence=0.9)
    gate = HandGate(_FakeDetector([hand]), rectifier)

    under_hand = TileObservation(tile=Tile.of(1, 1), confidence=1.0, position=(20.0, 20.0), frame_index=0)
    far_away = TileObservation(tile=Tile.of(2, 2), confidence=1.0, position=(80.0, 80.0), frame_index=0)

    survivors = gate.filter(None, [under_hand, far_away], None)

    assert survivors == [far_away]
    assert gate.last_suppressed == [under_hand]


def test_hand_gate_does_not_suppress_a_genuine_tile_far_from_a_hand():
    rectifier = TableRectifier([(0, 0), (100, 0), (100, 100), (0, 100)], (100, 100))
    hand = HandRegion(x0=10, y0=10, x1=20, y1=20, confidence=0.9)
    gate = HandGate(_FakeDetector([hand]), rectifier)

    far_tile = TileObservation(tile=Tile.of(3, 3), confidence=1.0, position=(90.0, 90.0), frame_index=0)

    survivors = gate.filter(None, [far_tile], None)

    assert survivors == [far_tile]
    assert gate.last_suppressed == []


def test_hand_gate_skips_detection_when_no_motion():
    rectifier = TableRectifier([(0, 0), (100, 0), (100, 100), (0, 100)], (100, 100))
    detector = _FakeDetector([])
    gate = HandGate(detector, rectifier)
    zero_motion = np.zeros((10, 10), dtype=np.uint8)

    gate.filter(None, [], None)  # first call, motion_mask=None -- always detects
    assert detector.call_count == 1

    gate.filter(None, [], zero_motion)  # no motion -- should reuse, not re-detect
    assert detector.call_count == 1


import numpy as np  # noqa: E402 -- used only by the motion-gating test above
