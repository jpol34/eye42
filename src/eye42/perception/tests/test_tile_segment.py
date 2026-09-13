from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from eye42.perception.tile_segment import _cxcywh_to_xywh_topleft

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_MODEL_PATH = Path(__file__).resolve().parents[4] / "models" / "tile_segmenter.onnx"


def test_cxcywh_to_xywh_topleft_converts_center_format_to_corner_format():
    """cv2.dnn.NMSBoxes needs (x, y, width, height) with (x, y) the TOP-LEFT corner --
    a real bug here once passed (x1, y1, x2, y2) directly, which NMSBoxes silently
    misread as (x, y, width, height), corrupting every IoU computation it made. Pure
    function, no ONNX/model dependency -- runs regardless of whether a trained model
    exists."""
    boxes_cxcywh = np.array([[100.0, 100.0, 40.0, 20.0], [0.0, 0.0, 10.0, 10.0]])

    converted = _cxcywh_to_xywh_topleft(boxes_cxcywh)

    expected = np.array([[80.0, 90.0, 40.0, 20.0], [-5.0, -5.0, 10.0, 10.0]])
    assert np.allclose(converted, expected)


@pytest.mark.skipif(not _MODEL_PATH.exists(), reason=f"no trained model at {_MODEL_PATH} -- run tools/train_tile_segmenter.py first")
def test_segmenter_splits_the_real_touching_cluster_fixture():
    pytest.importorskip("onnxruntime")
    from eye42.perception.tile_segment import TileSegmenter

    crop = cv2.imread(str(_FIXTURES_DIR / "real_footage_touching_cluster.jpg"))
    segmenter = TileSegmenter(_MODEL_PATH)

    regions = segmenter.find_tiles(crop)

    # RESEARCH.md documents this fixture's genuine touching cluster as
    # 3-4 tiles, plus several already-isolated ones nearby -- a working
    # segmenter should report noticeably more than TileLocalizer's current
    # single merged/vanished reading of it.
    assert len(regions) >= 3
