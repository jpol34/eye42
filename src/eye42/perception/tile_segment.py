"""Trained-model inference for splitting touching-tile clusters.

TileLocalizer's contour-based find_tiles() can't split tiles that touch
edge-to-edge (see RESEARCH.md). This module wraps a YOLOv8-seg model
(trained via tools/train_tile_segmenter.py on synthetic data from
eye42.perception.synth_data, then exported to ONNX) for use on a crop
find_tiles() flags as an oversized/ambiguous cluster rather than a confident
single tile.

Output shape follows ultralytics' standard single-class YOLOv8-seg ONNX
export: output0 is (1, 4 + num_classes + 32, num_anchors) -- box (cx, cy, w,
h), per-class confidence, then 32 mask coefficients; output1 (proto) is
(1, 32, 160, 160). This project trains exactly one class ("tile"), so
num_classes is always 1 here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from eye42.perception.tile_detect import TileRegion

_INPUT_SIZE = 640  # must match the imgsz tools/train_tile_segmenter.py exports at
_CONFIDENCE_THRESHOLD = 0.4
_NMS_IOU_THRESHOLD = 0.5
_MASK_THRESHOLD = 0.5
_PROTO_SIZE = 160  # ultralytics' standard proto-mask resolution for a 640 input


def _letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def _preprocess(crop_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    padded, scale, pad = _letterbox(crop_bgr, _INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    return blob, scale, pad


def _undo_letterbox(x: float, y: float, scale: float, pad: Tuple[int, int]) -> Tuple[float, float]:
    pad_x, pad_y = pad
    return (x - pad_x) / scale, (y - pad_y) / scale


def _cxcywh_to_xywh_topleft(boxes_cxcywh: np.ndarray) -> np.ndarray:
    """cv2.dnn.NMSBoxes expects (x, y, width, height) with (x, y) the top-left corner --
    NOT (x1, y1, x2, y2). Passing an (x1, y1, x2, y2) array directly is silently
    misread as (x, y, width, height), corrupting every IoU computation, so boxes
    must be converted through this function first."""
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    return np.column_stack([x1, y1, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]])


def _postprocess(
    outputs: List[np.ndarray],
    scale: float,
    pad: Tuple[int, int],
    crop_size: Tuple[int, int],
) -> List[TileRegion]:
    output0, proto = outputs[0][0], outputs[1][0]  # (37, N), (32, 160, 160)
    predictions = output0.T  # (N, 37): [cx, cy, w, h, conf, 32 mask coefs]

    confidences = predictions[:, 4]
    keep = confidences >= _CONFIDENCE_THRESHOLD
    predictions = predictions[keep]
    confidences = confidences[keep]
    if len(predictions) == 0:
        return []

    boxes_cxcywh = predictions[:, :4]
    nms_indices = cv2.dnn.NMSBoxes(
        _cxcywh_to_xywh_topleft(boxes_cxcywh).tolist(), confidences.tolist(),
        _CONFIDENCE_THRESHOLD, _NMS_IOU_THRESHOLD,
    )
    if len(nms_indices) == 0:
        return []
    nms_indices = np.array(nms_indices).flatten()

    crop_w, crop_h = crop_size
    regions: List[TileRegion] = []
    for i in nms_indices:
        mask_coefs = predictions[i, 5:]
        proto_flat = proto.reshape(32, -1)
        mask = 1.0 / (1.0 + np.exp(-(mask_coefs @ proto_flat)))
        mask = mask.reshape(_PROTO_SIZE, _PROTO_SIZE)
        mask = cv2.resize(mask, (_INPUT_SIZE, _INPUT_SIZE))
        mask_binary = (mask > _MASK_THRESHOLD).astype(np.uint8)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), (w, h), angle = cv2.minAreaRect(largest)
        x, y, bw, bh = cv2.boundingRect(largest)

        cx, cy = _undo_letterbox(cx, cy, scale, pad)
        x, y = _undo_letterbox(x, y, scale, pad)
        w, h, bw, bh = w / scale, h / scale, bw / scale, bh / scale

        regions.append(TileRegion(
            x=int(max(0, x)), y=int(max(0, y)), width=int(min(bw, crop_w)), height=int(min(bh, crop_h)),
            center=(cx, cy), angle=angle, rect_size=(w, h),
        ))
    return regions


class TileSegmenter:
    """Wraps an ONNX-exported YOLOv8-seg model. Output is structurally
    compatible with TileLocalizer.find_tiles() (a List[TileRegion]), so a
    caller can compose them: contour-based find_tiles() for a confident
    single tile, this for an ambiguous cluster's own crop."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def find_tiles(self, crop_bgr: np.ndarray) -> List[TileRegion]:
        """Runs the segmenter on crop_bgr (typically an ambiguous cluster's
        own bounding-box crop, not a full frame) and returns one TileRegion
        per detected tile instance, in crop_bgr's own pixel coordinates."""
        h, w = crop_bgr.shape[:2]
        blob, scale, pad = _preprocess(crop_bgr)
        outputs = self._session.run(None, {self._input_name: blob})
        return _postprocess(outputs, scale, pad, (w, h))
