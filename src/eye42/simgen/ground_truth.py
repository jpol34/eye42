"""Writes per-frame ground truth to disk: YOLO-seg format (matching what
`tools/gen_synthetic_tiles.py` already produces, so `tools/train_tile_segmenter.py` needs
no changes to consume this data source) plus a richer JSON sidecar (3D pose, visible
fraction, game state) for future use -- validating occlusion reasoning, seat attribution,
event-stream diffing against a real perception run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from eye42.simgen.render import Camera, TileRenderInfo, render_ground_truth
from eye42.simgen.trajectory import FrameSnapshot


def _polygon_from_mask(mask: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Same approach as tools/gen_synthetic_tiles.py's ``_polygon_from_mask`` -- reused,
    not reinvented, just operating on render.py's occlusion-correct masks instead of a
    2D compositor's masks."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    h, w = mask.shape
    return [(float(x) / w, float(y) / h) for x, y in largest.reshape(-1, 2)]


def yolo_seg_lines(infos: Sequence[TileRenderInfo]) -> List[str]:
    """One YOLO-seg label line ("0 x1 y1 x2 y2 ...", class 0 = "tile") per tile with any
    visible pixels -- a fully-occluded tile (visible_fraction == 0) contributes nothing,
    matching how a real, undetectable tile would have no label either."""
    lines = []
    for info in infos:
        if info.visible_fraction <= 0.0:
            continue
        polygon = _polygon_from_mask(info.visible_mask)
        if polygon is None:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
        lines.append(f"0 {coords}")
    return lines


def _tile_identity(tile) -> str:
    return f"{tile.high}-{tile.low}"


def frame_ground_truth_dict(frame: FrameSnapshot, infos: Sequence[TileRenderInfo]) -> dict:
    """The richer per-frame JSON record -- position/orientation/visible-fraction per
    tile, plus which tiles remain in each seat's rack, for uses a bare YOLO label can't
    support (occlusion-reasoning validation, seat-attribution checks, event-stream
    diffing against a real perception run)."""
    position_by_tile = {ts.tile: ts.position for ts in frame.tile_states}
    orientation_by_tile = {ts.tile: ts.orientation_quat for ts in frame.tile_states}
    return {
        "frame": frame.index,
        "tiles": [
            {
                "identity": _tile_identity(info.tile),
                "position_m": position_by_tile[info.tile],
                "orientation_quat": orientation_by_tile[info.tile],
                "visible_fraction": info.visible_fraction,
            }
            for info in infos
        ],
        "seat_racks": {str(seat): [_tile_identity(t) for t in tiles] for seat, tiles in frame.seat_racks.items()},
        "occluders": [
            {"name": part.name, "center_m": part.center_m, "orientation_quat": part.orientation_quat}
            for part in frame.occluders
        ],
    }


def write_frame_truth_jsonl(frames: Sequence[FrameSnapshot], camera: Camera, path: Path) -> None:
    """Writes one JSON object per line, one line per frame -- see
    ``frame_ground_truth_dict`` for the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for frame in frames:
            infos = render_ground_truth(frame.tile_states, camera, frame.occluders)
            handle.write(json.dumps(frame_ground_truth_dict(frame, infos)) + "\n")
