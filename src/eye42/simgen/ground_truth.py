"""Writes per-frame ground truth to disk: YOLO-seg format (the same "0 x1 y1 x2 y2 ..."
line shape `tools/gen_synthetic_tiles.py` produces, so `tools/train_tile_segmenter.py`
needs no changes to consume this data source -- but NOT the same label semantics: this
module emits one line per disjoint visible fragment of a tile, since the hand+forearm
occluder rig routinely splits a tile's visible region into separate pieces, while
gen_synthetic_tiles.py's own compositor never produces disjoint masks in the first place
(its per-tile mask is an unsubtracted solid paste region, always one contiguous blob --
see synth_data.py's SyntheticTileInstance docstring) and so still emits exactly one line
per tile) plus a richer JSON sidecar (3D pose, visible fraction, game state) for future
use -- validating occlusion reasoning, seat attribution, event-stream diffing against a
real perception run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from eye42.simgen.render import Camera, TileRenderInfo, render_ground_truth
from eye42.simgen.trajectory import FrameSnapshot


def _polygons_from_mask(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    """One polygon per disjoint visible fragment of ``mask`` (not just the largest) --
    an occluder crossing a tile's middle can split its visible pixels into separate
    pieces, and a real, still-visible piece shouldn't be silently dropped just because
    it's smaller than another piece. Matches ultralytics' own mask-to-YOLO-label
    converter's convention (one line per disjoint contour, same class), not a novel
    scheme. RETR_EXTERNAL still won't split out a true enclosed hole (an occluder
    entirely inside a tile's silhouette, touching no edge) -- a separate, pre-existing
    limitation this doesn't address, not worsened by it either."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        polygons.append([(float(x) / w, float(y) / h) for x, y in contour.reshape(-1, 2)])
    return polygons


def yolo_seg_lines(infos: Sequence[TileRenderInfo]) -> List[str]:
    """One YOLO-seg label line ("0 x1 y1 x2 y2 ...", class 0 = "tile") per disjoint
    visible fragment of a tile -- a fully-occluded tile (visible_fraction == 0), or a
    fragment too small to matter, contributes nothing, matching how a real,
    undetectable-or-negligible tile region would have no label either."""
    lines = []
    for info in infos:
        if info.visible_fraction <= 0.0:
            continue
        for polygon in _polygons_from_mask(info.visible_mask):
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
