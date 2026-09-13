"""Synthetic domino-tile rendering and scene compositing for training data.

TileLocalizer can't split tiles that touch edge-to-edge (see RESEARCH.md); a
small instance-segmentation model trained on synthetic scenes is the planned
fix. This module renders individual tiles (the same green-body/white-pip
appearance test_tile_detect.py's synthetic tests already use) and composites
them into multi-tile scenes -- some touching, some overlapping -- alongside
the per-tile ground truth (mask, center, angle, identity) a training
pipeline needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

BODY_COLOR = (0, 150, 0)  # pure green, BGR -- unambiguous against any reasonable HSV threshold
PIP_COLOR = (255, 255, 255)  # pure white

# Standard domino/die pip layout: a 3x3 grid of candidate positions per half.
_GRID_POSITIONS = {
    "TL": (0, 0), "TR": (2, 0), "ML": (0, 1), "MR": (2, 1), "BL": (0, 2), "BR": (2, 2), "C": (1, 1),
}
_PIP_LAYOUTS = {
    0: [],
    1: ["C"],
    2: ["TL", "BR"],
    3: ["TL", "C", "BR"],
    4: ["TL", "TR", "BL", "BR"],
    5: ["TL", "TR", "BL", "BR", "C"],
    6: ["TL", "TR", "ML", "MR", "BL", "BR"],
}

_DEFAULT_HALF_SIZE = (80, 60)  # (width, height) -- stacked crop aspect (120/80=1.5) sits inside TileLocalizer's 1.4-3.2 band


def draw_tile_half(pip_positions: Sequence[Tuple[int, int]], size: Tuple[int, int] = (120, 120)) -> np.ndarray:
    """Draws one tile half: a green rectangle with a white pip dot at each
    given pixel position. The shared low-level primitive both this module's
    canonical-layout renderer and test_tile_detect.py's hand-specified-
    position synthetic tests build on."""
    half = np.full((size[1], size[0], 3), BODY_COLOR, dtype=np.uint8)
    for cx, cy in pip_positions:
        cv2.circle(half, (cx, cy), 8, PIP_COLOR, -1)
    return half


def draw_tile_crop(
    pips_top: Sequence[Tuple[int, int]],
    pips_bottom: Sequence[Tuple[int, int]],
    size: Tuple[int, int] = (120, 120),
) -> np.ndarray:
    """Stacks two halves into one upright tile crop."""
    return np.vstack([draw_tile_half(pips_top, size), draw_tile_half(pips_bottom, size)])


def pip_layout_fractions(count: int) -> List[Tuple[float, float]]:
    """The standard 0-6 domino pip layout as (u, v) fractions in [0, 1] of a tile
    half's own box -- the single source of truth both this module's pixel-space
    renderer and eye42.simgen.render's 3D texture builder place dots from, so the
    two never drift apart into two different-looking pip layouts."""
    if count not in _PIP_LAYOUTS:
        raise ValueError(f"no canonical pip layout for {count} (expected 0-6)")
    us = [0.25, 0.5, 0.75]
    vs = [0.25, 0.5, 0.75]
    return [(us[gx], vs[gy]) for gx, gy in (_GRID_POSITIONS[name] for name in _PIP_LAYOUTS[count])]


def _canonical_pip_positions(count: int, size: Tuple[int, int]) -> List[Tuple[int, int]]:
    w, h = size
    return [(int(u * w), int(v * h)) for u, v in pip_layout_fractions(count)]


def render_tile(top: int, bottom: int, half_size: Tuple[int, int] = _DEFAULT_HALF_SIZE) -> np.ndarray:
    """Renders one upright tile crop using the standard 0-6 domino pip
    layout for each half, rather than caller-specified pip positions."""
    return draw_tile_crop(
        _canonical_pip_positions(top, half_size),
        _canonical_pip_positions(bottom, half_size),
        half_size,
    )


@dataclass(frozen=True)
class SyntheticTileInstance:
    """Ground truth for one placed tile in a generated scene. ``mask`` is a
    boolean array the same size as the scene, True where this tile's pixels
    were painted -- later tiles composited on top are not subtracted from an
    earlier tile's mask, so an occluded tile's recorded mask can include
    pixels a later tile actually covers. Acceptable for a first training set
    (real occlusion-aware masks are a refinement, not required to start
    training), but worth knowing before trusting mask-based metrics."""

    top: int
    bottom: int
    center: Tuple[int, int]
    angle: float
    mask: np.ndarray


def _paste_rotated(scene: np.ndarray, tile: np.ndarray, center: Tuple[int, int], angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotates tile by angle and alpha-composites it onto scene centered at
    center, clipped to scene bounds. Returns (updated scene, boolean mask of
    painted pixels)."""
    th, tw = tile.shape[:2]
    diag = int(np.ceil(np.hypot(th, tw))) + 2
    canvas = np.zeros((diag, diag, 3), dtype=np.uint8)
    alpha = np.zeros((diag, diag), dtype=np.uint8)
    y0, x0 = (diag - th) // 2, (diag - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = tile
    alpha[y0:y0 + th, x0:x0 + tw] = 255

    rot_mat = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, rot_mat, (diag, diag))
    rotated_alpha = cv2.warpAffine(alpha, rot_mat, (diag, diag))

    h, w = scene.shape[:2]
    cx, cy = center
    sx0, sy0 = cx - diag // 2, cy - diag // 2

    src_x0, src_y0 = max(0, -sx0), max(0, -sy0)
    dst_x0, dst_y0 = max(0, sx0), max(0, sy0)
    dst_x1, dst_y1 = min(w, sx0 + diag), min(h, sy0 + diag)
    src_x1, src_y1 = src_x0 + max(0, dst_x1 - dst_x0), src_y0 + max(0, dst_y1 - dst_y0)

    mask_bool = np.zeros((h, w), dtype=bool)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return scene, mask_bool

    region_alpha = rotated_alpha[src_y0:src_y1, src_x0:src_x1]
    region_tile = rotated[src_y0:src_y1, src_x0:src_x1]
    alpha_f = (region_alpha.astype(np.float32) / 255.0)[..., None]
    dst_slice = scene[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    blended = dst_slice * (1 - alpha_f) + region_tile.astype(np.float32) * alpha_f
    scene[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
    mask_bool[dst_y0:dst_y1, dst_x0:dst_x1] = region_alpha > 127
    return scene, mask_bool


def composite_scene(
    background: np.ndarray,
    num_tiles: int,
    rng: random.Random,
    cluster_center: Optional[Tuple[int, int]] = None,
    spread: int = 40,
    allow_overlap: bool = False,
    max_overlap_pixels: int = 500,  # ~5% of one tile's ~9600px mask -- "touching," not heavily overlapping
    max_attempts_per_tile: int = 40,
) -> Tuple[np.ndarray, List[SyntheticTileInstance]]:
    """Composites num_tiles randomly-rotated tiles onto background, jittered
    around cluster_center (defaults to the background's own center) within
    spread pixels -- a tight spread produces touching/overlapping placements
    rather than tiles scattered across the whole frame, mimicking a boneyard
    pile or a loose hand-fan.

    Unless allow_overlap is set, each candidate placement is rejected and
    retried (up to max_attempts_per_tile times) if it would overlap an
    already-placed tile's mask by more than max_overlap_pixels -- real
    dominoes resting flat on a table can touch edge-to-edge but can't occupy
    the same 2D projection past a small perspective sliver, so an
    unconstrained random placement would produce unrealistic heavy overlaps.
    Falls back to the last attempted placement if none satisfy the
    threshold, rather than silently placing fewer than num_tiles."""
    scene = background.copy()
    h, w = scene.shape[:2]
    if cluster_center is None:
        cluster_center = (w // 2, h // 2)

    instances: List[SyntheticTileInstance] = []
    for _ in range(num_tiles):
        top, bottom = rng.randint(0, 6), rng.randint(0, 6)
        tile = render_tile(top, bottom)

        best: Optional[Tuple[np.ndarray, np.ndarray, int, int, float]] = None
        best_overlap = None
        for _attempt in range(max_attempts_per_tile):
            angle = rng.uniform(0, 360)
            cx = cluster_center[0] + rng.randint(-spread, spread)
            cy = cluster_center[1] + rng.randint(-spread, spread)
            trial_scene, trial_mask = _paste_rotated(scene.copy(), tile, (cx, cy), angle)
            overlap = sum(int((trial_mask & inst.mask).sum()) for inst in instances)
            if allow_overlap or overlap <= max_overlap_pixels:
                best = (trial_scene, trial_mask, cx, cy, angle)
                break
            if best_overlap is None or overlap < best_overlap:
                best, best_overlap = (trial_scene, trial_mask, cx, cy, angle), overlap

        trial_scene, trial_mask, cx, cy, angle = best
        scene = trial_scene
        instances.append(SyntheticTileInstance(top=top, bottom=bottom, center=(cx, cy), angle=angle, mask=trial_mask))

    return scene, instances
