"""Hand detection and exclusion-zone gating for the perception pipeline.

A hand/forearm resting near the tiles sometimes passes TileLocalizer's green
HSV mask and reads as tile material -- every geometry-only fix for this
(blob shape statistics, watershed, pip-divider periodicity) failed, since a
hand blob and a genuine multi-tile cluster land in overlapping territory on
every shape statistic at this camera's resolution/compression (see
RESEARCH.md). MediaPipe Hands sidesteps this entirely: it's a purpose-built
hand detector with no dependency on color/shape at all, so this module asks
"is a hand here" directly instead of inferring it from the tile-color mask.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from eye42.perception.tile_detect import TableRectifier, TileObservation

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
_DEFAULT_MODEL_PATH = Path.home() / ".cache" / "eye42" / "hand_landmarker.task"

# A hand entering frame is far smaller-area motion than a trick sweep
# (EventSegmenter._sweep_in_progress's 0.15 threshold) -- this needs to catch
# a hand approaching the table, not just a full sweep. Needs real-footage
# tuning, same as EventSegmenter._HAND_BOUNDARY_SUSTAINED_FRAMES already is.
_MIN_MOTION_FRACTION = 0.0005

# How far past a detected hand's own landmark-derived bounding box to still
# treat as "under the hand" -- covers the box-vs-true-hand-extent slack, not
# a generic safety margin: real footage measurement (a genuine tile at
# rectified (590, 273), the nearest hand's own box edge at x=632) leaves only
# ~42px of headroom before an exclusion zone reaches a real tile, so this
# must stay small.
_DEFAULT_EXCLUSION_MARGIN = 8.0


@dataclass(frozen=True)
class HandRegion:
    """One detected hand, in the raw camera frame's pixel space (not
    rectified) -- MediaPipe's hand model expects an undistorted,
    non-perspective-warped image, unlike TileLocalizer's rectified input."""

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model(model_path: Path) -> Path:
    """Downloads the HandLandmarker model to model_path if not already
    cached, verifying it against a pinned checksum. MediaPipe's Tasks API
    has no built-in auto-download -- this is bespoke fetch-if-missing logic,
    not a library feature."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    except OSError as exc:
        raise RuntimeError(
            f"HandDetector's model file is missing ({model_path}) and couldn't be "
            f"downloaded from {_MODEL_URL} (no network?). Fetch it manually to that "
            f"path before starting a live session without connectivity."
        ) from exc
    actual = _sha256(model_path)
    if actual != _MODEL_SHA256:
        model_path.unlink()
        raise RuntimeError(
            f"Downloaded model at {_MODEL_URL} does not match the pinned checksum "
            f"(expected {_MODEL_SHA256}, got {actual}) -- refusing to use it."
        )
    return model_path


class HandDetector:
    """Wraps MediaPipe Tasks HandLandmarker. Runs on the RAW frame, before
    TableRectifier's homography -- MediaPipe's hand model expects an
    undistorted, non-perspective-warped camera image."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_hand_confidence: float = 0.5,
        max_hands: int = 4,
    ) -> None:
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

        self._Image = Image
        self._ImageFormat = ImageFormat
        resolved_path = _ensure_model(model_path or _DEFAULT_MODEL_PATH)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            num_hands=max_hands,
            min_hand_detection_confidence=min_hand_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[HandRegion]:
        """Bounding box per detected hand, derived from min/max over its 21
        landmarks (MediaPipe reports normalized [0, 1] landmark coordinates,
        scaled here to the frame's actual pixel size)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        regions: List[HandRegion] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            confidence = handedness[0].score if handedness else 0.0
            regions.append(HandRegion(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys), confidence=confidence))
        return regions


def _project_bbox(rectifier: TableRectifier, region: HandRegion) -> Tuple[float, float, float, float]:
    corners = [
        (region.x0, region.y0), (region.x1, region.y0),
        (region.x1, region.y1), (region.x0, region.y1),
    ]
    projected = rectifier.project_points(corners)
    xs, ys = projected[:, 0], projected[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


class HandGate:
    """Combines hand detection, motion-gated invocation, coordinate
    projection, and observation suppression, so a caller can drop a single
    filter() call between observe_frame() and EventSegmenter.feed() rather
    than wiring each of those steps in independently."""

    def __init__(
        self,
        detector: HandDetector,
        rectifier: TableRectifier,
        min_motion_fraction: float = _MIN_MOTION_FRACTION,
        exclusion_margin: float = _DEFAULT_EXCLUSION_MARGIN,
    ) -> None:
        self._detector = detector
        self._rectifier = rectifier
        self._min_motion_fraction = min_motion_fraction
        self._exclusion_margin = exclusion_margin
        self.last_hand_regions_raw: List[HandRegion] = []
        self.last_suppressed: List[TileObservation] = []

    def filter(
        self,
        raw_frame: np.ndarray,
        observations: Sequence[TileObservation],
        motion_mask: Optional[np.ndarray],
    ) -> List[TileObservation]:
        """Returns observations with any near a detected hand removed.
        Re-runs the hand detector only if motion_mask indicates recent
        motion; a static scene's hand status can't have changed since the
        last check. motion_mask is expected to be the PREVIOUS frame's mask
        (the caller's own segmenter hasn't processed the current frame yet
        at the point this is called) -- fine for a go/no-go gate, since
        motion-history-based motion in this pipeline is already a
        multi-frame smoothed signal, not something needing single-frame
        precision."""
        should_detect = motion_mask is None or (float(motion_mask.sum()) / motion_mask.size) >= self._min_motion_fraction
        if should_detect:
            self.last_hand_regions_raw = self._detector.detect(raw_frame)

        exclusion_zones = []
        for region in self.last_hand_regions_raw:
            x0, y0, x1, y1 = _project_bbox(self._rectifier, region)
            m = self._exclusion_margin
            exclusion_zones.append((x0 - m, y0 - m, x1 + m, y1 + m))

        survivors: List[TileObservation] = []
        suppressed: List[TileObservation] = []
        for obs in observations:
            x, y = obs.position
            under_hand = any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in exclusion_zones)
            (suppressed if under_hand else survivors).append(obs)

        self.last_suppressed = suppressed
        return survivors
