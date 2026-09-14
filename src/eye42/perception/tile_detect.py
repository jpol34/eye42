"""Camera perception: rectified frame -> tile observations -> play events.

Pipeline shape: homography rectification -> tile localization -> 28-class
identity classification -> settle-time-debounced event segmentation (with
motion-trail player attribution) -> trick-sweep / hand-boundary detection,
feeding eye42.engine.events.TilePlayed / IrregularEndSignal into a HandState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from eye42.engine.events import TilePlayed
from eye42.engine.tiles import Tile

SEAT_ORDER = ("N", "E", "S", "W")  # matches tools/calibrate.py's TL/TR/BR/BL corner labeling
SEAT_INDEX = {name: i for i, name in enumerate(SEAT_ORDER)}


@dataclass(frozen=True)
class TileObservation:
    """A single detected tile on the rectified table, before it's decided to be
    a genuine "play" (see the settle-time debounce in EventSegmenter)."""

    tile: Tile
    confidence: float
    position: Tuple[float, float]  # rectified-plane coordinates
    frame_index: int


class TableRectifier:
    """Wraps the one-time 4-corner homography computed for a recording session.
    See tools/crop_rectify.py for the throwaway version used to validate a
    capture setup before committing to real sessions, and tools/calibrate.py
    for the real per-session calibration this class is built from.
    """

    def __init__(self, corners: Sequence[Tuple[float, float]], output_size: Tuple[int, int]) -> None:
        if len(corners) != 4:
            raise ValueError(f"TableRectifier needs exactly 4 corners (TL, TR, BR, BL), got {len(corners)}")
        self.output_size = output_size
        src = np.array(corners, dtype=np.float32)
        w, h = output_size
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(src, dst)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self._matrix, self.output_size)

    def project_points(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Projects raw-frame points through this rectifier's own homography,
        landing them in the same rectified-plane space find_tiles()/
        TileObservation.position already use."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._matrix).reshape(-1, 2)


# A tile's green is highly distinguishable from a white table -- wide enough
# to tolerate ordinary room-lighting variation without picking up the table
# surface itself.
_TILE_HSV_LOW = np.array([35, 40, 20])
_TILE_HSV_HIGH = np.array([95, 255, 210])
_MIN_TILE_AREA = 400.0
_MAX_TILE_AREA = 20000.0
# A domino's long side runs a little over 2x its short side; the tolerance
# band allows for the localizer's bounding-box vs. true-edge rounding.
_MIN_TILE_ASPECT = 1.4
_MAX_TILE_ASPECT = 3.2


@dataclass(frozen=True)
class TileRegion:
    """A tile-shaped region found on a rectified frame. ``x``/``y``/``width``/
    ``height`` are the axis-aligned bounding box (for area/aspect filtering);
    ``center``/``angle``/``rect_size`` are ``cv2.minAreaRect``'s own rotated
    box, needed to bring the tile to a canonical upright crop before
    splitting it into its two pip-bearing halves (see
    ``extract_aligned_crop``) -- the two aren't interchangeable, since a
    rotated rectangle's bounding box is larger than the rectangle itself."""

    x: int
    y: int
    width: int
    height: int
    center: Tuple[float, float]
    angle: float
    rect_size: Tuple[float, float]


def _tile_color_contours(rectified_frame: np.ndarray) -> List[np.ndarray]:
    """The tile-color mask + contour extraction shared by ``TileLocalizer``
    and cluster detection, so the two never drift apart under future
    threshold/kernel tuning."""
    hsv = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _TILE_HSV_LOW, _TILE_HSV_HIGH)
    # A tile's own pip-divider line is bright/white and can cut its green
    # mask into two separate blobs -- closing with a fixed kernel only
    # bridges that gap up to some rectified resolution, since the
    # divider's pixel width scales with resolution too; sizing the
    # kernel off the frame itself keeps the bridge reliable regardless
    # of the rectifier's output size.
    kernel_size = max(5, round(min(rectified_frame.shape[:2]) * 0.005)) | 1  # odd size
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


class TileLocalizer:
    """Finds candidate tile bounding boxes/orientations on a rectified frame."""

    def find_tiles(self, rectified_frame: np.ndarray) -> List[TileRegion]:
        regions: List[TileRegion] = []
        for contour in _tile_color_contours(rectified_frame):
            area = cv2.contourArea(contour)
            if not (_MIN_TILE_AREA <= area <= _MAX_TILE_AREA):
                continue
            (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
            short_side, long_side = min(w, h), max(w, h)
            if short_side < 1:
                continue
            aspect = long_side / short_side
            if not (_MIN_TILE_ASPECT <= aspect <= _MAX_TILE_ASPECT):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            regions.append(TileRegion(
                x=x, y=y, width=bw, height=bh, center=(cx, cy),
                angle=angle, rect_size=(w, h),
            ))
        return regions


_CLUSTER_AREA_THRESHOLD = _MAX_TILE_AREA * 3  # several touching tiles merged
# into one blob (a boneyard pile, an in-progress shuffle) rather than a
# single tile or two touching ones -- real shuffle piles measure well past
# this multiple of a single tile's own max area.


def unseparated_tile_cluster_regions(rectified_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes (x, y, w, h) of contours far too large to be one or two
    touching tiles. ``TileLocalizer.find_tiles`` already rejects such an
    oversized contour as non-tile-shaped, but that leaves real, isolated
    tiles right at the edge of the same cluster free to be tracked as
    ordinary settled plays even while the cluster itself (a shuffle in
    progress, tiles still being scrambled) makes any single position within
    or beside it meaningless to report as a play. Scoped to bounding boxes,
    not a frame-wide flag, so a persistent but stationary cluster elsewhere
    on the table (a boneyard sitting in its own corner all hand) can't
    freeze plays it has nothing to do with."""
    return [
        cv2.boundingRect(c)
        for c in _tile_color_contours(rectified_frame)
        if cv2.contourArea(c) > _CLUSTER_AREA_THRESHOLD
    ]


def _near_any_region(
    position: Tuple[float, float], regions: List[Tuple[int, int, int, int]], margin: float,
) -> bool:
    x, y = position
    return any(
        rx - margin <= x <= rx + rw + margin and ry - margin <= y <= ry + rh + margin
        for rx, ry, rw, rh in regions
    )


def extract_aligned_crop(rectified_frame: np.ndarray, region: TileRegion, margin: float = 1.3) -> np.ndarray:
    """Rotates and crops a tile region to a canonical upright orientation
    (long axis vertical), regardless of how the tile sits on the table --
    pip-half splitting only works on a consistently oriented crop.

    ``cv2.minAreaRect``'s angle convention differs across OpenCV builds (the
    sign and which side it's measured from both vary), so rather than trust
    it directly this reconstructs the box's corners (``cv2.boxPoints``) and
    computes the rotation needed to bring its actual longest edge to
    vertical -- correct regardless of angle-convention quirks.
    """
    box = cv2.boxPoints((region.center, region.rect_size, region.angle))
    edge_vecs = [box[(i + 1) % 4] - box[i] for i in range(4)]
    edge_lens = [float(np.hypot(*v)) for v in edge_vecs]
    dx, dy = edge_vecs[int(np.argmax(edge_lens))]
    # cv2.getRotationMatrix2D(angle) rotates a vector at (atan2(dy,dx)) to
    # (atan2(dy,dx) - angle); solving for the angle that lands it pointing
    # straight up (atan2 == -90) gives this +90 offset.
    angle = float(np.degrees(np.arctan2(dy, dx))) + 90.0

    cx, cy = region.center
    long_side = max(region.rect_size) * margin
    short_side = min(region.rect_size) * margin

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(rectified_frame, rot_mat, (rectified_frame.shape[1], rectified_frame.shape[0]))

    x0 = max(int(cx - short_side / 2), 0)
    x1 = min(int(cx + short_side / 2), rectified_frame.shape[1])
    y0 = max(int(cy - long_side / 2), 0)
    y1 = min(int(cy + long_side / 2), rectified_frame.shape[0])
    return rotated[y0:y1, x0:x1]


class TileIdentityClassifier(Protocol):
    """A per-tile-half or whole-tile classifier over the 28 known tile classes.

    Per the plan: perspective-corrected crop -> small template-match or
    lightweight CNN, NOT Hough-circle pip counting (known failure modes: the
    center spinner pin reads as a false pip, blanks are indistinguishable from
    a failed detection).
    """

    def classify(self, tile_crop: np.ndarray) -> List[Tuple[Tile, float]]:
        """Returns candidate tile identities ranked by confidence (best guess
        first, score 0-1 each), not just a single committed identity -- the
        same "track weighted candidates, don't commit to one guess" convention
        engine.trump_inference.TrumpHypothesisTracker uses for ambiguity."""
        ...


# White pips are near-saturation-zero and near-max-value against this tile
# set's green -- a much simpler and more reliable signal than shape alone.
_PIP_HSV_LOW = np.array([0, 0, 165])
_PIP_HSV_HIGH = np.array([180, 70, 255])
_MIN_PIP_AREA = 8.0
_MAX_PIP_AREA = 450.0
_MERGED_BLOB_RATIO = 1.6  # a blob this many times the half's own single-pip unit is treated as N touching pips
_MAX_PIPS_PER_HALF = 6  # a blob estimating past this is contamination (crop-edge background), not touching pips
_ALT_COUNT_MARGIN = 0.15  # a ratio within this fraction of a rounding boundary is genuinely ambiguous


def _count_pips(half: np.ndarray) -> Tuple[int, bool, Optional[int]]:
    """Counts pips in one tile half via color, not Hough circles: the spinner
    pin is the wrong color (metal/black, not white), so it never survives
    this filter, and a genuinely blank half correctly returns 0 rather than
    looking like a failed detection.

    A tightly-packed pattern (5, 6) can have pips touching closely enough
    that a plain color threshold merges them into one blob -- confirmed
    against real photos of this tile set, not assumed; at typical camera
    resolution a pip is only ~10px across, too coarse for a distance-
    transform peak count to reliably separate (the transform's few discrete
    levels form flat plateaus rather than clean single peaks). Instead, each
    half calibrates its own "one pip" unit from its own smallest surviving
    blob, and any larger blob's pip count is estimated as its area's ratio
    to that unit -- self-calibrating to whatever resolution/distance this
    session's camera happens to be at. A ratio past what any real domino
    half could hold is treated as background contamination (e.g. a sliver
    of table caught at a rotated crop's corner) and discarded rather than
    folded into the count.

    Returns (count, any_borderline, alt_count) so the caller can reflect an
    estimated or discarded blob in its confidence, and offer a second
    plausible reading when there's exactly one. ``alt_count`` is only ever
    set when a single blob's area ratio sits close enough to a rounding
    boundary to be genuinely ambiguous between two counts; a half with more
    than one independently ambiguous blob has no well-defined single
    alternate (combining separate guesses would be exactly that -- a guess,
    not a measurement), so it reports None.
    """
    if half.size == 0:
        return 0, True, None
    hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _PIP_HSV_LOW, _PIP_HSV_HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    areas = [a for a in areas if a >= _MIN_PIP_AREA]
    if not areas:
        return 0, False, None

    single_pip_candidates = [a for a in areas if a <= _MAX_PIP_AREA] or areas
    unit_area = float(np.median(single_pip_candidates))

    count = 0
    borderline = False
    ambiguous_deltas: List[int] = []
    for area in areas:
        if area <= unit_area * _MERGED_BLOB_RATIO:
            count += 1
            continue
        ratio = area / unit_area
        estimated = max(1, round(ratio))
        borderline = True
        if estimated > _MAX_PIPS_PER_HALF:
            continue  # too many estimated to be real pips -- contamination, discard entirely
        count += estimated
        if abs(ratio - estimated) >= 0.5 - _ALT_COUNT_MARGIN:
            ambiguous_deltas.append(-1 if ratio < estimated else 1)

    alt_count = None
    if len(ambiguous_deltas) == 1:
        alt = count + ambiguous_deltas[0]
        if 0 <= alt <= _MAX_PIPS_PER_HALF:
            alt_count = alt
    return count, borderline, alt_count


class OpenCVTileClassifier:
    """Concrete ``TileIdentityClassifier``: splits an already-upright crop
    (see ``extract_aligned_crop``) into its two halves along the long axis
    and counts pips in each via color + circularity."""

    def classify(self, tile_crop: np.ndarray) -> List[Tuple[Tile, float]]:
        h, w = tile_crop.shape[:2]
        # A gap at the split point excludes the tile's own divider groove,
        # which otherwise bridges pips on either side of it into one
        # low-circularity blob -- pips themselves sit well clear of the
        # divider, so this doesn't cut into real ones the way a uniform
        # inset from every edge would (that also eats edge-adjacent pips).
        if h >= w:
            gap = max(2, round(h * 0.06))
            mid = h // 2
            half_a, half_b = tile_crop[: mid - gap, :], tile_crop[mid + gap:, :]
        else:
            gap = max(2, round(w * 0.06))
            mid = w // 2
            half_a, half_b = tile_crop[:, : mid - gap], tile_crop[:, mid + gap:]

        count_a, borderline_a, alt_a = _count_pips(half_a)
        count_b, borderline_b, alt_b = _count_pips(half_b)
        if not (0 <= count_a <= 6 and 0 <= count_b <= 6):
            return []

        confidence = 0.6 if (borderline_a or borderline_b) else 1.0
        candidates = [(Tile.of(count_a, count_b), confidence)]

        # Only one half's ambiguity gets a second candidate: with both halves
        # independently ambiguous, the alternates would combine into several
        # equally-unfounded guesses rather than one real second reading.
        if alt_a is not None and alt_b is None:
            candidates.append((Tile.of(alt_a, count_b), confidence * 0.5))
        elif alt_b is not None and alt_a is None:
            candidates.append((Tile.of(count_a, alt_b), confidence * 0.5))

        return candidates


def observe_frame(
    rectifier: TableRectifier,
    localizer: TileLocalizer,
    classifier: TileIdentityClassifier,
    frame: np.ndarray,
    frame_index: int,
) -> Tuple[np.ndarray, List[TileObservation]]:
    """Runs stages 1-3 for one raw camera frame. Returns the rectified frame
    (the segmenter needs it for motion tracking) alongside whatever tiles
    were observed."""
    rectified = rectifier.rectify(frame)
    observations: List[TileObservation] = []
    for region in localizer.find_tiles(rectified):
        crop = extract_aligned_crop(rectified, region)
        candidates = classifier.classify(crop)
        if not candidates:
            continue
        tile, confidence = candidates[0]
        observations.append(TileObservation(
            tile=tile, confidence=confidence, position=region.center, frame_index=frame_index,
        ))
    return rectified, observations


@dataclass
class _PendingTile:
    position: Tuple[float, float]
    stable_frames: int = 1
    # tile -> that tile's per-frame confidences, one entry per matched frame this
    # candidate has seen -- len(sum of all lists) always equals stable_frames, so
    # a settling identity is resolved by plurality vote across the whole settle
    # window (see _resolve_identity) rather than trusting whichever frame happens
    # to be last, which a single misclassified frame could otherwise corrupt.
    votes: Dict[Tile, List[float]] = field(default_factory=dict)


def _resolve_identity(votes: Dict[Tile, List[float]]) -> Tuple[Tile, float]:
    """Most frames agreeing wins; ties broken by higher total confidence, and any
    further tie by whichever tile was observed first -- an arbitrary but stable
    and deterministic rule, not a claim that it's the "more correct" choice.
    Reported confidence is the mean of the winning tile's own votes."""
    winner = max(votes, key=lambda t: (len(votes[t]), sum(votes[t])))
    confidences = votes[winner]
    return winner, sum(confidences) / len(confidences)


_POSITION_TOLERANCE = 15.0  # rectified-plane pixels; "the same tile" across frames
_MHI_DURATION = 1.0  # seconds of motion history to retain
_ATTRIBUTION_ROI_RADIUS = 70  # pixels around a settled tile to inspect for its motion trail
_SWEEP_SUSTAINED_FRAMES = 3  # filters a single noisy motion spike from firing the
# confirmed-position prune below; much shorter than HandBoundaryDetector's 20-frame
# reshuffle window since a trick-sweep is a brief, ballistic motion, not sustained scrambling.


class EventSegmenter:
    """Turns a stream of per-frame TileObservations into discrete play events,
    using a settle-time debounce (a tile only counts as "played" after N stable
    frames — same pattern as DGT chessboard debouncing) and a rolling motion-
    history buffer (the same per-frame window the debounce already needs) to
    attribute a settled tile to whichever of the 4 known seats its motion
    trail arrived from -- a played tile's resting position alone says nothing
    about who played it, since it can end up anywhere in the shared trick
    area.

    Explicit trick-sweep handling: if all 4 tiles of a trick aren't logged
    before the winner's sweep clears the area, this degrades gracefully
    rather than raising or silently under-counting -- the same force-closed
    pattern engine.trick.Trick and engine.hand.HandState use (see
    IrregularityKind.TRICK_FORCE_CLOSED), so a missed sweep-tile is flagged
    as a real-event irregularity, not a crash. A trick-sweep's motion is
    large-area and multi-tile, categorically different from a single play,
    so it's excluded from attribution rather than misread as one.

    A sustained trick-sweep also prunes ``_confirmed_positions`` back to just
    the hand's baseline: players can toss a trick anywhere on the table, not
    one fixed zone, so a later trick's play can land at the same position an
    earlier trick's play did within the same hand, and without pruning that
    position would still read as "already confirmed" and be silently dropped.
    """

    def __init__(self, settle_frames: int = 15, seats: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.settle_frames = settle_frames
        self.seats = seats or {}
        self._pending: List[_PendingTile] = []
        self._confirmed_positions: List[Tuple[float, float]] = []
        self._mhi: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._timestamp = 0.0
        # Exposed for HandBoundaryDetector, which needs the same per-frame
        # motion mask this class already computes for debounce/attribution --
        # recomputing it separately would just be the same cv2.absdiff twice.
        self.last_motion_mask: Optional[np.ndarray] = None
        # A freshly dealt hand's own fanned tiles (and the boneyard) sit
        # perfectly still for as long as nobody is playing -- indistinguishable,
        # by position alone, from a tile that just settled after being played.
        # So a new hand starts in a "settling" phase: whatever tiles are on
        # the table once dealing/shuffling motion has genuinely quieted down
        # (not just momentarily still) are recorded as a baseline and never
        # reported as plays, exactly like an already-confirmed play -- only a
        # tile that appears at a position outside that baseline is a real play.
        self._settling = True
        self._baseline_positions: List[Tuple[float, float]] = []
        self._quiet_frames = 0
        # Players can toss a trick's tiles anywhere on the table, not one
        # designated area -- the same physical position can end up hosting a
        # play in more than one trick within a hand, so _confirmed_positions
        # (below) needs pruning at each trick-sweep, not just per-hand.
        self._sweep_streak = 0

    def _update_motion(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._timestamp += 1.0
        if self._prev_gray is None or self._mhi is None:
            self._prev_gray = gray
            self._mhi = np.zeros(gray.shape, dtype=np.float32)
            return np.zeros(gray.shape, dtype=np.uint8)

        diff = cv2.absdiff(gray, self._prev_gray)
        _, motion_mask = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        cv2.motempl.updateMotionHistory(motion_mask, self._mhi, self._timestamp, _MHI_DURATION)
        self._prev_gray = gray
        return motion_mask

    def _sweep_in_progress(self, motion_mask: np.ndarray) -> bool:
        """A trick-sweep moves a large fraction of the table at once; a
        single play's motion is a small, localized blob near one tile."""
        table_area = motion_mask.shape[0] * motion_mask.shape[1]
        return bool(motion_mask.sum() > 0.15 * table_area)

    def _attribute(self, position: Tuple[float, float]) -> Tuple[int, float]:
        """Reads the motion-history trail arriving at ``position`` and
        matches its direction against the known seat positions. Returns
        (player, player_confidence); a low confidence means the caller should
        record the play but flag it, not trust it blindly -- see
        IrregularityKind.AMBIGUOUS_ATTRIBUTION."""
        if self._mhi is None or not self.seats:
            return 0, 0.0

        x, y = int(position[0]), int(position[1])
        y0 = max(y - _ATTRIBUTION_ROI_RADIUS, 0)
        y1 = min(y + _ATTRIBUTION_ROI_RADIUS, self._mhi.shape[0])
        x0 = max(x - _ATTRIBUTION_ROI_RADIUS, 0)
        x1 = min(x + _ATTRIBUTION_ROI_RADIUS, self._mhi.shape[1])
        roi = self._mhi[y0:y1, x0:x1]
        if roi.size == 0 or float(roi.max()) == 0.0:
            return 0, 0.0

        grad_mask, orientation = cv2.motempl.calcMotionGradient(roi, 0.25, _MHI_DURATION, apertureSize=5)
        if not grad_mask.any():
            return 0, 0.0
        arrival_angle = cv2.motempl.calcGlobalOrientation(orientation, grad_mask, roi, self._timestamp, _MHI_DURATION)
        # The trail points in the direction motion moved TOWARD; a tile
        # arrived FROM the opposite direction.
        origin_angle = np.radians((arrival_angle + 180.0) % 360.0)
        origin_vector = (np.cos(origin_angle), np.sin(origin_angle))

        best_seat, best_score = None, -2.0
        for seat_name, seat_pos in self.seats.items():
            to_seat = (seat_pos[0] - x, seat_pos[1] - y)
            norm = float(np.hypot(*to_seat))
            if norm == 0:
                continue
            to_seat_unit = (to_seat[0] / norm, to_seat[1] / norm)
            score = origin_vector[0] * to_seat_unit[0] + origin_vector[1] * to_seat_unit[1]
            if score > best_score:
                best_seat, best_score = seat_name, score

        if best_seat is None:
            return 0, 0.0
        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        return SEAT_INDEX[best_seat], confidence

    def _closest(self, position: Tuple[float, float], observations: List[TileObservation]) -> Optional[TileObservation]:
        best, best_dist = None, _POSITION_TOLERANCE
        for obs in observations:
            dist = float(np.hypot(obs.position[0] - position[0], obs.position[1] - position[1]))
            if dist <= best_dist:
                best, best_dist = obs, dist
        return best

    def _already_confirmed(self, position: Tuple[float, float]) -> bool:
        return any(
            np.hypot(position[0] - p[0], position[1] - p[1]) <= _POSITION_TOLERANCE
            for p in self._confirmed_positions
        )

    def _merge_baseline(self, observations: List[TileObservation]) -> None:
        for obs in observations:
            if not any(
                np.hypot(obs.position[0] - p[0], obs.position[1] - p[1]) <= _POSITION_TOLERANCE
                for p in self._baseline_positions
            ):
                self._baseline_positions.append(obs.position)

    def feed(self, frame: np.ndarray, observations: List[TileObservation]) -> List[TilePlayed]:
        """Feed one frame (for motion tracking) and that frame's observed
        tiles; returns any plays that just settled (usually none or one, but
        never assumed to be at most one)."""
        motion_mask = self._update_motion(frame)
        self.last_motion_mask = motion_mask
        sweeping = self._sweep_in_progress(motion_mask)

        if self._settling:
            self._merge_baseline(observations)
            self._quiet_frames = 0 if sweeping else self._quiet_frames + 1
            if self._quiet_frames >= self.settle_frames:
                self._confirmed_positions = list(self._baseline_positions)
                self._settling = False
            return []

        cluster_regions: Optional[List[Tuple[int, int, int, int]]] = None
        self._sweep_streak = self._sweep_streak + 1 if sweeping else 0
        if self._sweep_streak == _SWEEP_SUSTAINED_FRAMES:
            # Fires once per sweep (streak keeps climbing past this value until
            # sweeping ends and resets to 0), not every sweeping frame -- an
            # unconditional per-frame prune could wipe a position that was
            # confirmed only moments earlier in the same still-ongoing sweep,
            # letting that same physical tile silently re-settle and get
            # reported as a second, spurious play.
            self._confirmed_positions = list(self._baseline_positions)

        remaining = [o for o in observations if not self._already_confirmed(o.position)]
        played: List[TilePlayed] = []
        still_pending: List[_PendingTile] = []

        for pending in self._pending:
            match = self._closest(pending.position, remaining)
            if match is None:
                continue  # tile vanished before settling -- drop the candidate
            remaining.remove(match)
            pending.position = match.position
            if sweeping:
                # A trick-sweep's large-area motion means no play can be
                # reliably attributed, and _confirmed_positions was just
                # pruned back toward baseline above -- freeze this
                # candidate's progress rather than accumulating votes or
                # confirming, so a sweep that runs long can't grow a
                # pending tile's vote history unboundedly and can't
                # silently re-confirm an already-played tile whose
                # confirmed status the prune above just cleared. Progress
                # already made resumes, unlost, once the sweep ends.
                still_pending.append(pending)
                continue
            # Computed lazily (once per frame, cached above) and only once
            # at least one non-sweeping pending candidate needs it -- a
            # frame with nothing pending, or where every pending candidate
            # is already frozen by a sweep, never pays for it at all.
            if cluster_regions is None:
                cluster_regions = unseparated_tile_cluster_regions(frame)
            near_cluster = _near_any_region(pending.position, cluster_regions, margin=_ATTRIBUTION_ROI_RADIUS)
            if near_cluster:
                # No position within or beside an unseparated tile cluster
                # (a boneyard pile, an in-progress shuffle) is a meaningful
                # settled play -- same freeze-in-place treatment as a
                # sweep, above. Scoped to nearby candidates only, so a
                # stationary cluster elsewhere on the table (a boneyard
                # sitting in its own corner all hand) can't freeze plays
                # it has nothing to do with.
                still_pending.append(pending)
                continue
            pending.stable_frames += 1
            pending.votes.setdefault(match.tile, []).append(match.confidence)
            if pending.stable_frames >= self.settle_frames:
                self._confirmed_positions.append(pending.position)
                player, player_confidence = self._attribute(pending.position)
                tile, confidence = _resolve_identity(pending.votes)
                played.append(TilePlayed(
                    player=player, tile=tile, confidence=confidence,
                    player_confidence=player_confidence,
                ))
            else:
                still_pending.append(pending)

        for obs in remaining:
            still_pending.append(_PendingTile(position=obs.position, votes={obs.tile: [obs.confidence]}))

        self._pending = still_pending
        return played

    def reset(self) -> None:
        """Clears per-hand play-tracking state (pending/confirmed tile
        positions) when a new hand starts -- motion history stays
        continuous across the reset since it's tracking physical movement,
        not hand-scoped state, and a reshuffle's own motion is exactly what
        triggered this in the first place (see HandBoundaryDetector).

        Re-enters the settling phase so the newly dealt hand's own tiles are
        captured as a baseline (see __init__) rather than reported as plays."""
        self._pending = []
        self._confirmed_positions = []
        self._settling = True
        self._baseline_positions = []
        self._quiet_frames = 0
        self._sweep_streak = 0


_HAND_BOUNDARY_AREA_FRACTION = 0.35  # a reshuffle moves far more of the table than any single trick-sweep
_HAND_BOUNDARY_SUSTAINED_FRAMES = 20  # needs real footage to tune; see ROADMAP


class HandBoundaryDetector:
    """Detects shuffle/reshuffle (new hand started) via large-scale, sustained
    tile motion across the whole table, distinct from in-trick motion.

    Edge-triggered, not level-triggered: a real shuffle's motion stays above
    threshold for many consecutive frames, so a naive "still above threshold"
    check would report a new hand starting on every one of those frames
    instead of once -- this fires only on the frame the sustained threshold
    is first crossed, then stays quiet until motion drops back down (the
    shuffle ends) before it can fire again for the next one.
    """

    def __init__(self) -> None:
        self._sustained_frames = 0
        self._fired = False

    def is_new_hand_starting(self, motion_field: np.ndarray) -> bool:
        table_area = motion_field.shape[0] * motion_field.shape[1]
        if motion_field.sum() <= _HAND_BOUNDARY_AREA_FRACTION * table_area:
            self._sustained_frames = 0
            self._fired = False
            return False
        self._sustained_frames += 1
        if self._sustained_frames >= _HAND_BOUNDARY_SUSTAINED_FRAMES and not self._fired:
            self._fired = True
            return True
        return False
