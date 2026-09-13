"""A procedural, rigid-box hand+forearm occluder rig for eye42.simgen.

Real-footage research (RESEARCH.md: "Hand/manipulation fidelity") found hand-tile contact
present in 94-100% of sampled frames, with occlusion shifting through finger gaps and
forearm coverage -- not a simple whole-hand blob. This module is deliberately NOT a bpy
armature/skinned mesh: render_ground_truth (render.py) must stay usable without the
optional ``sim-render`` extra, which rules out projecting a deformed skinned mesh's
vertices. Instead, hand_parts() does plain forward kinematics in numpy, returning a flat
list of oriented boxes -- the same (position, quaternion, half-extents) shape TileState/
TILE_HALF_EXTENTS_M already use, so render.py's existing per-object render loops and
projection math both apply to hand parts with no new concepts.

Phase 1 scope: one parameterized reach-place-retract pose path (place_hand), sampled at a
randomized late phase per frame -- not the full ~8-clip keyframed gesture library, not
contact-driven fingertip physics (tiles still move via physics.slide_toward; the hand is
posed near a tile, not causing its motion), not per-finger deformation. See ROADMAP.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from eye42.simgen.tile_geometry import TILE_HALF_EXTENTS_M

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_TILE_TOP_Z_M = TILE_HALF_EXTENTS_M[2]  # a resting tile's top surface height above the table plane

# Rough human-hand-scale proportions (meters), palm-down, fingers extending +x in the
# hand's own local frame before the pose's yaw/translation are applied.
_FOREARM_SEGMENT_LEN_M = 0.09
_FOREARM_HALF_EXTENTS_M = (0.045, 0.035, 0.018)
_PALM_LEN_M = 0.09
_PALM_HALF_EXTENTS_M = (0.045, 0.045, 0.010)
_FINGER_PROXIMAL_LEN_M = 0.035
_FINGER_DISTAL_LEN_M = 0.025
_FINGER_HALF_EXTENTS_M = (0.017, 0.009, 0.008)
_THUMB_PROXIMAL_LEN_M = 0.028
_THUMB_DISTAL_LEN_M = 0.020
_THUMB_HALF_EXTENTS_M = (0.014, 0.009, 0.008)
_FINGER_BASE_Y_OFFSETS_M = (-0.033, -0.011, 0.011, 0.033)  # lateral spacing across palm front
_MAX_CURL_RAD = 0.9  # ~52 degrees -- fingers stay closer to flat than a full fist, matching
# the footage finding that hands stay flat/palm-down throughout play, never fanning tiles.
_MAX_SPREAD_M = 0.012  # extra lateral offset at full spread
_THUMB_YAW_RAD = -0.8  # angled off to the side of the palm, not parallel to the fingers
_TABLE_CLEARANCE_M = 0.006  # how far the palm/finger undersides hover above a resting tile's
# top surface -- Phase 1 is kinematic (no contact physics), so this is a fixed offset, not
# a physically resolved contact.


@dataclass(frozen=True)
class RigidPart:
    """One rigid occluder segment -- the same shape TileState/TILE_HALF_EXTENTS_M use
    (position, quaternion, half-extents), so render.py's existing per-object machinery
    (rendering, corner projection) applies unchanged to hand parts."""

    name: str
    center_m: Tuple[float, float, float]
    orientation_quat: Tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's convention
    half_extents_m: Tuple[float, float, float]


@dataclass(frozen=True)
class HandPose:
    """The small parameter set driving hand_parts() -- not 13 independent transforms.
    wrist_m/yaw_rad place the whole rig; curl/spread give per-frame pose variety within
    a single flat, palm-down repertoire (see module docstring)."""

    wrist_m: Tuple[float, float, float]
    yaw_rad: float  # rotation about world z; local +x (toward fingertips) points this way
    curl: float  # 0 = flat, 1 = _MAX_CURL_RAD -- applied to every finger's distal segment
    spread: float  # 0 = fingers at their base spacing, 1 = spaced _MAX_SPREAD_M further apart


def _quat_axis_angle(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate_vector(quat: np.ndarray, v: Tuple[float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation @ np.array(v, dtype=float)


def _digit_parts(
    name: str,
    base_local: Tuple[float, float, float],
    yaw_offset_rad: float,
    curl_rad: float,
    proximal_len: float,
    distal_len: float,
    half_extents: Tuple[float, float, float],
    yaw_quat: np.ndarray,
    wrist_m: Tuple[float, float, float],
) -> List[RigidPart]:
    """One finger/thumb's two segments, in the pose's local frame, then placed into world
    space by yaw_quat + wrist_m. A curl hinges the distal segment downward (toward the
    table) about the joint at the end of the proximal segment -- real forward kinematics,
    not a fixed offset, so curl actually shortens the digit's forward reach as it bends."""
    digit_yaw = _quat_axis_angle((0.0, 0.0, 1.0), yaw_offset_rad)
    proximal_local_center = np.array(base_local) + _rotate_vector(digit_yaw, (proximal_len / 2, 0.0, 0.0))
    joint_local = np.array(base_local) + _rotate_vector(digit_yaw, (proximal_len, 0.0, 0.0))
    curl_quat = _quat_multiply(digit_yaw, _quat_axis_angle((0.0, 1.0, 0.0), curl_rad))
    distal_local_center = joint_local + _rotate_vector(curl_quat, (distal_len / 2, 0.0, 0.0))

    def _to_world(local_center: np.ndarray, local_quat: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        world_center = np.array(wrist_m) + _rotate_vector(yaw_quat, tuple(local_center))
        world_quat = _quat_multiply(yaw_quat, local_quat)
        return tuple(float(c) for c in world_center), tuple(float(q) for q in world_quat)

    proximal_center, proximal_quat = _to_world(proximal_local_center, digit_yaw)
    distal_center, distal_quat = _to_world(distal_local_center, curl_quat)
    return [
        RigidPart(f"{name}_proximal", proximal_center, proximal_quat, half_extents),
        RigidPart(f"{name}_distal", distal_center, distal_quat, half_extents),
    ]


def hand_parts(pose: HandPose) -> List[RigidPart]:
    """Forward-kinematics a HandPose into 13 world-space rigid boxes: 2 forearm segments,
    1 palm, 2 thumb segments, 2 segments each for 4 fingers."""
    yaw_quat = _quat_axis_angle((0.0, 0.0, 1.0), pose.yaw_rad)
    curl_rad = pose.curl * _MAX_CURL_RAD
    spread_scale = 1.0 + pose.spread * (_MAX_SPREAD_M / max(abs(_FINGER_BASE_Y_OFFSETS_M[0]), 1e-9))

    def _to_world(local_center: Tuple[float, float, float]) -> Tuple[float, float, float]:
        world = np.array(pose.wrist_m) + _rotate_vector(yaw_quat, local_center)
        return tuple(float(c) for c in world)

    world_yaw_quat = tuple(float(c) for c in yaw_quat)

    parts: List[RigidPart] = []

    # Forearm: two segments trailing straight back (-x, local) from the wrist. Raised by
    # the difference in half-thickness vs. the palm so its UNDERSIDE lines up with the
    # palm's underside -- the forearm is thicker (bigger half-extent z), so keeping it at
    # the same center z as the palm would sink its bottom below the table/tile surface.
    forearm_z_offset = _FOREARM_HALF_EXTENTS_M[2] - _PALM_HALF_EXTENTS_M[2]
    for i in range(2):
        local_x = -(_FOREARM_SEGMENT_LEN_M / 2 + i * _FOREARM_SEGMENT_LEN_M)
        parts.append(
            RigidPart(f"forearm_{i}", _to_world((local_x, 0.0, forearm_z_offset)), world_yaw_quat, _FOREARM_HALF_EXTENTS_M)
        )

    # Palm: between the wrist and the finger bases.
    palm_local_x = _PALM_LEN_M / 2
    parts.append(RigidPart("palm", _to_world((palm_local_x, 0.0, 0.0)), world_yaw_quat, _PALM_HALF_EXTENTS_M))

    palm_front_x = _PALM_LEN_M

    # 4 fingers, laterally spaced across the palm's front edge.
    for i, base_y in enumerate(_FINGER_BASE_Y_OFFSETS_M):
        base_local = (palm_front_x, base_y * spread_scale, 0.0)
        parts.extend(
            _digit_parts(
                f"finger_{i}", base_local, 0.0, curl_rad,
                _FINGER_PROXIMAL_LEN_M, _FINGER_DISTAL_LEN_M, _FINGER_HALF_EXTENTS_M,
                yaw_quat, pose.wrist_m,
            )
        )

    # Thumb: angled off the near side of the palm, shorter, same curl amount.
    thumb_base_local = (palm_local_x * 0.6, -(_PALM_HALF_EXTENTS_M[1] + 0.01) * spread_scale, 0.0)
    parts.extend(
        _digit_parts(
            "thumb", thumb_base_local, _THUMB_YAW_RAD, curl_rad,
            _THUMB_PROXIMAL_LEN_M, _THUMB_DISTAL_LEN_M, _THUMB_HALF_EXTENTS_M,
            yaw_quat, pose.wrist_m,
        )
    )

    return parts


def place_hand(from_xy: Tuple[float, float], to_xy: Tuple[float, float], phase: float, rng: random.Random) -> HandPose:
    """A single reach-place-retract path: the wrist arcs from an off-table rest point
    (behind from_xy) to hover over to_xy and back. ``phase`` in [0, 1] samples a point
    along that path -- callers should draw from a late-phase band (released-through-
    retracting) so the pose is physically coherent with a tile already settled at to_xy
    (see trajectory.py), not still mid-grasp.

    This is Phase 1's one parameterized motion, deliberately not the full ~8-clip
    keyframed gesture library RESEARCH.md's fuller recommendation calls for -- see
    ROADMAP.md's deferred list."""
    from_arr, to_arr = np.array(from_xy), np.array(to_xy)
    direction = from_arr - to_arr
    norm = np.linalg.norm(direction)
    approach_dir = direction / norm if norm > 1e-9 else np.array([0.0, 1.0])
    rest_xy = to_arr + approach_dir * 0.25  # off-table-ish rest point behind the target

    # phase 0 -> at rest_xy (retracted); phase ~0.6 -> at to_xy (placing); phase 1 -> back at rest_xy.
    # A simple triangular path through the placement point keeps this a one-parameter
    # function rather than a spline, adequate for a single stereotyped motion.
    if phase <= 0.6:
        t = phase / 0.6
        wrist_xy = rest_xy + (to_arr - rest_xy) * t
    else:
        t = (phase - 0.6) / 0.4
        wrist_xy = to_arr + (rest_xy - to_arr) * t

    yaw_rad = float(np.arctan2(-approach_dir[1], -approach_dir[0]))  # fingers point toward to_xy
    wrist_height = _TILE_TOP_Z_M + _TABLE_CLEARANCE_M + _PALM_HALF_EXTENTS_M[2]
    wrist_m = (float(wrist_xy[0]), float(wrist_xy[1]), wrist_height)

    curl = 0.15 + 0.15 * rng.random()  # near-flat, small per-frame variation
    spread = rng.random()
    return HandPose(wrist_m=wrist_m, yaw_rad=yaw_rad, curl=curl, spread=spread)
