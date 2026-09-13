from __future__ import annotations

import random

import numpy as np
import pytest

from eye42.simgen.hand import HandPose, hand_parts, place_hand
from eye42.simgen.tile_geometry import TILE_HALF_EXTENTS_M

_EXPECTED_PART_NAMES = {
    "forearm_0", "forearm_1", "palm",
    "thumb_proximal", "thumb_distal",
    "finger_0_proximal", "finger_0_distal",
    "finger_1_proximal", "finger_1_distal",
    "finger_2_proximal", "finger_2_distal",
    "finger_3_proximal", "finger_3_distal",
}


def test_hand_parts_returns_thirteen_named_rigid_boxes():
    pose = HandPose(wrist_m=(0.0, 0.0, 0.02), yaw_rad=0.0, curl=0.2, spread=0.5)

    parts = hand_parts(pose)

    assert {p.name for p in parts} == _EXPECTED_PART_NAMES
    assert len(parts) == 13


def test_hand_parts_orientation_quaternions_are_unit_length():
    pose = HandPose(wrist_m=(0.1, -0.2, 0.02), yaw_rad=1.3, curl=0.7, spread=0.9)

    for part in hand_parts(pose):
        assert np.linalg.norm(part.orientation_quat) == pytest.approx(1.0, abs=1e-6)


def _lowest_corner_z(part) -> float:
    """The actual lowest point of a rotated box, not just its center -- a curled
    fingertip's center can stay comfortably positive while its corner dips underground
    (this is exactly what an earlier, center-only version of this test failed to catch:
    hand_parts()'s own floor clamp exists specifically to prevent this)."""
    hx, hy, hz = part.half_extents_m
    local_corners = [
        (sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    w, x, y, z = part.orientation_quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    world_zs = [part.center_m[2] + (rotation @ np.array(c))[2] for c in local_corners]
    return min(world_zs)


def test_hand_parts_stay_above_the_table_plane():
    """Checks every part's actual lowest CORNER, not its center -- across the full
    curl/spread/yaw domain HandPose allows, not just the narrower range place_hand()
    itself happens to emit."""
    tile_top_z = TILE_HALF_EXTENTS_M[2]
    for yaw in (0.0, 1.0, 3.0, -2.0):
        for curl in (0.0, 0.5, 1.0):
            for spread in (0.0, 0.5, 1.0):
                pose = HandPose(wrist_m=(0.0, 0.0, 0.02), yaw_rad=yaw, curl=curl, spread=spread)
                for part in hand_parts(pose):
                    assert _lowest_corner_z(part) >= tile_top_z - 1e-6, (yaw, curl, spread, part.name)


def test_curling_a_finger_shortens_its_forward_reach():
    """A curled finger's distal segment hinges toward the table, not just tilts in
    place -- its projected forward reach from the wrist should shrink as curl
    increases, proving hand_parts does real forward kinematics, not a fixed offset."""
    flat = HandPose(wrist_m=(0.0, 0.0, 0.05), yaw_rad=0.0, curl=0.0, spread=0.0)
    curled = HandPose(wrist_m=(0.0, 0.0, 0.05), yaw_rad=0.0, curl=1.0, spread=0.0)

    flat_tip = next(p for p in hand_parts(flat) if p.name == "finger_1_distal")
    curled_tip = next(p for p in hand_parts(curled) if p.name == "finger_1_distal")

    assert curled_tip.center_m[0] < flat_tip.center_m[0]


def test_place_hand_reaches_toward_the_target_at_full_placement_phase():
    """At phase 0.6 (place_hand's placement point, per its docstring), the wrist should
    sit close to to_xy, not still near the rest point behind from_xy."""
    from_xy, to_xy = (0.0, 0.35), (0.0, 0.0)

    pose = place_hand(from_xy, to_xy, phase=0.6, rng=random.Random(0))

    wrist_xy = np.array(pose.wrist_m[:2])
    assert np.linalg.norm(wrist_xy - np.array(to_xy)) < 0.02


def test_place_hand_yaw_points_from_the_rest_side_toward_the_target():
    """The rig should face toward to_xy (fingers leading), not away from it -- reaching
    from a rack at +y toward the trick area at the origin means yaw should point roughly
    in the -y direction."""
    pose = place_hand((0.0, 0.35), (0.0, 0.0), phase=0.6, rng=random.Random(0))

    forward = np.array([np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)])
    assert forward[1] < 0  # points toward -y, i.e. toward to_xy from a +y rest side


def test_hand_parts_never_interpenetrate_a_settled_tile_at_the_target():
    """No part's actual lowest corner should sit below a resting tile's top surface at
    the point the hand is reaching toward, across the whole phase range place_hand()
    emits -- Phase 1 is kinematic (no contact physics), so this is a fixed-clearance
    geometric invariant, not a physically resolved contact."""
    tile_top_z = TILE_HALF_EXTENTS_M[2]
    for phase in (0.0, 0.3, 0.6, 0.8, 1.0):
        pose = place_hand((0.0, 0.35), (0.0, 0.0), phase=phase, rng=random.Random(0))
        for part in hand_parts(pose):
            assert _lowest_corner_z(part) >= tile_top_z - 1e-6, (phase, part.name)


def _palm_thumb_gap(spread: float) -> float:
    pose = HandPose(wrist_m=(0.0, 0.0, 0.05), yaw_rad=0.0, curl=0.0, spread=spread)
    parts = {p.name: p for p in hand_parts(pose)}
    palm, thumb = parts["palm"], parts["thumb_proximal"]
    palm_near_edge_y = palm.center_m[1] - palm.half_extents_m[1]
    thumb_near_corner_y = thumb.center_m[1] + thumb.half_extents_m[1]
    return palm_near_edge_y - thumb_near_corner_y


def test_thumb_stays_attached_to_the_palm_across_the_spread_range():
    """A previous version scaled the thumb's whole base offset (which already included
    the palm's half-width) by spread, moving the thumb's ANCHOR away from the palm
    instead of just its spacing -- the gap grew from ~0.4cm at spread 0 to ~2.4cm at
    spread 1, a visibly detached, floating thumb. Only the small extra spacing term
    should scale with spread now, so the gap's growth across the full range should stay
    within that term's own magnitude (_MAX_SPREAD_M's ballpark), not blow up."""
    gap_at_zero = _palm_thumb_gap(0.0)
    gap_at_full = _palm_thumb_gap(1.0)

    assert gap_at_full - gap_at_zero < 0.015, (gap_at_zero, gap_at_full)


def test_place_hands_rest_point_lies_beyond_from_xy_not_between_it_and_to_xy():
    """At phase 0 (fully retracted), the wrist should be farther from to_xy than
    from_xy itself is -- i.e. genuinely beyond the rack, not landing partway between
    the rack and the trick area (which an earlier fixed-distance version of this could
    do whenever |from-to| was smaller than that fixed distance)."""
    from_xy, to_xy = (0.0, 0.35), (0.0, 0.0)

    pose = place_hand(from_xy, to_xy, phase=0.0, rng=random.Random(0))

    wrist_dist = np.linalg.norm(np.array(pose.wrist_m[:2]) - np.array(to_xy))
    from_dist = np.linalg.norm(np.array(from_xy) - np.array(to_xy))
    assert wrist_dist > from_dist
