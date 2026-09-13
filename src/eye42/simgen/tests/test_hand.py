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


def test_hand_parts_stay_above_the_table_plane():
    for yaw in (0.0, 1.0, 3.0, -2.0):
        for curl in (0.0, 0.5, 1.0):
            pose = HandPose(wrist_m=(0.0, 0.0, 0.02), yaw_rad=yaw, curl=curl, spread=0.5)
            for part in hand_parts(pose):
                assert part.center_m[2] > 0.0, (yaw, curl, part.name)


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
    """No part should sit below a resting tile's top surface at the point the hand is
    reaching toward -- Phase 1 is kinematic (no contact physics), so this is a fixed-
    clearance geometric invariant, not a physically resolved contact."""
    pose = place_hand((0.0, 0.35), (0.0, 0.0), phase=0.6, rng=random.Random(0))
    tile_top_z = TILE_HALF_EXTENTS_M[2]

    for part in hand_parts(pose):
        assert part.center_m[2] - part.half_extents_m[2] >= tile_top_z - 1e-6, part.name
