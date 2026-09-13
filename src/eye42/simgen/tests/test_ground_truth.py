from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")

from eye42.engine.tiles import Tile
from eye42.simgen.ground_truth import frame_ground_truth_dict, write_frame_truth_jsonl, yolo_seg_lines
from eye42.simgen.physics import TileState
from eye42.simgen.render import default_camera, render_ground_truth
from eye42.simgen.trajectory import FrameSnapshot

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _snapshot() -> FrameSnapshot:
    states = (
        TileState(tile=Tile.of(6, 6), position=(-0.1, 0.0, 0.005), orientation_quat=_IDENTITY_QUAT),
        TileState(tile=Tile.of(2, 1), position=(0.1, 0.0, 0.005), orientation_quat=_IDENTITY_QUAT),
    )
    return FrameSnapshot(index=0, tile_states=states, seat_racks={0: (Tile.of(3, 3),), 1: ()})


def test_yolo_seg_lines_one_per_visible_tile_with_normalized_coords():
    camera = default_camera((320, 320))
    snapshot = _snapshot()
    infos = render_ground_truth(snapshot.tile_states, camera)

    lines = yolo_seg_lines(infos)

    assert len(lines) == 2
    for line in lines:
        parts = line.split()
        assert parts[0] == "0"
        coords = [float(v) for v in parts[1:]]
        assert all(0.0 <= v <= 1.0 for v in coords), "polygon coords must be normalized"


def test_fully_occluded_tile_gets_no_label_line():
    camera = default_camera((320, 320))
    states = (
        TileState(tile=Tile.of(1, 1), position=(0.0, 0.0, 0.005), orientation_quat=_IDENTITY_QUAT),
        TileState(tile=Tile.of(2, 2), position=(0.0, 0.0, 0.005), orientation_quat=_IDENTITY_QUAT),
    )
    infos = render_ground_truth(states, camera)

    lines = yolo_seg_lines(infos)

    # Two exactly-coincident, same-height tiles: the occlusion logic must pick a single
    # winner and gate the other's visible_fraction to 0 -- if it didn't (e.g. occlusion
    # were silently broken), both tiles would still show visible_fraction 1.0 and this
    # would emit 2 labels for a spot only one tile can actually be seen at.
    assert len(lines) <= 1


def test_frame_ground_truth_dict_records_identity_pose_and_racks():
    camera = default_camera((320, 320))
    snapshot = _snapshot()
    infos = render_ground_truth(snapshot.tile_states, camera)

    record = frame_ground_truth_dict(snapshot, infos)

    identities = {t["identity"] for t in record["tiles"]}
    assert identities == {"6-6", "2-1"}
    assert record["seat_racks"]["0"] == ["3-3"]
    assert record["seat_racks"]["1"] == []


def test_write_frame_truth_jsonl_writes_one_valid_json_line_per_frame(tmp_path):
    camera = default_camera((320, 320))
    frames = [_snapshot(), FrameSnapshot(index=1, tile_states=_snapshot().tile_states, seat_racks={0: (), 1: ()})]
    out = tmp_path / "truth" / "frames.jsonl"

    write_frame_truth_jsonl(frames, camera, out)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["frame"] for p in parsed] == [0, 1]
