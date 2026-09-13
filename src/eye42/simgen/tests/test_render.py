from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from eye42.engine.tiles import Tile
from eye42.simgen.physics import TileState
from eye42.simgen.render import Camera, default_camera, render_debug_preview, render_ground_truth

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _flat_tile(tile: Tile, x: float, y: float, z: float = 0.005) -> TileState:
    return TileState(tile=tile, position=(x, y, z), orientation_quat=_IDENTITY_QUAT)


def test_a_lone_tile_is_fully_visible():
    camera = default_camera((320, 320))
    states = [_flat_tile(Tile.of(6, 6), 0.0, 0.0)]

    infos = render_ground_truth(states, camera)

    assert len(infos) == 1
    assert infos[0].visible_fraction == pytest.approx(1.0, abs=0.02)
    assert infos[0].visible_mask.sum() > 0


def test_a_nearer_tile_occludes_a_farther_one_at_the_same_xy():
    """The exact scenario synth_data.py's docstring documents as broken (an occluded
    tile's mask isn't corrected for what a tile on top of it covers) -- here it must be
    correct, because occlusion order comes from real camera depth, not paste order."""
    camera = default_camera((320, 320))
    far_tile = TileState(tile=Tile.of(1, 1), position=(0.0, 0.0, 0.005), orientation_quat=_IDENTITY_QUAT)
    near_tile = TileState(tile=Tile.of(2, 2), position=(0.0, 0.0, 0.015), orientation_quat=_IDENTITY_QUAT)

    infos = render_ground_truth([far_tile, near_tile], camera)
    by_tile = {info.tile: info for info in infos}

    assert by_tile[Tile.of(2, 2)].visible_fraction == pytest.approx(1.0, abs=0.02)
    assert by_tile[Tile.of(1, 1)].visible_fraction < 0.5, "the covered tile must show reduced visibility"
    # No pixel may be claimed by both tiles -- masks must partition the shared region.
    overlap = by_tile[Tile.of(1, 1)].visible_mask & by_tile[Tile.of(2, 2)].visible_mask
    assert not overlap.any()


def test_widely_separated_tiles_are_each_fully_visible():
    camera = default_camera((640, 640))
    states = [_flat_tile(Tile.of(0, 0), -0.15, -0.15), _flat_tile(Tile.of(6, 0), 0.15, 0.15)]

    infos = render_ground_truth(states, camera)

    assert all(info.visible_fraction == pytest.approx(1.0, abs=0.02) for info in infos)


def test_debug_preview_produces_a_correctly_shaped_bgr_image():
    camera = default_camera((200, 150))
    states = [_flat_tile(Tile.of(3, 1), 0.0, 0.0)]

    image = render_debug_preview(states, camera)

    assert image.shape == (150, 200, 3)
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    "cam",
    [
        default_camera((200, 200)),
        # A deliberately non-default placement -- the exact case an earlier version
        # (Blender's own to_track_quat heuristic vs. this module's hand-rolled
        # Gram-Schmidt basis, computed independently) would have silently misaligned,
        # since the two only coincidentally agreed for one hardcoded camera before.
        Camera(image_size=(200, 200), focal_px=350.0, position_m=(0.4, -1.2, 0.6), look_at_m=(0.1, 0.05, 0.0)),
    ],
)
def test_ground_truth_projection_agrees_with_the_photoreal_render(cam):
    """render_ground_truth's projected tile position and render_photoreal's actual
    rendered pixels must land in the same place -- otherwise YOLO labels computed from
    one wouldn't correspond to the image produced by the other, silently."""
    pytest.importorskip("bpy")
    from eye42.simgen.render import render_photoreal

    tile = Tile.of(5, 5)
    states = [_flat_tile(tile, 0.05, -0.03)]

    infos = render_ground_truth(states, cam)
    image = render_photoreal(states, cam, samples=8)

    px, py = infos[0].polygon_px[0]  # one corner of the projected footprint
    w, h = cam.image_size
    assert 0 <= px < w and 0 <= py < h, "the projected corner should land inside the frame"
    # Sample a small patch around the projected tile center (average of its corners)
    # and confirm it's not the flat gray background -- i.e. the tile is actually there.
    cx = int(sum(p[0] for p in infos[0].polygon_px) / 4)
    cy = int(sum(p[1] for p in infos[0].polygon_px) / 4)
    patch = image[max(0, cy - 3) : cy + 3, max(0, cx - 3) : cx + 3]
    background_bgr = image[2, 2]  # a corner pixel, guaranteed background
    assert not np.allclose(patch.mean(axis=(0, 1)), background_bgr, atol=15), (
        "the ground-truth-projected tile center should land on the rendered tile, not the background"
    )


def test_photoreal_render_produces_a_real_image():
    bpy = pytest.importorskip("bpy")
    from eye42.simgen.render import render_photoreal

    camera = default_camera((160, 120))
    states = [_flat_tile(Tile.of(4, 4), 0.0, 0.0), _flat_tile(Tile.of(1, 0), 0.15, 0.1)]

    image = render_photoreal(states, camera, samples=8)

    assert image is not None
    assert image.shape == (120, 160, 3)
    assert image.std() > 0, "a real render should have visual variation, not a flat blank frame"
