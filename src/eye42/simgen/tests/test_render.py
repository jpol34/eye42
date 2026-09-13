from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from eye42.engine.tiles import Tile
from eye42.simgen.hand import RigidPart
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


def test_photoreal_render_applies_independent_sensor_noise_per_call():
    """Two renders of the identical scene must not be byte-identical -- proving the
    post-render sensor-noise step (an unmeasured placeholder against an artificially
    noise-free render, see render.py's _SENSOR_NOISE_SIGMA_RANGE comment) actually runs
    and varies per call, rather than being dead code or a fixed pattern every frame
    would otherwise share."""
    bpy = pytest.importorskip("bpy")
    from eye42.simgen.render import render_photoreal

    camera = default_camera((160, 120))
    states = [_flat_tile(Tile.of(4, 4), 0.0, 0.0)]

    first = render_photoreal(states, camera, samples=8)
    second = render_photoreal(states, camera, samples=8)

    assert not np.array_equal(first, second)


def test_photoreal_render_noise_is_reproducible_given_the_same_rng():
    """gen_sim_dataset.py's --seed is meant to make a --photoreal dataset fully
    reproducible; the sensor-noise step must honor a caller-supplied rng rather than
    reaching into numpy's own global random state, or --seed would stop reproducing
    byte-identical images even though every other pipeline stage stays seed-driven."""
    bpy = pytest.importorskip("bpy")
    from eye42.simgen.render import render_photoreal
    import random

    camera = default_camera((160, 120))
    states = [_flat_tile(Tile.of(4, 4), 0.0, 0.0)]

    first = render_photoreal(states, camera, samples=8, rng=random.Random(42))
    second = render_photoreal(states, camera, samples=8, rng=random.Random(42))

    assert np.array_equal(first, second)


def _box_occluder(x: float, y: float, half_x: float, half_y: float, name: str = "occluder") -> RigidPart:
    """A flat box occluder centered above the table, for controlled occlusion tests --
    not a real hand.hand_parts() rig, just a simple probe shape."""
    return RigidPart(name, (x, y, 0.02), _IDENTITY_QUAT, (half_x, half_y, 0.005))


def test_an_occluder_directly_over_a_tile_reduces_its_visible_fraction():
    camera = default_camera((320, 320))
    tile = _flat_tile(Tile.of(6, 6), 0.0, 0.0)
    occluder = _box_occluder(0.0, 0.0, 0.05, 0.05)

    unoccluded_fraction = render_ground_truth([tile], camera)[0].visible_fraction
    occluded_fraction = render_ground_truth([tile], camera, occluders=[occluder])[0].visible_fraction

    assert unoccluded_fraction == pytest.approx(1.0, abs=0.02)
    assert occluded_fraction < 0.3


def test_no_occluders_argument_matches_the_default_empty_tuple():
    """render_ground_truth's occluders param must be purely additive -- every pre-
    existing call site (this file's other tests included) calls it with exactly 2
    positional args, so a call with occluders explicitly empty must match one without
    the argument at all, byte for byte."""
    camera = default_camera((320, 320))
    tile = _flat_tile(Tile.of(3, 2), 0.0, 0.0)

    without_arg = render_ground_truth([tile], camera)
    with_empty = render_ground_truth([tile], camera, occluders=())

    assert without_arg[0].visible_fraction == with_empty[0].visible_fraction
    assert np.array_equal(without_arg[0].visible_mask, with_empty[0].visible_mask)


def test_a_gap_between_two_occluders_leaves_the_tile_partially_visible():
    """Two separate occluder parts straddling a tile with a gap between them must NOT
    be treated as one merged blob -- the gap should survive as real visible tile pixels,
    the exact finger-gap-visibility behavior real footage shows (RESEARCH.md)."""
    camera = default_camera((320, 320))
    tile = _flat_tile(Tile.of(6, 6), 0.0, 0.0)
    left = _box_occluder(-0.03, 0.0, 0.008, 0.05, "left")
    right = _box_occluder(0.03, 0.0, 0.008, 0.05, "right")

    infos = render_ground_truth([tile], camera, occluders=[left, right])

    assert 0.1 < infos[0].visible_fraction < 0.9


def test_render_debug_preview_draws_occluders_not_just_tiles():
    """gen_sim_dataset.py uses render_debug_preview by default (--photoreal is opt-in)
    -- if it silently ignored occluders, the default-generated corpus would have
    ground-truth labels claiming hand occlusion the images never actually show."""
    camera = default_camera((320, 320))
    tile = _flat_tile(Tile.of(6, 6), 0.0, 0.0)
    occluder = _box_occluder(0.0, 0.0, 0.05, 0.05)

    without_occluder = render_debug_preview([tile], camera)
    with_occluder = render_debug_preview([tile], camera, occluders=[occluder])

    assert not np.array_equal(without_occluder, with_occluder)


def test_photoreal_render_shows_occluder_pixels_where_ground_truth_says_theyre_visible():
    """Mirrors the tile/ground-truth-agreement test above, but for an occluder: the
    pixel where render_ground_truth says a tile is occluded must NOT be tile-colored in
    the photoreal image (it should be the skin-colored occluder instead), proving the
    occluder participates in the real rendered image, not just the label computation."""
    bpy = pytest.importorskip("bpy")
    from eye42.simgen.render import render_photoreal

    # A tight, close-in camera (same style as the tile/ground-truth-agreement test's cam1
    # above) -- default_camera's wide table-framing view makes a single tile only a few
    # pixels across, too small to reliably sample a patch from.
    camera = Camera(image_size=(320, 320), focal_px=6000.0, position_m=(0.0, -0.08, 0.09), look_at_m=(0.0, 0.0, 0.01))
    tile = _flat_tile(Tile.of(6, 6), 0.0, 0.0)
    occluder = _box_occluder(0.0, 0.0, 0.05, 0.05)

    infos = render_ground_truth([tile], camera, occluders=[occluder])
    assert infos[0].visible_fraction < 0.3, "test setup assumption: the occluder should mostly cover the tile"

    tile_only = render_photoreal([tile], camera, samples=16)
    with_occluder = render_photoreal([tile], camera, samples=16, occluders=[occluder])

    # Sample at the OCCLUDER's own projected center, not the tile's -- under an oblique
    # camera, a point higher up (the occluder) projects to a different screen location
    # than a point at table level (the tile), even directly above it in world space.
    cx, cy = camera.project(occluder.center_m)
    cx, cy = int(cx), int(cy)
    tile_patch = tile_only[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
    occluded_patch = with_occluder[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
    assert not np.allclose(tile_patch.mean(axis=(0, 1)), occluded_patch.mean(axis=(0, 1)), atol=15)
