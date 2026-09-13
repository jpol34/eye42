from __future__ import annotations

import random

import numpy as np

from eye42.perception.synth_data import composite_scene, pip_layout_fractions, render_tile


def test_render_tile_produces_the_expected_crop_shape():
    tile = render_tile(3, 5, half_size=(80, 60))
    assert tile.shape == (120, 80, 3)


def test_pip_layout_fractions_reproduce_render_tiles_pixel_positions():
    """render_tile's pixel-space pip placement is built on pip_layout_fractions
    (see synth_data.py) -- this pins that the two agree exactly, since
    render_tile's own tests only check aggregate white-pixel counts, not
    per-dot position, and wouldn't catch the fraction table silently drifting
    from the pixel-space layout it's meant to be the single source of truth for.
    Cross-checks by rendering a single-count tile and confirming a white dot
    lands at each position pip_layout_fractions implies."""
    half_size = (80, 60)
    w, h = half_size
    for count in range(7):
        fractions = pip_layout_fractions(count)
        for u, v in fractions:
            assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
        px_positions = [(int(u * w), int(v * h)) for u, v in fractions]
        crop = render_tile(count, 0, half_size=half_size)
        top_half = crop[:h, :, :]
        for px, py in px_positions:
            assert tuple(top_half[py, px]) == (255, 255, 255), f"expected a pip dot center at {(px, py)} for count={count}"


def test_render_tile_pip_counts_are_visually_distinguishable():
    blank = render_tile(0, 0)
    six_six = render_tile(6, 6)
    # A blank half has no white pixels beyond the pure-green body; a 6-6
    # tile's pip dots add a measurable amount of white.
    assert (six_six == 255).sum() > (blank == 255).sum()


def test_composite_scene_produces_one_instance_per_requested_tile():
    background = np.full((300, 300, 3), 230, dtype=np.uint8)
    rng = random.Random(0)

    scene, instances = composite_scene(background, num_tiles=4, rng=rng)

    assert scene.shape == background.shape
    assert len(instances) == 4


def test_composite_scene_default_placement_keeps_overlap_small():
    background = np.full((300, 300, 3), 230, dtype=np.uint8)
    rng = random.Random(1)

    _, instances = composite_scene(background, num_tiles=3, rng=rng, spread=60)

    for i, a in enumerate(instances):
        for b in instances[i + 1:]:
            overlap = int((a.mask & b.mask).sum())
            assert overlap <= 500


def test_composite_scene_allow_overlap_can_exceed_the_default_limit():
    background = np.full((300, 300, 3), 230, dtype=np.uint8)
    rng = random.Random(2)

    # Tightly clustered with overlap explicitly allowed -- some pair should
    # end up overlapping more than the default no-overlap ceiling.
    _, instances = composite_scene(background, num_tiles=5, rng=rng, spread=5, allow_overlap=True)

    overlaps = [
        int((a.mask & b.mask).sum())
        for i, a in enumerate(instances)
        for b in instances[i + 1:]
    ]
    assert max(overlaps) > 500
