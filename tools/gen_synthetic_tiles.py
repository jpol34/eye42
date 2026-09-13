#!/usr/bin/env python3
"""Generates a synthetic training set of composited touching-tile scenes for
tools/train_tile_segmenter.py, in YOLO segmentation format.

Renders individual domino tiles (see eye42.perception.synth_data) and
composites them onto backgrounds -- plain color fills by default, or real
photo backgrounds (including hand photos, as hard negatives) if
--backgrounds-dir is given -- writing each scene's image and per-tile
polygon labels.

Output layout (matches what ultralytics' YOLO training expects):
    <output-dir>/images/train/*.jpg, images/val/*.jpg
    <output-dir>/labels/train/*.txt, labels/val/*.txt   (one line per tile:
        "0 x1 y1 x2 y2 ..." -- normalized polygon points, class 0 = "tile")
    <output-dir>/data.yaml

Usage:
    python tools/gen_synthetic_tiles.py --count 2000 --output-dir synth_data
    python tools/gen_synthetic_tiles.py --count 500 --backgrounds-dir path/to/real/frames
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eye42.perception.synth_data import composite_scene  # noqa: E402

_SCENE_SIZE = (640, 640)
_PLAIN_BACKGROUND_COLOR = (235, 235, 235)  # off-white, close to the real table's color


def _polygon_from_mask(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Returns the largest contour in mask as a normalized polygon, or None
    if the mask is empty (can happen if a tile was placed fully outside the
    scene bounds)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    h, w = mask.shape
    return [(float(x) / w, float(y) / h) for x, y in largest.reshape(-1, 2)]


def _load_backgrounds(backgrounds_dir: Path | None, size: tuple[int, int]) -> list[np.ndarray]:
    if backgrounds_dir is None:
        return [np.full((size[1], size[0], 3), _PLAIN_BACKGROUND_COLOR, dtype=np.uint8)]
    images = []
    for path in sorted(backgrounds_dir.glob("*.jpg")) + sorted(backgrounds_dir.glob("*.png")):
        img = cv2.imread(str(path))
        if img is not None:
            images.append(cv2.resize(img, size))
    if not images:
        sys.exit(f"error: no .jpg/.png images found in {backgrounds_dir}")
    return images


def _write_dataset_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: tile\n"
    )


def generate(
    count: int,
    output_dir: Path,
    backgrounds_dir: Path | None,
    val_fraction: float,
    min_tiles: int,
    max_tiles: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    backgrounds = _load_backgrounds(backgrounds_dir, _SCENE_SIZE)

    for split_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / split_dir).mkdir(parents=True, exist_ok=True)

    num_val = max(1, int(count * val_fraction))
    for i in range(count):
        split = "val" if i < num_val else "train"
        background = rng.choice(backgrounds)
        num_tiles = rng.randint(min_tiles, max_tiles)
        margin = 100
        w, h = _SCENE_SIZE
        cluster_center = (rng.randint(margin, w - margin), rng.randint(margin, h - margin))

        scene, instances = composite_scene(background, num_tiles, rng, cluster_center=cluster_center, spread=50)

        lines = []
        for inst in instances:
            polygon = _polygon_from_mask(inst.mask)
            if polygon is None:
                continue
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            lines.append(f"0 {coords}")

        image_path = output_dir / "images" / split / f"scene_{i:05d}.jpg"
        label_path = output_dir / "labels" / split / f"scene_{i:05d}.txt"
        cv2.imwrite(str(image_path), scene)
        label_path.write_text("\n".join(lines))

    _write_dataset_yaml(output_dir)
    print(f"Wrote {count} scenes ({num_val} val, {count - num_val} train) to {output_dir}")
    print(f"Dataset config: {output_dir / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=2000, help="number of scenes to generate (default 2000)")
    parser.add_argument("--output-dir", type=Path, default=Path("synth_data"))
    parser.add_argument(
        "--backgrounds-dir", type=Path, default=None,
        help="directory of real .jpg/.png photos to composite onto (including hand photos, as hard negatives); "
             "defaults to a plain off-white background if omitted",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-tiles", type=int, default=2)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(
        count=args.count,
        output_dir=args.output_dir,
        backgrounds_dir=args.backgrounds_dir,
        val_fraction=args.val_fraction,
        min_tiles=args.min_tiles,
        max_tiles=args.max_tiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
