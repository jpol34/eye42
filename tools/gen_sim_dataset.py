#!/usr/bin/env python3
"""Generates a training/validation dataset from eye42.simgen's 3D physics simulation:
either static physics-settled tile scenes, or full director-scripted 7-trick hands
rendered frame-by-frame, in YOLO segmentation format (matching what
tools/gen_synthetic_tiles.py produces) plus a richer per-frame JSONL ground-truth
sidecar (see eye42.simgen.ground_truth).

Output layout:
    <output-dir>/images/train/*.jpg, images/val/*.jpg
    <output-dir>/labels/train/*.txt, labels/val/*.txt   (YOLO-seg polygons, class 0 = "tile")
    <output-dir>/truth/frames.jsonl                     (rich per-frame ground truth)
    <output-dir>/data.yaml

Usage:
    python tools/gen_sim_dataset.py --scenes 200 --output-dir sim_data
    python tools/gen_sim_dataset.py --hands 5 --output-dir sim_data --photoreal
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eye42.engine.tiles import full_set  # noqa: E402
from eye42.simgen.director import script_one_hand  # noqa: E402
from eye42.simgen.ground_truth import write_frame_truth_jsonl, yolo_seg_lines  # noqa: E402
from eye42.simgen.physics import TileSimulation  # noqa: E402
from eye42.simgen.render import default_camera, render_debug_preview, render_ground_truth  # noqa: E402
from eye42.simgen.trajectory import FrameSnapshot, simulate_hand  # noqa: E402

_IMAGE_SIZE = (640, 640)


def _render_frame(frame: FrameSnapshot, camera, photoreal: bool):
    if photoreal:
        from eye42.simgen.render import render_photoreal

        return render_photoreal(frame.tile_states, camera)
    return render_debug_preview(frame.tile_states, camera)


def _write_frame(frame: FrameSnapshot, camera, photoreal: bool, image_path: Path, label_path: Path) -> None:
    image = _render_frame(frame, camera, photoreal)
    infos = render_ground_truth(frame.tile_states, camera)
    cv2.imwrite(str(image_path), image)
    label_path.write_text("\n".join(yolo_seg_lines(infos)))


def _write_dataset_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: tile\n"
    )


def generate(
    output_dir: Path,
    num_scenes: int,
    num_hands: int,
    val_fraction: float,
    photoreal: bool,
    seed: int,
) -> None:
    rng = random.Random(seed)
    for split_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / split_dir).mkdir(parents=True, exist_ok=True)

    camera = default_camera(_IMAGE_SIZE)
    all_frames: list[FrameSnapshot] = []
    scene_index = 0

    for i in range(num_scenes):
        split = "val" if i < num_scenes * val_fraction else "train"
        tiles = rng.sample(sorted(full_set()), k=rng.randint(2, 8))
        sim = TileSimulation(tiles, seed=seed + i)
        states = sim.drop_and_settle(spread=0.15)
        frame = FrameSnapshot(index=scene_index, tile_states=tuple(states), seat_racks={})
        all_frames.append(frame)
        _write_frame(
            frame, camera, photoreal,
            output_dir / "images" / split / f"scene_{scene_index:05d}.jpg",
            output_dir / "labels" / split / f"scene_{scene_index:05d}.txt",
        )
        scene_index += 1

    for h in range(num_hands):
        intents = script_one_hand(random.Random(seed + 1000 + h))
        for frame in simulate_hand(intents, seed=seed + 1000 + h):
            split = "val" if scene_index % 10 == 0 else "train"
            all_frames.append(frame)
            _write_frame(
                frame, camera, photoreal,
                output_dir / "images" / split / f"hand{h:03d}_frame_{frame.index:05d}.jpg",
                output_dir / "labels" / split / f"hand{h:03d}_frame_{frame.index:05d}.txt",
            )
            scene_index += 1

    write_frame_truth_jsonl(all_frames, camera, output_dir / "truth" / "frames.jsonl")
    _write_dataset_yaml(output_dir)
    print(f"Wrote {scene_index} frames to {output_dir}")
    print(f"Dataset config: {output_dir / 'data.yaml'}")
    print(f"Rich ground truth: {output_dir / 'truth' / 'frames.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=0, help="number of static physics-settled scenes")
    parser.add_argument("--hands", type=int, default=0, help="number of full scripted 7-trick hands to render")
    parser.add_argument("--output-dir", type=Path, default=Path("sim_data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--photoreal", action="store_true",
        help="render via headless Blender/Cycles (requires the sim-render extra) instead of the flat debug preview",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.scenes == 0 and args.hands == 0:
        sys.exit("error: specify --scenes and/or --hands (both default to 0)")

    generate(
        output_dir=args.output_dir,
        num_scenes=args.scenes,
        num_hands=args.hands,
        val_fraction=args.val_fraction,
        photoreal=args.photoreal,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
