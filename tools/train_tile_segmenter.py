#!/usr/bin/env python3
"""Trains a YOLOv8-seg model on a synthetic touching-tile dataset (see
tools/gen_synthetic_tiles.py) and exports it to ONNX for
eye42.perception.tile_segment's live-runtime inference path.

Training-only tooling: needs the perception-training extra (ultralytics,
which pulls in torch) -- not required for a live session, which only needs
onnxruntime to run the exported model (see the perception-segmenter extra).

Usage:
    python tools/train_tile_segmenter.py --data synth_data/data.yaml --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def train(data_yaml: Path, epochs: int, model_size: str, output_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "error: ultralytics is not installed. Install the training extra first:\n"
            "  pip install -e .[perception-training]"
        )

    model = YOLO(f"yolov8{model_size}-seg.pt")
    model.train(data=str(data_yaml), epochs=epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = model.export(format="onnx")
    Path(exported).replace(output_path)
    print(f"Trained and exported model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="path to a data.yaml (from gen_synthetic_tiles.py)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8-seg size (default n, nano)")
    parser.add_argument("--output", type=Path, default=Path("models/tile_segmenter.onnx"))
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f"error: {args.data} not found")

    train(args.data, args.epochs, args.model_size, args.output)


if __name__ == "__main__":
    main()
