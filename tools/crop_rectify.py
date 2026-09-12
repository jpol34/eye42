#!/usr/bin/env python3
"""Throwaway capture-spec validation script (see the project plan's "first task").

Grabs a single frame from a test clip, lets you click the 4 corners of the table
surface, rectifies it to a top-down view via a one-time homography, and saves
crops of a few tile-sized regions so you can eyeball pip legibility before
committing real game nights to a camera setup.

This is NOT the real Phase 2 pipeline -- it's a fast, disposable sanity check.

Usage:
    python tools/crop_rectify.py path/to/test_clip.mp4 [--frame-time SECONDS]
    python tools/crop_rectify.py path/to/test_clip.mp4 --corners x1,y1 x2,y2 x3,y3 x4,y4

Corner order (clicked or passed explicitly): top-left, top-right, bottom-right,
bottom-left of the table surface, in image pixel coordinates.

Output (written next to the input video):
    <name>_frame.png       the extracted source frame, with clicked corners drawn
    <name>_rectified.png   the top-down rectified table
    <name>_tile_crop_N.png a few sample crops from the rectified image, so you
                           can zoom in and check pip legibility directly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RECTIFIED_SIZE = (1400, 1400)  # output width, height in pixels -- square table
TILE_CROP_SIZE = 160  # pixels per side; a domino at this size should show clear pips


def grab_frame(video_path: Path, frame_time: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"error: could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"error: could not read a frame at t={frame_time}s -- try a different time")
    return frame


def pick_corners_interactively(frame: np.ndarray) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    window = "click 4 table corners: TL, TR, BR, BL -- press ENTER when done"
    display = frame.copy()

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(display, str(len(corners)), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_click)
    print("Click the table's 4 corners in order: top-left, top-right, bottom-right, bottom-left.")
    print("Press ENTER (or any key) once all 4 are placed.")
    while True:
        key = cv2.waitKey(50)
        if len(corners) == 4 and key != -1:
            break
        if key == 27:  # Esc
            sys.exit("cancelled")
    cv2.destroyAllWindows()
    return corners


def rectify(frame: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    src = np.array(corners, dtype=np.float32)
    w, h = RECTIFIED_SIZE
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h))


def sample_tile_crops(rectified: np.ndarray, out_stem: Path, n: int = 4) -> None:
    h, w = rectified.shape[:2]
    # A handful of spread-out sample points -- not meant to hit real tiles,
    # just to let you check pixel-level sharpness/pip legibility anywhere on
    # the rectified surface.
    points = [(w // 4, h // 4), (3 * w // 4, h // 4), (w // 2, h // 2), (w // 2, 3 * h // 4)]
    for i, (cx, cy) in enumerate(points[:n], start=1):
        half = TILE_CROP_SIZE // 2
        crop = rectified[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
        cv2.imwrite(str(out_stem.parent / f"{out_stem.name}_tile_crop_{i}.png"), crop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame-time", type=float, default=5.0, help="seconds into the clip to grab (default 5.0)")
    parser.add_argument(
        "--corners", nargs=4, metavar="x,y",
        help="4 corners as x,y x,y x,y x,y (TL TR BR BL) -- skips interactive picking",
    )
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"error: {args.video} not found")

    frame = grab_frame(args.video, args.frame_time)

    if args.corners:
        corners = [tuple(int(v) for v in c.split(",")) for c in args.corners]
    else:
        corners = pick_corners_interactively(frame)

    annotated = frame.copy()
    for i, (x, y) in enumerate(corners, start=1):
        cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(annotated, str(i), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    stem = args.video.parent / args.video.stem
    cv2.imwrite(f"{stem}_frame.png", annotated)

    rectified = rectify(frame, corners)
    cv2.imwrite(f"{stem}_rectified.png", rectified)
    sample_tile_crops(rectified, stem)

    print(f"Wrote {stem}_frame.png, {stem}_rectified.png, and sample tile crops.")
    print("Check: is the rectified table square/undistorted, and are the tile-crop")
    print("samples sharp enough to make out pips? If not, adjust camera position,")
    print("resolution, or lighting before recording real sessions.")


if __name__ == "__main__":
    main()
