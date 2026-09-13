#!/usr/bin/env python3
"""Calibrates a real domino table setup: picks the 4 corners of the play area
from a photo or video frame, rectifies it to a top-down view, and writes
``calibration.json`` for the live-session harness to load.

The 4 corners double as the 4 seats' known table-side positions (N/E/S/W),
so this is also the reference geometry player-attribution measures a played
tile's motion trail against.

Usage:
    python tools/calibrate.py path/to/photo.jpg --corners x1,y1 x2,y2 x3,y3 x4,y4
    python tools/calibrate.py path/to/clip.mp4 --interactive [--frame-time SECONDS]

--corners is the default, dependency-free way to run this: pass the 4 corners
directly. --interactive opens a click-to-pick GUI window instead, which needs
a GUI-capable OpenCV build; this project declares a headless build, so
--interactive will fail with a clear error unless a GUI-capable build is
installed separately.

Corner order (clicked or passed explicitly): top-left, top-right,
bottom-right, bottom-left of the play area, in source-image pixel
coordinates. These map to seats N, E, S, W respectively -- sit the camera so
that mapping matches the table's actual seating before calibrating.

Output:
    calibration.json (or --output PATH) -- corners (both pixel and per-seat),
    rectified output size, ready for TableRectifier to load.
    <name>_calibration_frame.png   the source frame, with corners drawn and
                                    labeled by seat
    <name>_calibration_rectified.png   the top-down rectified preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

RECTIFIED_SIZE = (1400, 1400)  # output width, height in pixels -- square table
SEAT_ORDER = ("N", "E", "S", "W")  # matches corner order: TL, TR, BR, BL


def grab_frame(path: Path, frame_time: float) -> np.ndarray:
    """A still image loads directly; anything else is treated as a video and
    a frame is grabbed at ``frame_time`` seconds in, same as crop_rectify.py."""
    image = cv2.imread(str(path))
    if image is not None:
        return image

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"error: could not open {path} as an image or video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"error: could not read a frame at t={frame_time}s -- try a different time")
    return frame


def pick_corners_interactively(frame: np.ndarray) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    window = "click 4 corners in seat order: N, E, S, W -- press ENTER when done"
    display = frame.copy()

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(display, SEAT_ORDER[len(corners) - 1], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_click)
    print("Click the 4 corners in seat order: N (top-left), E (top-right), S (bottom-right), W (bottom-left).")
    print("Press ENTER (or any key) once all 4 are placed.")
    while True:
        key = cv2.waitKey(50)
        if len(corners) == 4 and key != -1:
            break
        if key == 27:  # Esc
            sys.exit("cancelled")
    cv2.destroyAllWindows()
    return corners


def rectify(frame: np.ndarray, corners: list[tuple[int, int]], size: tuple[int, int] = RECTIFIED_SIZE) -> np.ndarray:
    src = np.array(corners, dtype=np.float32)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (w, h))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a photo or video of the table")
    parser.add_argument("--frame-time", type=float, default=5.0, help="seconds into a video to grab (default 5.0)")
    parser.add_argument(
        "--corners", nargs=4, metavar="x,y",
        help="4 corners as x,y x,y x,y x,y, in seat order N, E, S, W. Default way to supply corners.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help=(
            "Pick corners by clicking in a GUI window instead of passing --corners. "
            "Requires a GUI-capable OpenCV build; this project declares a headless one."
        ),
    )
    parser.add_argument(
        "--size", default=None, metavar="WIDTHxHEIGHT",
        help=f"rectified output size (default {RECTIFIED_SIZE[0]}x{RECTIFIED_SIZE[1]})",
    )
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"error: {args.source} not found")

    size = RECTIFIED_SIZE
    if args.size:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))

    frame = grab_frame(args.source, args.frame_time)

    if args.corners:
        corners = [tuple(int(v) for v in c.split(",")) for c in args.corners]
    elif args.interactive:
        try:
            corners = pick_corners_interactively(frame)
        except cv2.error as exc:
            sys.exit(
                "error: GUI corner-picking failed -- this project's declared "
                "dependency is a headless OpenCV build, which has no GUI support. "
                "Install a GUI-capable build to use --interactive, "
                "or pass --corners x,y x,y x,y x,y instead.\n"
                f"(underlying error: {exc})"
            )
    else:
        sys.exit(
            "error: pass --corners x,y x,y x,y x,y, or --interactive (needs a "
            "GUI-capable OpenCV build -- see --help)"
        )

    annotated = frame.copy()
    for seat, (x, y) in zip(SEAT_ORDER, corners):
        cv2.circle(annotated, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(annotated, seat, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    stem = args.source.parent / args.source.stem
    cv2.imwrite(f"{stem}_calibration_frame.png", annotated)

    rectified = rectify(frame, corners, size)
    cv2.imwrite(f"{stem}_calibration_rectified.png", rectified)

    calibration = {
        "corners": {seat: list(xy) for seat, xy in zip(SEAT_ORDER, corners)},
        "rectified_size": list(size),
    }
    args.output.write_text(json.dumps(calibration, indent=2))

    print(f"Wrote {args.output} -- the real table calibration for this camera setup.")
    print(f"Wrote {stem}_calibration_frame.png and {stem}_calibration_rectified.png for a visual check.")
    print("Check: is the rectified table square/undistorted? If not, re-run with corrected corners")
    print("before starting a real session -- this file is what TableRectifier and player attribution")
    print("both load their geometry from.")


if __name__ == "__main__":
    main()
