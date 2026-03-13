#!/usr/bin/env python3
"""
Step 1.2 – Simple OpenCV annotation tool for eye bounding boxes.

Opens each unannotated cattle image and lets the user draw a rectangle
around the eye region.  Saves YOLO-format .txt label files.

Controls:
    Click + drag   Draw a bounding box
    s              Save current box and advance to next image
    r              Redo – clear the current box
    n              Skip image (no annotation saved)
    q              Quit (progress is saved; resume later)

Usage:
    python -m src.step1.annotate_eyes [--class_filter healthy|pinkeye|all]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

from src.step1.config import (
    ANNOTATIONS_DIR,
    ANNOTATION_RULE,
    CLASS_NAMES,
    DETECTION_CLASS_ID,
    get_source_images,
)

# ── Globals for mouse callback ────────────────────────────────────────────────
drawing = False
ix, iy = -1, -1
box: tuple[int, int, int, int] | None = None
img_display: np.ndarray | None = None
img_clean: np.ndarray | None = None

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

_INSTRUCTION_LINES = [
    "ANNOTATION RULE",
    " - Draw ONE box around the visible eye region.",
    " - Include the eyeball/cornea + small surrounding eyelid tissue.",
    " - Do NOT include excessive background, ears, or hair.",
    " - Keep style consistent across healthy and pinkeye images.",
]


def _mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, box, img_display, img_clean

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img_clean.copy()
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            box = (x1, y1, x2, y2)
            img_display = img_clean.copy()
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)


def _save_yolo_label(label_path: Path, box_xyxy: tuple, img_w: int, img_h: int,
                     scale: float) -> None:
    """Convert display-space xyxy box to normalised YOLO format and save."""
    x1, y1, x2, y2 = box_xyxy
    x1 = x1 / scale
    y1 = y1 / scale
    x2 = x2 / scale
    y2 = y2 / scale

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write(f"{DETECTION_CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def _already_annotated(image_path: Path) -> bool:
    label_path = ANNOTATIONS_DIR / (image_path.stem + ".txt")
    return label_path.exists()


def _overlay_instructions(img: np.ndarray) -> None:
    """Burn a small instruction block into the bottom-left of the image."""
    y0 = img.shape[0] - len(_INSTRUCTION_LINES) * 20 - 10
    for i, line in enumerate(_INSTRUCTION_LINES):
        y = y0 + i * 20
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                    cv2.LINE_AA)


def annotate(filter_class: str = "all") -> None:
    global img_display, img_clean, box

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Print annotation rule to terminal at session start
    print()
    print(ANNOTATION_RULE)
    print()

    all_images = get_source_images()
    if filter_class != "all":
        all_images = [(p, c) for p, c in all_images if c == filter_class]

    pending = [(p, c) for p, c in all_images if not _already_annotated(p)]
    done_count = len(all_images) - len(pending)

    print(f"Total images: {len(all_images)}  |  Already annotated: {done_count}  |  Remaining: {len(pending)}")
    if not pending:
        print("All images are already annotated.")
        return

    win = "Annotate Eye – draw box then press 's' to save"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, _mouse_cb)

    for idx, (img_path, cls_label) in enumerate(pending):
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            print(f"  [SKIP] Cannot read {img_path}")
            continue

        orig_h, orig_w = img_orig.shape[:2]
        scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
        if scale < 1.0:
            disp = cv2.resize(img_orig, None, fx=scale, fy=scale)
        else:
            disp = img_orig.copy()

        header = (f"[{done_count + idx + 1}/{len(all_images)}] "
                  f"{cls_label}: {img_path.name}  "
                  f"(s=save  r=redo  n=skip  q=quit)")
        img_clean = disp.copy()
        cv2.putText(img_clean, header, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        _overlay_instructions(img_clean)
        img_display = img_clean.copy()
        box = None

        while True:
            cv2.imshow(win, img_display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("Quit. Progress saved – run again to resume.")
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                print(f"  [SKIP] {img_path.name}")
                break
            elif key == ord("r"):
                box = None
                img_display = img_clean.copy()
            elif key == ord("s"):
                if box is None:
                    print("  No box drawn yet – draw a box first, or press 'n' to skip.")
                    continue
                label_path = ANNOTATIONS_DIR / (img_path.stem + ".txt")
                _save_yolo_label(label_path, box, orig_w, orig_h, scale)
                print(f"  [SAVED] {label_path.name}  box={box}")
                break

    cv2.destroyAllWindows()
    final_count = sum(1 for _ in ANNOTATIONS_DIR.glob("*.txt"))
    print(f"\nAnnotation session complete.  Total annotations: {final_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate eye bounding boxes")
    parser.add_argument("--class_filter", default="all",
                        choices=CLASS_NAMES + ["all"],
                        help="Only annotate images from this class")
    args = parser.parse_args()
    annotate(filter_class=args.class_filter)


if __name__ == "__main__":
    main()
