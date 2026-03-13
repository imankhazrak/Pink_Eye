#!/usr/bin/env python3
"""
Step 1.7 – Run YOLO inference on every source image (or use ground-truth
annotations), crop the eye region with padding, and save results with
the original disease label.

Two crop modes
--------------
  --source predicted   (default) Use the trained YOLO detector.
                       Output goes to ``outputs/eye_crops_pred/``.
  --source ground_truth Use the manual YOLO annotation files.
                        Output goes to ``outputs/eye_crops_gt/``.

Generating both datasets enables the Step 2 ablation study that
compares classification on perfect/manual crops vs detector crops.

Confidence threshold
--------------------
The default threshold is set in config.py (0.25).  Override with
``--conf``.  To sweep multiple thresholds at once, pass several
values separated by commas:

    python -m src.step1.crop_eyes --conf 0.15,0.25,0.40

Each threshold produces its own metadata CSV so results are
directly comparable.

Usage:
    python -m src.step1.crop_eyes
    python -m src.step1.crop_eyes --source ground_truth
    python -m src.step1.crop_eyes --source predicted --conf 0.25 --padding 0.10
    python -m src.step1.crop_eyes --conf 0.15,0.25,0.40
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from src.step1.config import (
    ANNOTATIONS_DIR,
    CONFIDENCE_THRESHOLD,
    CROP_PADDING_FRACTION,
    DETECTION_CLASS_ID,
    DETECTOR_WEIGHTS_DIR,
    EYE_CROPS_GT_DIR,
    EYE_CROPS_PRED_DIR,
    METADATA_DIR,
    ensure_dirs,
    find_source_image,
    get_source_images,
)


# ── Box helpers ───────────────────────────────────────────────────────────────

def _select_best_box(results) -> tuple | None:
    """Pick the highest-confidence predicted box.
    Returns (x1, y1, x2, y2, conf, num_boxes) or None."""
    best = None
    total = 0
    for r in results:
        for box in r.boxes:
            total += 1
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            if best is None or conf > best[4]:
                best = (int(coords[0]), int(coords[1]),
                        int(coords[2]), int(coords[3]), conf)
    if best is None:
        return None
    return (*best, total)


def _parse_gt_label(label_path: Path, img_w: int, img_h: int):
    """Parse a YOLO-format annotation file into absolute xyxy boxes.
    Returns list of (x1, y1, x2, y2)."""
    boxes = []
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        if cls_id != DETECTION_CLASS_ID:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _crop_with_padding(img, x1, y1, x2, y2, pad_frac: float):
    """Crop a region with fractional padding, clipped to image bounds."""
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        return None, (cx1, cy1, cx2, cy2)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


# ── Predicted-box cropping ────────────────────────────────────────────────────

def _crop_predicted(padding: float, conf_thresh: float) -> None:
    weights = DETECTOR_WEIGHTS_DIR / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights}. "
            "Run train_detector.py first."
        )

    model = YOLO(str(weights))
    images = get_source_images()
    out_dir = EYE_CROPS_PRED_DIR
    tag = f"conf{conf_thresh}"

    print(f"\n[PREDICTED] Cropping {len(images)} images "
          f"(conf>={conf_thresh}, padding={padding:.0%}) …")

    meta_rows: list[dict] = []
    fail_rows: list[dict] = []
    counters: dict[str, int] = {"healthy": 0, "pinkeye": 0}

    for img_path, cls_label in images:
        img = cv2.imread(str(img_path))
        if img is None:
            fail_rows.append(_fail_row(img_path, cls_label, "unreadable_image", 0, 0.0))
            continue

        results = model.predict(str(img_path), conf=conf_thresh, verbose=False)
        best = _select_best_box(results)

        if best is None:
            num_boxes = sum(len(r.boxes) for r in results)
            fail_rows.append(_fail_row(img_path, cls_label, "no_detection", num_boxes, 0.0))
            continue

        x1, y1, x2, y2, conf, num_boxes = best
        crop, (cx1, cy1, cx2, cy2) = _crop_with_padding(img, x1, y1, x2, y2, padding)

        if crop is None or crop.size == 0:
            fail_rows.append(_fail_row(img_path, cls_label, "invalid_crop", num_boxes, conf))
            continue

        counters[cls_label] += 1
        crop_name = f"{cls_label}_{counters[cls_label]:04d}_eye.jpg"
        crop_path = out_dir / cls_label / crop_name
        cv2.imwrite(str(crop_path), crop)

        status = "success" if num_boxes == 1 else "multiple_boxes_selected_best"
        meta_rows.append(_meta_row(
            img_path, crop_path, cls_label, conf, cx1, cy1, cx2, cy2,
            padding, "predicted", status,
        ))

    _save_csvs(meta_rows, fail_rows, suffix=f"_pred_{tag}")
    _print_summary("PREDICTED", meta_rows, fail_rows, counters)


# ── Ground-truth cropping ─────────────────────────────────────────────────────

def _crop_ground_truth(padding: float) -> None:
    images = get_source_images()
    out_dir = EYE_CROPS_GT_DIR

    print(f"\n[GROUND TRUTH] Cropping {len(images)} images "
          f"(padding={padding:.0%}) …")

    meta_rows: list[dict] = []
    fail_rows: list[dict] = []
    counters: dict[str, int] = {"healthy": 0, "pinkeye": 0}

    for img_path, cls_label in images:
        label_path = ANNOTATIONS_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            fail_rows.append(_fail_row(img_path, cls_label, "no_annotation", 0, 0.0))
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            fail_rows.append(_fail_row(img_path, cls_label, "unreadable_image", 0, 0.0))
            continue

        h, w = img.shape[:2]
        boxes = _parse_gt_label(label_path, w, h)
        if not boxes:
            fail_rows.append(_fail_row(img_path, cls_label, "empty_annotation", 0, 0.0))
            continue

        x1, y1, x2, y2 = boxes[0]
        crop, (cx1, cy1, cx2, cy2) = _crop_with_padding(img, x1, y1, x2, y2, padding)

        if crop is None or crop.size == 0:
            fail_rows.append(_fail_row(img_path, cls_label, "invalid_crop", len(boxes), 1.0))
            continue

        counters[cls_label] += 1
        crop_name = f"{cls_label}_{counters[cls_label]:04d}_eye.jpg"
        crop_path = out_dir / cls_label / crop_name
        cv2.imwrite(str(crop_path), crop)

        meta_rows.append(_meta_row(
            img_path, crop_path, cls_label, 1.0, cx1, cy1, cx2, cy2,
            padding, "ground_truth", "success",
        ))

    _save_csvs(meta_rows, fail_rows, suffix="_gt")
    _print_summary("GROUND TRUTH", meta_rows, fail_rows, counters)


# ── Row builders ──────────────────────────────────────────────────────────────

def _meta_row(original, cropped, cls_label, conf, cx1, cy1, cx2, cy2,
              padding, crop_source, status) -> dict:
    return {
        "original_image": str(original),
        "cropped_image": str(cropped),
        "class_label": cls_label,
        "detector_confidence": round(conf, 4),
        "x_min": cx1,
        "y_min": cy1,
        "x_max": cx2,
        "y_max": cy2,
        "crop_width": cx2 - cx1,
        "crop_height": cy2 - cy1,
        "padding": padding,
        "crop_source": crop_source,
        "status": status,
    }


def _fail_row(img_path, cls_label, reason, num_boxes, max_conf) -> dict:
    return {
        "image_path": str(img_path),
        "class_label": cls_label,
        "reason": reason,
        "num_boxes": num_boxes,
        "max_confidence": round(max_conf, 4),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _save_csvs(meta_rows, fail_rows, suffix: str = "") -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    meta_csv = METADATA_DIR / f"eye_crops{suffix}.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    fail_csv = METADATA_DIR / f"failed_detections{suffix}.csv"
    pd.DataFrame(fail_rows).to_csv(fail_csv, index=False)


def _print_summary(label: str, meta_rows, fail_rows, counters) -> None:
    print("\n" + "=" * 60)
    print(f"CROPPING SUMMARY  [{label}]")
    print("=" * 60)
    print(f"  Successful crops : {len(meta_rows)}")
    for cls, cnt in counters.items():
        print(f"    {cls:>10s}: {cnt}")
    print(f"  Failed images    : {len(fail_rows)}")
    if fail_rows:
        reasons: dict[str, int] = {}
        for f in fail_rows:
            reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
        for reason, cnt in reasons.items():
            print(f"    {reason}: {cnt}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop eye regions from cattle images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default="predicted",
        choices=["predicted", "ground_truth"],
        help="Box source: 'predicted' (YOLO) or 'ground_truth' (manual annotations)",
    )
    parser.add_argument(
        "--padding", type=float, default=CROP_PADDING_FRACTION,
        help=f"Fractional padding around the box (default: {CROP_PADDING_FRACTION})",
    )
    parser.add_argument(
        "--conf", type=str, default=str(CONFIDENCE_THRESHOLD),
        help="Confidence threshold(s) for predicted mode. "
             "Pass comma-separated values to sweep, e.g. 0.15,0.25,0.40  "
             f"(default: {CONFIDENCE_THRESHOLD})",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.source == "ground_truth":
        _crop_ground_truth(args.padding)
    else:
        thresholds = [float(t.strip()) for t in args.conf.split(",")]
        for thresh in thresholds:
            _crop_predicted(args.padding, thresh)


if __name__ == "__main__":
    main()
