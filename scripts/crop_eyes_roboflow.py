#!/usr/bin/env python3
"""
Crop eye regions from images using the trained YOLO detector.

Works with any folder of images (e.g. data/annotated_data, data/dataset).
Saves cropped eyes to outputs/eye_crops_pred/crops/ with metadata.

If images have disease labels (healthy/pinkeye) via folder structure,
use src.step1.crop_eyes instead.

Usage:
    python scripts/crop_eyes_roboflow.py
    python scripts/crop_eyes_roboflow.py --source data/dataset/images/train
    python scripts/crop_eyes_roboflow.py --conf 0.25 --padding 0.10
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "annotated_data" / "train" / "images"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eye_crops_pred" / "crops"
METADATA_DIR = PROJECT_ROOT / "outputs" / "metadata"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_images(src_dir: Path) -> list[Path]:
    """Return list of image paths in directory."""
    images = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def _select_best_box(results):
    """Pick highest-confidence predicted box. Returns (x1,y1,x2,y2,conf,num_boxes) or None."""
    best = None
    total = 0
    for r in results:
        for box in r.boxes:
            total += 1
            conf = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            if best is None or conf > best[4]:
                best = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), conf)
    if best is None:
        return None
    return (*best, total)


def _crop_with_padding(img, x1, y1, x2, y2, pad_frac: float):
    """Crop region with fractional padding, clipped to image bounds."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop eyes using YOLO detector")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help=f"Source image directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"YOLO weights path (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--padding", type=float, default=0.10, help="Padding fraction (0.10 = 10%%)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    if not args.source.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    images = _get_images(args.source)
    if not images:
        print(f"No images found in {args.source}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    meta_rows = []
    fail_rows = []

    print(f"Cropping {len(images)} images (conf>={args.conf}, padding={args.padding:.0%})")

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            fail_rows.append({"image": str(img_path), "reason": "unreadable", "confidence": 0.0})
            continue

        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        best = _select_best_box(results)

        if best is None:
            num_boxes = sum(len(r.boxes) for r in results)
            fail_rows.append({"image": str(img_path), "reason": "no_detection", "num_boxes": num_boxes})
            continue

        x1, y1, x2, y2, conf, num_boxes = best
        crop, (cx1, cy1, cx2, cy2) = _crop_with_padding(img, x1, y1, x2, y2, args.padding)

        if crop is None or crop.size == 0:
            fail_rows.append({"image": str(img_path), "reason": "invalid_crop", "confidence": conf})
            continue

        crop_name = f"{img_path.stem}_eye.jpg"
        crop_path = args.out / crop_name
        cv2.imwrite(str(crop_path), crop)

        meta_rows.append({
            "original_image": str(img_path),
            "cropped_image": str(crop_path),
            "confidence": round(conf, 4),
            "x_min": cx1, "y_min": cy1, "x_max": cx2, "y_max": cy2,
            "crop_width": cx2 - cx1, "crop_height": cy2 - cy1,
            "status": "success" if num_boxes == 1 else "multiple_selected_best",
        })

    # Save metadata
    if meta_rows:
        df = pd.DataFrame(meta_rows)
        csv_path = METADATA_DIR / "eye_crops_roboflow.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Metadata: {csv_path}")

    if fail_rows:
        df_fail = pd.DataFrame(fail_rows)
        fail_path = METADATA_DIR / "eye_crops_failed_roboflow.csv"
        df_fail.to_csv(fail_path, index=False)
        print(f"  Failed: {fail_path}")

    print(f"\n  Success: {len(meta_rows)} crops -> {args.out}")
    print(f"  Failed:  {len(fail_rows)}")
    print(f"\nNext: Inspect crops and use for pink-eye classifier dataset.")


if __name__ == "__main__":
    main()
