#!/usr/bin/env python3
"""
Draw bounding boxes on sample images for QA of eye annotations.

Reads from data/annotated_data/train/ and saves preview images to
outputs/annotation_preview/.

Usage:
    python scripts/preview_annotations.py
    python scripts/preview_annotations.py --samples 8
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DATA_DIR = PROJECT_ROOT / "data" / "annotated_data" / "train"
IMAGES_DIR = ANNOTATED_DATA_DIR / "images"
LABELS_DIR = ANNOTATED_DATA_DIR / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation_preview"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_yolo_line(line: str, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Parse one YOLO line (class x_center y_center width height) to xyxy in pixels."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        _, xc, yc, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return (x1, y1, x2, y2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview eye annotations")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample images (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect image-label pairs
    pairs = []
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in {e.lower() for e in IMAGE_EXTENSIONS}:
            continue
        stem = img_path.stem
        lbl_path = LABELS_DIR / (stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    if not pairs:
        print("No image-label pairs found.")
        return

    rng = random.Random(args.seed)
    samples = rng.sample(pairs, min(args.samples, len(pairs)))

    for img_path, lbl_path in samples:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Could not read image: {img_path}: {e}")
            continue

        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        text = lbl_path.read_text().strip()
        for line in text.splitlines():
            box = _parse_yolo_line(line, img_w, img_h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            draw.text((x1, max(0, y1 - 14)), "eye", fill="lime")

        out_path = OUTPUT_DIR / (img_path.stem + "_preview.jpg")
        img.save(str(out_path))
        print(f"  Saved: {out_path.name}")

    print(f"\nPreview images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
