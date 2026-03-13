#!/usr/bin/env python3
"""
Validate YOLO annotations in data/annotated_data/train/ (Roboflow export).

Checks performed:
  1. For each image in train/images/, verify matching .txt in train/labels/
  2. Label file non-empty
  3. Line format: 0 x_center y_center width height (5 space-separated tokens)
  4. class_id == 0
  5. x_center, y_center, width, height in [0, 1], width > 0, height > 0
  6. Flag very small or large boxes (<1% or >90% of image)

Outputs:
  - outputs/metadata/annotation_check_roboflow_report.csv
  - Terminal summary

Usage:
    python scripts/check_annotations_roboflow.py
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DATA_DIR = PROJECT_ROOT / "data" / "annotated_data" / "train"
IMAGES_DIR = ANNOTATED_DATA_DIR / "images"
LABELS_DIR = ANNOTATED_DATA_DIR / "labels"
METADATA_DIR = PROJECT_ROOT / "outputs" / "metadata"

DETECTION_CLASS_ID = 0
SUSPICIOUS_MIN_SIZE = 0.01  # box side < 1% of image dimension
SUSPICIOUS_MAX_SIZE = 0.90  # box side > 90% of image dimension

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_matching_image(stem: str) -> Optional[Path]:
    """Find image file matching the given stem in train/images/."""
    for ext in IMAGE_EXTENSIONS:
        for suffix in (ext, ext.upper()):
            candidate = IMAGES_DIR / (stem + suffix)
            if candidate.exists():
                return candidate
    return None


def _check_one(label_path: Path) -> list[dict]:
    """Return a list of issue dicts (empty list = all OK)."""
    stem = label_path.stem
    rows: list[dict] = []

    def _row(level: str, message: str, line_num: int = 0) -> dict:
        return {
            "file": label_path.name,
            "image_stem": stem,
            "line": line_num,
            "level": level,
            "message": message,
        }

    # 1. Matching source image
    img = _find_matching_image(stem)
    if img is None:
        rows.append(_row("ERROR", "No matching source image found"))
        return rows

    # 2. Non-empty
    text = label_path.read_text().strip()
    if not text:
        rows.append(_row("ERROR", "Label file is empty"))
        return rows

    for line_idx, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()

        # 3. Token count
        if len(parts) != 5:
            rows.append(_row("ERROR", f"Expected 5 tokens, got {len(parts)}", line_idx))
            continue

        # 4. Class ID
        try:
            cls_id = int(parts[0])
        except ValueError:
            rows.append(_row("ERROR", f"Non-integer class_id: '{parts[0]}'", line_idx))
            continue
        if cls_id != DETECTION_CLASS_ID:
            rows.append(
                _row("ERROR", f"class_id={cls_id}, expected {DETECTION_CLASS_ID}", line_idx)
            )

        # 5. Coordinate validity
        try:
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            rows.append(_row("ERROR", "Non-numeric coordinate values", line_idx))
            continue

        for name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
            if not 0.0 <= val <= 1.0:
                rows.append(_row("ERROR", f"{name}={val:.6f} outside [0,1]", line_idx))

        # 6. Positive size
        if w <= 0:
            rows.append(_row("ERROR", f"width={w:.6f} <= 0", line_idx))
        if h <= 0:
            rows.append(_row("ERROR", f"height={h:.6f} <= 0", line_idx))

        # 7. Suspicious sizes
        if 0 < w < SUSPICIOUS_MIN_SIZE or 0 < h < SUSPICIOUS_MIN_SIZE:
            rows.append(_row("WARN", f"Very small box (w={w:.4f}, h={h:.4f})", line_idx))
        if w > SUSPICIOUS_MAX_SIZE or h > SUSPICIOUS_MAX_SIZE:
            rows.append(_row("WARN", f"Very large box (w={w:.4f}, h={h:.4f})", line_idx))

    if not rows:
        rows.append(_row("OK", "Valid", 0))
    return rows


def check_all() -> pd.DataFrame:
    """Run all checks and return report DataFrame."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_DIR.exists():
        print(f"Labels directory not found: {LABELS_DIR}")
        return pd.DataFrame()

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    if not label_files:
        print("No annotation files found in", LABELS_DIR)
        return pd.DataFrame()

    # Check for images without labels
    image_stems = set()
    for img_path in IMAGES_DIR.iterdir():
        if img_path.suffix.lower() in {e.lower() for e in IMAGE_EXTENSIONS}:
            image_stems.add(img_path.stem)

    label_stems = {p.stem for p in label_files}
    missing_labels = image_stems - label_stems
    if missing_labels:
        for stem in sorted(missing_labels)[:5]:
            print(f"  [WARN] Image has no label: {stem}")
        if len(missing_labels) > 5:
            print(f"  [WARN] ... and {len(missing_labels) - 5} more")

    all_rows: list[dict] = []
    for lf in label_files:
        all_rows.extend(_check_one(lf))

    df = pd.DataFrame(all_rows)
    report_path = METADATA_DIR / "annotation_check_roboflow_report.csv"
    df.to_csv(report_path, index=False)

    # Terminal summary
    n_files = len(label_files)
    n_errors = len(df[df["level"] == "ERROR"])
    n_warns = len(df[df["level"] == "WARN"])
    n_ok = len(df[df["level"] == "OK"])

    print("=" * 60)
    print("ANNOTATION VALIDATION REPORT (Roboflow)")
    print("=" * 60)
    print(f"  Source           : {ANNOTATED_DATA_DIR}")
    print(f"  Label files      : {n_files}")
    print(f"  Images (total)   : {len(image_stems)}")
    print(f"  Valid (no issues): {n_ok}")
    print(f"  Warnings         : {n_warns}")
    print(f"  Errors           : {n_errors}")

    if n_errors > 0:
        print("\n  ERROR DETAILS:")
        for _, r in df[df["level"] == "ERROR"].iterrows():
            print(f"    {r['file']}  line {r['line']}:  {r['message']}")

    if n_warns > 0:
        print("\n  WARNING DETAILS:")
        for _, r in df[df["level"] == "WARN"].iterrows():
            print(f"    {r['file']}  line {r['line']}:  {r['message']}")

    print(f"\n  Full report saved to {report_path}")
    print("=" * 60)

    return df


def main() -> None:
    if not ANNOTATED_DATA_DIR.exists():
        print(f"Annotated data directory not found: {ANNOTATED_DATA_DIR}")
        sys.exit(1)

    df = check_all()
    n_errors = len(df[df["level"] == "ERROR"]) if not df.empty else 0
    sys.exit(1 if n_errors > 0 else 0)


if __name__ == "__main__":
    main()
