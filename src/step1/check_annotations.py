#!/usr/bin/env python3
"""
Step 1.3 – Validate all YOLO annotation files before building the
detection dataset.

Checks performed on every .txt label file in data/annotations/:
  1. Matching source image exists
  2. File is not empty
  3. Every line has exactly 5 space-separated tokens
  4. class_id == 0
  5. x_center, y_center, width, height are floats in [0, 1]
  6. width > 0 and height > 0
  7. Flags suspiciously small or large boxes

Outputs:
  - outputs/metadata/annotation_check_report.csv
  - Terminal summary

Usage:
    python -m src.step1.check_annotations
"""

import pandas as pd

from src.step1.config import (
    ANNOTATIONS_DIR,
    DETECTION_CLASS_ID,
    METADATA_DIR,
    ensure_dirs,
    find_source_image,
)

SUSPICIOUS_MIN_SIZE = 0.01   # box side < 1 % of image dimension
SUSPICIOUS_MAX_SIZE = 0.90   # box side > 90 % of image dimension


def _check_one(label_path) -> list[dict]:
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
    match = find_source_image(stem)
    if match is None:
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
            rows.append(_row("ERROR", f"class_id={cls_id}, expected {DETECTION_CLASS_ID}", line_idx))

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
    ensure_dirs()

    label_files = sorted(ANNOTATIONS_DIR.glob("*.txt"))
    if not label_files:
        print("No annotation files found in", ANNOTATIONS_DIR)
        return pd.DataFrame()

    all_rows: list[dict] = []
    for lf in label_files:
        all_rows.extend(_check_one(lf))

    df = pd.DataFrame(all_rows)
    report_path = METADATA_DIR / "annotation_check_report.csv"
    df.to_csv(report_path, index=False)

    # Terminal summary
    n_files = len(label_files)
    n_errors = len(df[df["level"] == "ERROR"])
    n_warns = len(df[df["level"] == "WARN"])
    n_ok = len(df[df["level"] == "OK"])

    print("=" * 60)
    print("ANNOTATION VALIDATION REPORT")
    print("=" * 60)
    print(f"  Label files checked : {n_files}")
    print(f"  Valid (no issues)   : {n_ok}")
    print(f"  Warnings            : {n_warns}")
    print(f"  Errors              : {n_errors}")

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
    check_all()


if __name__ == "__main__":
    main()
