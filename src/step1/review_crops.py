#!/usr/bin/env python3
"""
Step 1.8 – Manual quality review of cropped eye images.

Displays each crop alongside the original image (with bounding box
overlay) and lets the user classify the crop quality.

Controls:
    g / Enter   Mark as 'good', advance
    b           Mark as 'bad', advance
    u           Mark as 'uncertain', advance
    q           Quit (progress is saved; resume later)

All decisions are written to ``outputs/metadata/crop_review.csv`` with
a timestamp so that review sessions are fully auditable.

Non-interactive mode (``--auto``) generates a summary grid without
requiring user input.

Usage:
    python -m src.step1.review_crops                        # interactive (pred)
    python -m src.step1.review_crops --source ground_truth  # interactive (gt)
    python -m src.step1.review_crops --auto                 # grid only
"""

import argparse
from datetime import datetime, timezone

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.step1.config import (
    EYE_CROPS_PRED_DIR,
    EYE_CROPS_GT_DIR,
    METADATA_DIR,
    OUTPUTS_DIR,
    CLASS_NAMES,
    ensure_dirs,
)

REVIEW_CSV = METADATA_DIR / "crop_review.csv"

REVIEW_COLUMNS = [
    "cropped_image",
    "original_image",
    "class_label",
    "crop_source",
    "reviewer_decision",
    "optional_note",
    "timestamp",
]


# ── Load / save helpers ──────────────────────────────────────────────────────

def _load_meta(source: str) -> pd.DataFrame:
    """Load the crop metadata CSV that matches the chosen source."""
    pattern = "_gt" if source == "ground_truth" else "_pred_conf"
    candidates = sorted(METADATA_DIR.glob("eye_crops*.csv"))
    for c in candidates:
        if pattern in c.name:
            return pd.read_csv(c)
    # Fallback: try the generic name
    generic = METADATA_DIR / "eye_crops.csv"
    if generic.exists():
        return pd.read_csv(generic)
    raise FileNotFoundError(
        f"No crop metadata CSV found in {METADATA_DIR}. Run crop_eyes.py first."
    )


def _load_existing_review() -> pd.DataFrame:
    if REVIEW_CSV.exists():
        return pd.read_csv(REVIEW_CSV)
    return pd.DataFrame(columns=REVIEW_COLUMNS)


def _save_review(df: pd.DataFrame) -> None:
    df.to_csv(REVIEW_CSV, index=False)
    print(f"Review saved to {REVIEW_CSV}  ({len(df)} entries)")


# ── Interactive review ────────────────────────────────────────────────────────

def interactive_review(source: str) -> None:
    ensure_dirs()
    meta = _load_meta(source)
    existing = _load_existing_review()
    already_reviewed = set(existing["cropped_image"].tolist())

    pending = meta[~meta["cropped_image"].isin(already_reviewed)].reset_index(drop=True)
    print(f"Total crops: {len(meta)}  |  Already reviewed: {len(already_reviewed)}  |  "
          f"Remaining: {len(pending)}")
    if pending.empty:
        print("All crops already reviewed.")
        return

    new_rows: list[dict] = []
    win = "Review Crop – g=good  b=bad  u=uncertain  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    for _, row in pending.iterrows():
        crop_path = Path(row["cropped_image"])
        orig_path = Path(row["original_image"])

        crop_img = cv2.imread(str(crop_path))
        orig_img = cv2.imread(str(orig_path))

        if crop_img is None:
            new_rows.append(_review_row(row, source, "unreadable", "auto-skipped"))
            continue

        if orig_img is not None:
            x1, y1 = int(row["x_min"]), int(row["y_min"])
            x2, y2 = int(row["x_max"]), int(row["y_max"])
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            scale = 400.0 / max(orig_img.shape[:2])
            orig_small = cv2.resize(orig_img, None, fx=scale, fy=scale)
        else:
            orig_small = np.zeros((400, 400, 3), dtype=np.uint8)

        crop_resized = cv2.resize(crop_img, (400, 400))
        combined = np.hstack([orig_small, crop_resized])

        reviewed_so_far = len(already_reviewed) + len(new_rows)
        conf = row.get("detector_confidence", row.get("confidence", 0))
        label_text = (f"[{reviewed_so_far + 1}/{len(meta)}]  "
                      f"{row['class_label']}  conf={conf:.2f}")
        cv2.putText(combined, label_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        hint = "g=good  b=bad  u=uncertain  q=quit"
        cv2.putText(combined, hint, (10, combined.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(win, combined)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("g") or key == 13:
                new_rows.append(_review_row(row, source, "good"))
                break
            elif key == ord("b"):
                new_rows.append(_review_row(row, source, "bad"))
                break
            elif key == ord("u"):
                new_rows.append(_review_row(row, source, "uncertain"))
                break
            elif key == ord("q"):
                _flush(existing, new_rows)
                print("Quit. Progress saved – run again to resume.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    _flush(existing, new_rows)
    _print_summary()


def _review_row(meta_row, source: str, decision: str, note: str = "") -> dict:
    return {
        "cropped_image": str(meta_row["cropped_image"]),
        "original_image": str(meta_row["original_image"]),
        "class_label": meta_row["class_label"],
        "crop_source": source,
        "reviewer_decision": decision,
        "optional_note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _flush(existing: pd.DataFrame, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    _save_review(combined)


def _print_summary() -> None:
    if not REVIEW_CSV.exists():
        return
    df = pd.read_csv(REVIEW_CSV)
    counts = df["reviewer_decision"].value_counts()
    print("\nReview summary:")
    for decision, n in counts.items():
        print(f"  {decision}: {n}")


# ── Auto grid ─────────────────────────────────────────────────────────────────

def auto_grid(source: str, n_per_class: int = 6) -> None:
    """Save a sample grid of crops for quick visual inspection."""
    ensure_dirs()
    crops_dir = EYE_CROPS_GT_DIR if source == "ground_truth" else EYE_CROPS_PRED_DIR

    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(3 * n_per_class, 3 * len(CLASS_NAMES)))
    if len(CLASS_NAMES) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for row, cls in enumerate(CLASS_NAMES):
        cls_dir = crops_dir / cls
        if not cls_dir.exists():
            continue
        crops = sorted(cls_dir.glob("*.jpg"))
        if not crops:
            continue
        chosen_idx = rng.choice(len(crops), size=min(n_per_class, len(crops)), replace=False)
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(chosen_idx):
                img = cv2.cvtColor(cv2.imread(str(crops[chosen_idx[col]])), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(cls, fontsize=9)
            ax.axis("off")

    title = f"Eye Crop Samples ({source})"
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUTPUTS_DIR / f"crop_review_grid_{source}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Crop review grid saved to {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Review cropped eye images")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive mode: generate summary grid only")
    parser.add_argument("--source", default="predicted",
                        choices=["predicted", "ground_truth"],
                        help="Which crop dataset to review")
    args = parser.parse_args()

    if args.auto:
        auto_grid(args.source)
    else:
        interactive_review(args.source)


if __name__ == "__main__":
    main()
