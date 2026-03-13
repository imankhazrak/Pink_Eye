#!/usr/bin/env python3
"""
Step 1.6 – Validate the trained YOLO eye detector.

Runs YOLO val on the validation (or test) split, prints metrics, and
generates a visual grid of predicted boxes overlaid on sample images.

Usage:
    python -m src.step1.validate_detector
    python -m src.step1.validate_detector --split test
    python -m src.step1.validate_detector --conf 0.30
"""

import argparse
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.step1.config import (
    CONFIDENCE_THRESHOLD,
    DETECTION_DATASET_DIR,
    DETECTOR_VALIDATION_DIR,
    DETECTOR_WEIGHTS_DIR,
    IMAGE_EXTENSIONS,
    YOLO_CONFIG_PATH,
    ensure_dirs,
)


def run_validation(split: str = "val", conf: float = CONFIDENCE_THRESHOLD) -> None:
    ensure_dirs()

    weights = DETECTOR_WEIGHTS_DIR / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights}. Run train_detector.py first."
        )

    print(f"Loading model from {weights}")
    model = YOLO(str(weights))

    print(f"Running validation on '{split}' split …")
    metrics = model.val(
        data=str(YOLO_CONFIG_PATH),
        split=split,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("DETECTOR VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Split       : {split}")
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("=" * 60)

    _save_visual_grid(model, split, conf)


def _save_visual_grid(model: YOLO, split: str, conf: float,
                      n_samples: int = 8) -> None:
    """Draw predicted boxes on sample images and save a grid."""
    img_dir = DETECTION_DATASET_DIR / "images" / split
    if not img_dir.exists():
        print(f"[WARN] Image directory {img_dir} not found; skipping visual grid.")
        return

    image_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        return

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(image_paths),
                        size=min(n_samples, len(image_paths)), replace=False)

    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(chosen):
        img_path = image_paths[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        results = model.predict(str(img_path), conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                c = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"{c:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        r_idx, c_idx = divmod(i, cols)
        axes[r_idx, c_idx].imshow(img)
        axes[r_idx, c_idx].set_title(img_path.name, fontsize=8)
        axes[r_idx, c_idx].axis("off")

    for i in range(len(chosen), rows * cols):
        r_idx, c_idx = divmod(i, cols)
        axes[r_idx, c_idx].axis("off")

    plt.suptitle(f"Detector Predictions – {split} split (conf >= {conf})",
                 fontsize=14)
    plt.tight_layout()
    out = DETECTOR_VALIDATION_DIR / f"visual_grid_{split}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual grid saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO eye detector")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()
    run_validation(args.split, args.conf)


if __name__ == "__main__":
    main()
