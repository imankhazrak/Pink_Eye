#!/usr/bin/env python3
"""
Step 1.5 – Train a YOLO eye detector.

Loads pretrained COCO weights for the model specified in config.py
(default: yolov8n.pt), fine-tunes on the annotated eye dataset, and
saves the best checkpoint.

Model choice
------------
The default baseline detector is ``yolov8n.pt`` (nano).
To compare with a higher-capacity model, set ``YOLO_MODEL`` in
config.py to ``"yolov8s.pt"`` (small) or pass ``--model yolov8s.pt``
on the command line.

Usage:
    python -m src.step1.train_detector
    python -m src.step1.train_detector --model yolov8s.pt --epochs 80
    python -m src.step1.train_detector --epochs 100 --imgsz 640 --batch 16
"""

import argparse
import shutil
from ultralytics import YOLO

from src.step1.config import (
    DETECTOR_DIR,
    DETECTOR_WEIGHTS_DIR,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_PATIENCE,
    YOLO_CONFIG_PATH,
    YOLO_MODEL,
    RANDOM_SEED,
    ensure_dirs,
)


def train(model_name: str, epochs: int, imgsz: int, batch: int) -> None:
    ensure_dirs()

    if not YOLO_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"YOLO config not found at {YOLO_CONFIG_PATH}. "
            "Run build_detection_dataset.py first."
        )

    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    run_name = f"eye_detector_{model_name.replace('.pt', '')}"
    print(f"Starting training: model={model_name}, epochs={epochs}, "
          f"imgsz={imgsz}, batch={batch}")

    model.train(
        data=str(YOLO_CONFIG_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=TRAIN_PATIENCE,
        save=True,
        project=str(DETECTOR_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        seed=RANDOM_SEED,
        verbose=True,
    )

    best_src = DETECTOR_DIR / "runs" / run_name / "weights" / "best.pt"
    if best_src.exists():
        DETECTOR_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = DETECTOR_WEIGHTS_DIR / "best.pt"
        shutil.copy2(str(best_src), str(dest))
        print(f"\nBest weights copied to {dest}")
    else:
        print("[WARN] best.pt not found in training output.")

    print("\nTraining complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO eye detector")
    parser.add_argument("--model", type=str, default=YOLO_MODEL,
                        help="YOLO model file, e.g. yolov8n.pt or yolov8s.pt "
                             f"(default: {YOLO_MODEL})")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train(args.model, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
