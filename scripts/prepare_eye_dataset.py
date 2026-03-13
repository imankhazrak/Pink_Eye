#!/usr/bin/env python3
"""
Split Roboflow eye annotations 80/10/10 and build YOLO dataset.

Reads from data/annotated_data/train/ and writes to data/dataset/:

    data/dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

Split: 80% train, 10% val, 10% test (random, seed 42).

Usage:
    python scripts/prepare_eye_dataset.py
"""

import shutil
from pathlib import Path

import yaml

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DATA_DIR = PROJECT_ROOT / "data" / "annotated_data" / "train"
IMAGES_SRC = ANNOTATED_DATA_DIR / "images"
LABELS_SRC = ANNOTATED_DATA_DIR / "labels"

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_image_label_pairs() -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) for every annotated pair."""
    pairs = []
    for img_path in sorted(IMAGES_SRC.iterdir()):
        if img_path.suffix.lower() not in {e.lower() for e in IMAGE_EXTENSIONS}:
            continue
        stem = img_path.stem
        lbl_path = LABELS_SRC / (stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"  [WARN] No label for image: {img_path.name}")
    return pairs


def main() -> None:
    if not ANNOTATED_DATA_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {ANNOTATED_DATA_DIR}")

    pairs = _get_image_label_pairs()
    if not pairs:
        print("No image-label pairs found.")
        return

    n = len(pairs)
    print(f"Found {n} image-label pairs.")

    # Random split 80/10/10
    import random

    rng = random.Random(RANDOM_SEED)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }

    # Clear and create output structure
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split_name, split_pairs in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(str(img_path), str(img_dir / img_path.name))
            shutil.copy2(str(lbl_path), str(lbl_dir / (img_path.stem + ".txt")))

        print(f"  {split_name:>5}: {len(split_pairs):>4} images")

    # Write data.yaml
    yaml_path = DATASET_DIR / "data.yaml"
    cfg = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "eye"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset written to {DATASET_DIR}")
    print(f"YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()
