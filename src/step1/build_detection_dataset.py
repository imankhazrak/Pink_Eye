#!/usr/bin/env python3
"""
Step 1.4 – Build the YOLO detection dataset from annotations.

Reads annotated images + labels, splits them into train / val / test,
copies files into the YOLO directory structure, and generates the YAML
config.

Splitting strategy (chosen automatically):
  - If ``data/animal_ids.csv`` exists and contains an ``animal_id``
    column, *group-aware* splitting is used so that all images of the
    same animal stay in the same split (prevents data leakage).
  - Otherwise, a *stratified* split by disease label is used.

Usage:
    python -m src.step1.build_detection_dataset
"""

import shutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from src.step1.config import (
    ANIMAL_ID_CSV,
    ANNOTATIONS_DIR,
    DETECTION_CLASS_NAME,
    DETECTION_DATASET_DIR,
    RANDOM_SEED,
    CLASS_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    YOLO_CONFIG_PATH,
    find_source_image,
)


# ── Locate annotated data ────────────────────────────────────────────────────

def _find_annotated_pairs() -> list[tuple[Path, Path, str]]:
    """Return (image_path, label_path, class_label) for every annotated image."""
    pairs = []
    for label_path in sorted(ANNOTATIONS_DIR.glob("*.txt")):
        match = find_source_image(label_path.stem)
        if match is None:
            print(f"  [WARN] No source image for annotation {label_path.name}")
            continue
        img_path, cls = match
        pairs.append((img_path, label_path, cls))
    return pairs


# ── Splitting helpers ─────────────────────────────────────────────────────────

def _load_animal_ids(pairs) -> pd.Series | None:
    """Try to load animal_id for each annotated image. Returns None on failure."""
    if ANIMAL_ID_CSV is None or not ANIMAL_ID_CSV.exists():
        return None
    try:
        df = pd.read_csv(ANIMAL_ID_CSV)
    except Exception:
        return None
    if "animal_id" not in df.columns or "image" not in df.columns:
        print("  [WARN] animal_ids.csv missing 'animal_id' or 'image' column; "
              "falling back to stratified split.")
        return None

    lookup = dict(zip(df["image"], df["animal_id"]))
    ids = [lookup.get(img.stem, lookup.get(img.name)) for img, _, _ in pairs]
    if any(v is None for v in ids):
        missing = sum(1 for v in ids if v is None)
        print(f"  [WARN] {missing}/{len(ids)} images have no animal_id; "
              "falling back to stratified split.")
        return None
    return pd.Series(ids)


def _group_aware_split(pairs, groups, seed):
    """Split keeping all images of the same animal in one split."""
    labels = np.array([cls for _, _, cls in pairs])
    indices = np.arange(len(pairs))
    groups_arr = np.array(groups)

    val_test_frac = VAL_RATIO + TEST_RATIO
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_frac, random_state=seed)
    train_idx, valtest_idx = next(gss1.split(indices, labels, groups_arr))

    relative_test = TEST_RATIO / val_test_frac
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    sub_indices = np.arange(len(valtest_idx))
    val_sub, test_sub = next(gss2.split(
        sub_indices, labels[valtest_idx], groups_arr[valtest_idx]))

    val_idx = valtest_idx[val_sub]
    test_idx = valtest_idx[test_sub]

    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }


def _stratified_split(pairs, seed):
    """Stratified train / val / test split by disease label."""
    labels = [cls for _, _, cls in pairs]

    val_test_ratio = VAL_RATIO + TEST_RATIO
    train_pairs, valtest_pairs, _, valtest_labels = train_test_split(
        pairs, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    relative_test = TEST_RATIO / val_test_ratio
    val_pairs, test_pairs = train_test_split(
        valtest_pairs,
        test_size=relative_test,
        stratify=valtest_labels,
        random_state=seed,
    )

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


# ── File operations ───────────────────────────────────────────────────────────

def _copy_split(split_name: str, pairs: list[tuple[Path, Path, str]]) -> None:
    img_dir = DETECTION_DATASET_DIR / "images" / split_name
    lbl_dir = DETECTION_DATASET_DIR / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path, _ in pairs:
        shutil.copy2(str(img_path), str(img_dir / img_path.name))
        shutil.copy2(str(lbl_path), str(lbl_dir / (img_path.stem + ".txt")))


def _write_yaml() -> None:
    cfg = {
        "path": str(DETECTION_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: DETECTION_CLASS_NAME},
    }
    YOLO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(YOLO_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"YOLO config written to {YOLO_CONFIG_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pairs = _find_annotated_pairs()
    if not pairs:
        print("No annotated images found. Run annotate_eyes.py first.")
        return

    print(f"Found {len(pairs)} annotated images.")
    labels = [cls for _, _, cls in pairs]
    for cls in CLASS_NAMES:
        print(f"  {cls}: {labels.count(cls)}")

    if DETECTION_DATASET_DIR.exists():
        shutil.rmtree(DETECTION_DATASET_DIR)

    # Choose splitting strategy
    animal_ids = _load_animal_ids(pairs)
    if animal_ids is not None:
        print("\n  Using GROUP-AWARE splitting (by animal_id).")
        splits = _group_aware_split(pairs, animal_ids, RANDOM_SEED)
    else:
        print("\n  Using STRATIFIED splitting (by disease label).")
        splits = _stratified_split(pairs, RANDOM_SEED)

    for split_name, split_pairs in splits.items():
        _copy_split(split_name, split_pairs)
        split_labels = [c for _, _, c in split_pairs]
        print(f"  {split_name:>5s}: {len(split_pairs):>4d} images  "
              f"(healthy={split_labels.count('healthy')}, "
              f"pinkeye={split_labels.count('pinkeye')})")

    _write_yaml()
    print("Detection dataset built successfully.")


if __name__ == "__main__":
    main()
