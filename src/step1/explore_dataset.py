#!/usr/bin/env python3
"""
Step 1.1 – Explore the source cattle-image dataset.

Prints per-class counts, image dimension statistics, class-imbalance
ratio, and saves a sample grid.

Usage:
    python -m src.step1.explore_dataset
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.step1.config import (
    CLASS_NAMES,
    OUTPUTS_DIR,
    get_source_images,
)


def collect_stats(images: list) -> dict:
    stats: dict[str, list] = {cls: [] for cls in CLASS_NAMES}
    for path, label in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [WARN] Cannot read: {path}")
            continue
        h, w = img.shape[:2]
        stats[label].append({"path": path, "width": w, "height": h})
    return stats


def print_summary(stats: dict) -> None:
    print("=" * 65)
    print("SOURCE DATASET SUMMARY")
    print("=" * 65)
    counts = {}
    total = 0
    for cls in CLASS_NAMES:
        entries = stats[cls]
        n = len(entries)
        counts[cls] = n
        total += n
        if n == 0:
            print(f"  {cls:>10s}: 0 images")
            continue
        widths = [e["width"] for e in entries]
        heights = [e["height"] for e in entries]
        print(f"  {cls:>10s}: {n:>4d} images  |  "
              f"W: {min(widths)}-{max(widths)} (mean {np.mean(widths):.0f})  |  "
              f"H: {min(heights)}-{max(heights)} (mean {np.mean(heights):.0f})")
    print(f"  {'TOTAL':>10s}: {total:>4d} images")

    if len(counts) == 2 and all(v > 0 for v in counts.values()):
        vals = list(counts.values())
        ratio = max(vals) / min(vals)
        minority = min(counts, key=counts.get)
        print(f"\n  Class imbalance ratio: {ratio:.1f}:1  (minority = {minority})")
    print("=" * 65)


def save_sample_grid(stats: dict, n_per_class: int = 4) -> None:
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(4 * n_per_class, 4 * len(CLASS_NAMES)))
    if len(CLASS_NAMES) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for row, cls in enumerate(CLASS_NAMES):
        entries = stats[cls]
        chosen = rng.choice(len(entries), size=min(n_per_class, len(entries)), replace=False)
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(chosen):
                e = entries[chosen[col]]
                img = cv2.cvtColor(cv2.imread(str(e["path"])), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"{cls}\n{e['width']}x{e['height']}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Sample Images from Source Dataset", fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUTPUTS_DIR / "sample_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample grid saved to {out}")


def main() -> None:
    images = get_source_images()
    if not images:
        print("No source images found. Check SOURCE_DATA_DIR in config.py.")
        return

    stats = collect_stats(images)
    print_summary(stats)
    save_sample_grid(stats)


if __name__ == "__main__":
    main()
