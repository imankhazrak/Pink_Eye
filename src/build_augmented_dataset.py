"""Build a merged real + synthetic dataset for classification experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ("healthy", "pink_eye")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge real and synthetic class folders")
    parser.add_argument("--real-root", type=str, required=True)
    parser.add_argument("--synthetic-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--limit-synth-per-class", type=int, default=0)
    return parser.parse_args()


def copy_images(src_dir: Path, dst_dir: Path, prefix: str, limit: int = 0) -> int:
    files = sorted([p for p in src_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    if limit > 0:
        files = files[:limit]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files, start=1):
        dst_name = f"{prefix}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst_dir / dst_name)
    return len(files)


def main() -> None:
    args = parse_args()
    real_root = Path(args.real_root)
    synth_root = Path(args.synthetic_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for class_name in CLASS_NAMES:
        out_class = out_root / class_name
        real_count = copy_images(real_root / class_name, out_class, prefix="real", limit=0)
        synth_count = copy_images(
            synth_root / class_name,
            out_class,
            prefix="synth",
            limit=args.limit_synth_per_class,
        )
        summary[class_name] = {"real": real_count, "synthetic": synth_count, "total": real_count + synth_count}

    print("Merged dataset created at:", out_root)
    for cls, counts in summary.items():
        print(f"{cls}: real={counts['real']} synthetic={counts['synthetic']} total={counts['total']}")


if __name__ == "__main__":
    main()
