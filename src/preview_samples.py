"""Create quick class-wise grids from generated samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.utils.image_utils import save_grid


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TO_TENSOR = transforms.ToTensor()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview generated images as grids")
    parser.add_argument("--samples-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-per-class", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    return parser.parse_args()


def load_images_as_tensor(image_paths: list[Path]) -> torch.Tensor:
    images = []
    for path in image_paths:
        images.append(TO_TENSOR(Image.open(path).convert("RGB")))
    return torch.stack(images, dim=0)


def main() -> None:
    args = parse_args()
    samples_root = Path(args.samples_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(samples_root.iterdir()):
        if not class_dir.is_dir():
            continue
        image_paths = sorted(
            [p for p in class_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        )[: args.max_per_class]
        if not image_paths:
            continue
        tensor = load_images_as_tensor(image_paths)
        save_grid((tensor * 2.0 - 1.0), output_dir / f"preview_{class_dir.name}.png", nrow=args.nrow)

    print(f"Preview grids saved at: {output_dir}")


if __name__ == "__main__":
    main()
