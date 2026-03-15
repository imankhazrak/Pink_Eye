"""Dataset for class-conditional DDPM training on eye crops."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_IDX = {
    "healthy": 0,
    "pink_eye": 1,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class EyeCropsDataset(Dataset):
    """Loads class-labeled eye crops for diffusion training."""

    def __init__(self, root_dir: str | Path, image_size: int = 64, augment: bool = True) -> None:
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[Path, int]] = []

        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in dataset root: {self.root_dir}")

        aug_transforms = []
        if augment:
            aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug_transforms,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), str(img_path)
