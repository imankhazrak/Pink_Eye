"""Dataset and transform utilities for Step 2 classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.config import IMAGENET_MEAN, IMAGENET_STD


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ClassMapping:
    """Class-name to index mapping for binary classification."""

    healthy: str = "healthy"
    pinkeye: str = "pink_eye"
    healthy_label: int = 0
    pinkeye_label: int = 1

    def to_dict(self) -> Dict[str, int]:
        return {
            self.healthy: self.healthy_label,
            self.pinkeye: self.pinkeye_label,
        }


class EyeCropsDataset(Dataset):
    """PyTorch dataset backed by a metadata frame of image paths and labels."""

    def __init__(
        self,
        records: pd.DataFrame,
        transform=None,
        minority_transform=None,
        minority_label: int = 1,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_label = minority_label

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.minority_transform is not None and label == self.minority_label:
            image = self.minority_transform(image)
        elif self.transform is not None:
            image = self.transform(image)
        return image, label, row["filename"]


def build_metadata_table(
    dataset_root: Path,
    class_mapping: ClassMapping,
) -> pd.DataFrame:
    """Create a metadata table with columns: path, filename, class_name, label."""
    label_map = class_mapping.to_dict()
    rows: List[Dict[str, object]] = []

    for class_name, class_label in label_map.items():
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}. "
                "Verify dataset path and class folder names."
            )

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "filename": image_path.name,
                    "class_name": class_name,
                    "label": int(class_label),
                }
            )

    if not rows:
        raise RuntimeError(f"No images found in dataset root: {dataset_root}")

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("filename").reset_index(drop=True)
    return frame


def make_fold_indices(
    labels: Sequence[int],
    num_folds: int,
    random_seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """Generate stratified K-fold train/validation indices."""
    indices = list(range(len(labels)))
    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed,
    )
    folds: List[Tuple[List[int], List[int]]] = []
    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds


def get_train_transforms(image_size: int):
    """Training transforms with geometric augmentation only."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=10)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int):
    """Validation/test transforms: resize + normalize only."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
