"""Configuration helpers for Step 2 classification experiments."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    """Container for train/eval settings used across the pipeline."""

    model_name: str = "resnet18"
    dataset_root: str = "outputs/eye_crops_classification"
    healthy_dir_name: str = "healthy"
    pinkeye_dir_name: str = "pink_eye"
    image_size: int = 224
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_folds: int = 5
    num_workers: int = 4
    random_seed: int = 42
    patience: int = 8
    min_delta: float = 1e-4
    device: str = "cuda"
    output_root: str = "outputs"

    # Imbalance controls
    loss_type: str = "weighted_ce"  # weighted_ce | focal
    use_weighted_sampler: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Model controls
    pretrained: bool = True
    freeze_backbone_epochs: int = 0

    # Logging / visualization controls
    gradcam_examples_per_fold: int = 6
    save_gradcam: bool = True

    # Augmentation controls
    augment_minority_only: bool = False
    minority_label: int = 1

    def resolve_paths(self, project_root: Path) -> Dict[str, Path]:
        """Return commonly used output directories and ensure they exist."""
        output_root = Path(self.output_root)
        if not output_root.is_absolute():
            output_root = project_root / output_root

        dirs = {
            "output_root": output_root,
            "models": output_root / "models",
            "logs": output_root / "logs",
            "metrics": output_root / "metrics",
            "plots": output_root / "plots",
            "explainability": output_root / "explainability",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        return dirs


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_config(config: TrainingConfig, output_path: Path) -> None:
    """Persist run configuration to JSON for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
