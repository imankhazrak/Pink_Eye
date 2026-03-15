"""Image conversion and grid utilities for synthetic outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torchvision.utils import make_grid, save_image


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] tensor range to [0, 1]."""
    return x.clamp(-1, 1).add(1).div(2)


def save_tensor_image(tensor: torch.Tensor, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm_to_01(tensor), str(path))


def save_grid(images: torch.Tensor, out_path: str | Path, nrow: int = 8) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(denorm_to_01(images), nrow=nrow)
    save_image(grid, str(path))


def count_images(paths: Iterable[Path]) -> int:
    return sum(1 for _ in paths)
