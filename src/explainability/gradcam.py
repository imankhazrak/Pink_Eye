"""Grad-CAM utilities for CNN classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.config import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Minimal Grad-CAM implementation using forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Return normalized heatmap in [0, 1] for a single image tensor."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class]
        score.backward(retain_graph=False)

        grads = self.gradients  # [B, C, H, W]
        acts = self.activations  # [B, C, H, W]
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def _denormalize_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor [3,H,W] to displayable RGB numpy array."""
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    return image


def save_gradcam_examples(
    model: torch.nn.Module,
    target_layer: Optional[torch.nn.Module],
    dataloader: Iterable,
    device: torch.device,
    output_dir: Path,
    max_examples: int = 6,
) -> int:
    """Generate and save Grad-CAM overlays for a validation loader."""
    if target_layer is None:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = GradCAM(model=model, target_layer=target_layer)
    model.eval()

    saved = 0
    try:
        for images, labels, filenames in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            for i in range(images.size(0)):
                if saved >= max_examples:
                    return saved

                image_i = images[i : i + 1]
                label_i = int(labels[i].item())
                filename_i = str(filenames[i])

                heatmap = gradcam.generate(image_i, target_class=label_i)
                rgb = _denormalize_to_numpy(images[i])

                plt.figure(figsize=(9, 3))
                plt.subplot(1, 3, 1)
                plt.imshow(rgb)
                plt.title("Input")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(heatmap, cmap="jet")
                plt.title("Grad-CAM")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(rgb)
                plt.imshow(heatmap, cmap="jet", alpha=0.45)
                plt.title(f"Overlay (label={label_i})")
                plt.axis("off")

                plt.tight_layout()
                out_name = output_dir / f"gradcam_{saved+1:03d}_{filename_i}.png"
                plt.savefig(out_name, dpi=200)
                plt.close()
                saved += 1
    finally:
        gradcam.remove_hooks()

    return saved
