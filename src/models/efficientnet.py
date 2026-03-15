"""EfficientNet-B0 baseline classifier for pinkeye detection."""

from typing import Tuple

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 and replace classifier head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return a late feature layer for Grad-CAM."""
    return model.features[-1]


def freeze_backbone(model: nn.Module) -> Tuple[int, int]:
    """Freeze all layers except classifier; return trainable/frozen counts."""
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    return trainable, frozen
