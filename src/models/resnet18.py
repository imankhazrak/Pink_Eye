"""ResNet18 baseline classifier for pinkeye detection."""

from typing import Tuple

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build ResNet18 and replace classification head."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_resnet18_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the deepest convolutional block for Grad-CAM."""
    return model.layer4[-1].conv2


def freeze_backbone(model: nn.Module) -> Tuple[int, int]:
    """Freeze all layers except final FC; return trainable/frozen counts."""
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    return trainable, frozen
