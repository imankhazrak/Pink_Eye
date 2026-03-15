"""Model factory for Step 2 classifiers."""

from typing import Optional

import torch.nn as nn


def build_model(model_name: str, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """Instantiate a model by canonical name."""
    key = model_name.lower().strip()
    if key == "resnet18":
        from src.models.resnet18 import build_resnet18

        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if key in {"efficientnet", "efficientnet_b0", "efficientnet-b0"}:
        from src.models.efficientnet import build_efficientnet_b0

        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if key in {"cnn_transformer", "hybrid"}:
        from src.models.cnn_transformer import build_cnn_transformer

        return build_cnn_transformer(num_classes=num_classes, pretrained=pretrained)
    if key in {"cnn_transformer_strong", "hybrid_strong", "strong_cnn_transformer"}:
        from src.models.cnn_transformer_strong import build_cnn_transformer_strong

        return build_cnn_transformer_strong(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unsupported model '{model_name}'. Choose from: "
        "resnet18, efficientnet, cnn_transformer, cnn_transformer_strong."
    )


def get_gradcam_target_layer(model_name: str, model: nn.Module) -> Optional[nn.Module]:
    """Return the model layer used for Grad-CAM (CNN models only)."""
    key = model_name.lower().strip()
    if key == "resnet18":
        from src.models.resnet18 import get_resnet18_gradcam_target_layer

        return get_resnet18_gradcam_target_layer(model)
    if key in {"efficientnet", "efficientnet_b0", "efficientnet-b0"}:
        from src.models.efficientnet import get_efficientnet_gradcam_target_layer

        return get_efficientnet_gradcam_target_layer(model)
    if key in {"cnn_transformer", "hybrid"}:
        from src.models.cnn_transformer import get_hybrid_gradcam_target_layer

        return get_hybrid_gradcam_target_layer(model)
    if key in {"cnn_transformer_strong", "hybrid_strong", "strong_cnn_transformer"}:
        from src.models.cnn_transformer_strong import get_strong_hybrid_gradcam_target_layer

        return get_strong_hybrid_gradcam_target_layer(model)
    return None
