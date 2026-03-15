"""Hybrid CNN + Transformer model for pinkeye classification."""

from typing import Optional

import timm
import torch
import torch.nn as nn


class CNNTransformerClassifier(nn.Module):
    """
    Lightweight hybrid model:
    image -> CNN feature map -> token projection -> transformer -> pooling -> head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        backbone_name: str = "resnet18",
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )

        feature_info = self.backbone.feature_info[-1]
        in_channels = int(feature_info["num_chs"])

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)[0]  # [B, C, H, W]
        tokens = self.proj(feat_map)  # [B, E, H, W]
        b, e, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, E]
        tokens = self.norm(tokens)
        encoded = self.transformer(tokens)  # [B, N, E]
        pooled = encoded.mean(dim=1)
        logits = self.head(pooled)
        return logits


def build_cnn_transformer(
    num_classes: int = 2,
    pretrained: bool = True,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 2,
) -> nn.Module:
    """Factory for hybrid CNN+Transformer classifier."""
    return CNNTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )


def get_hybrid_gradcam_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Return None for hybrid model; Grad-CAM is implemented for CNN baselines only.
    """
    return None
