"""Stronger hybrid CNN + Transformer classifier for eye-crop classification."""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


class StrongCNNTransformerClassifier(nn.Module):
    """
    Stronger hybrid CNN + Transformer classifier.

    Design:
    Input -> pretrained CNN feature map -> 1x1 projection -> tokens
    -> CLS token + positional embedding -> transformer encoder
    -> CLS representation -> MLP classification head.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.attn_dropout = attn_dropout

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )

        backbone_channels = self.backbone.feature_info.channels()[-1]

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.backbone(dummy)[0]
            _, _, h, w = feat.shape

        self.feature_h = h
        self.feature_w = w
        self.num_patches = h * w

        self.proj = nn.Sequential(
            nn.Conv2d(backbone_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        self.token_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
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
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[0]  # [B, C, H, W]
        feat = self.proj(feat)  # [B, E, H, W]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, E]

        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = tokens + self.pos_embed
        tokens = self.token_dropout(tokens)
        tokens = self.pre_norm(tokens)

        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)
        return logits


def build_cnn_transformer_strong(
    num_classes: int = 2,
    pretrained: bool = True,
    image_size: int = 224,
) -> nn.Module:
    """Factory for stronger CNN+Transformer model."""
    return StrongCNNTransformerClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
    )


def get_strong_hybrid_gradcam_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Return None; Grad-CAM is not implemented for transformer-hybrid models."""
    return None
