"""Class-conditional U-Net with self-attention for DDPM."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -factor)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial dimensions."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x + residual


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.class_mlp = nn.Linear(class_dim, out_ch) if class_dim is not None else None
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if self.class_mlp is not None and c_emb is not None:
            h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.res_conv(x)


class Down(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None, use_attn: bool = False
    ) -> None:
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, time_dim, class_dim)
        self.block2 = ResBlock(out_ch, out_ch, time_dim, class_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.pool = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor | None = None):
        x = self.block1(x, t_emb, c_emb)
        x = self.block2(x, t_emb, c_emb)
        x = self.attn(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, class_dim: int | None = None,
        use_attn: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block1 = ResBlock(in_ch + skip_ch, out_ch, time_dim, class_dim)
        self.block2 = ResBlock(out_ch, out_ch, time_dim, class_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        c_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb, c_emb)
        x = self.block2(x, t_emb, c_emb)
        x = self.attn(x)
        return x


class ConditionalUNet(nn.Module):
    """Class-conditional U-Net with self-attention for 64x64 / 128x128 DDPM.

    4 encoder levels, attention at the two deepest resolutions.
    Channel progression: base -> base -> 2*base -> 4*base -> 8*base
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        base_ch: int = 64,
        time_dim: int = 256,
    ) -> None:
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.class_embed = nn.Embedding(num_classes, time_dim)

        ch = base_ch
        self.in_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.down1 = Down(ch, ch, time_dim, time_dim, use_attn=False)
        self.down2 = Down(ch, ch * 2, time_dim, time_dim, use_attn=False)
        self.down3 = Down(ch * 2, ch * 4, time_dim, time_dim, use_attn=True)
        self.down4 = Down(ch * 4, ch * 8, time_dim, time_dim, use_attn=True)

        self.mid1 = ResBlock(ch * 8, ch * 8, time_dim, time_dim)
        self.mid_attn = SelfAttention(ch * 8)
        self.mid2 = ResBlock(ch * 8, ch * 8, time_dim, time_dim)

        self.up4 = Up(ch * 8, ch * 8, ch * 4, time_dim, time_dim, use_attn=True)
        self.up3 = Up(ch * 4, ch * 4, ch * 2, time_dim, time_dim, use_attn=True)
        self.up2 = Up(ch * 2, ch * 2, ch, time_dim, time_dim, use_attn=False)
        self.up1 = Up(ch, ch, ch, time_dim, time_dim, use_attn=False)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(y)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb, c_emb)
        x, s2 = self.down2(x, t_emb, c_emb)
        x, s3 = self.down3(x, t_emb, c_emb)
        x, s4 = self.down4(x, t_emb, c_emb)

        x = self.mid1(x, t_emb, c_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb, c_emb)

        x = self.up4(x, s4, t_emb, c_emb)
        x = self.up3(x, s3, t_emb, c_emb)
        x = self.up2(x, s2, t_emb, c_emb)
        x = self.up1(x, s1, t_emb, c_emb)

        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)
