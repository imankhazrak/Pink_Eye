"""Noise schedule utilities for DDPM."""

from __future__ import annotations

import math
import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021).

    Preserves signal much more gradually than linear, critical for
    small datasets and higher timestep counts.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alpha_bar = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)
