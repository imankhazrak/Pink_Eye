"""Exponential Moving Average helper for diffusion training."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EMA:
    """Maintains an EMA copy of a model's parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in model_state:
                v.copy_(self.decay * v + (1.0 - self.decay) * model_state[k].detach())
