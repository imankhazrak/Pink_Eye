"""DDPM wrapper (forward diffusion loss + reverse sampling)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.diffusion.scheduler import linear_beta_schedule, cosine_beta_schedule


class DDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        # Coefficients for q(x_{t-1} | x_t, x_0) posterior mean.
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod + 1e-20),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.model(x_noisy, t.float(), y)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        noise_pred = self.model(x, t.float(), y)

        # Predict x_0 first, then use posterior mean q(x_{t-1}|x_t,x_0) for better stability.
        # x0 reconstruction must use cumulative alpha at timestep t.
        x0_pred = (
            x - sqrt_one_minus_alpha_cumprod_t * noise_pred
        ) * self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        model_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x0_pred
            + self.posterior_mean_coef2[t][:, None, None, None] * x
        )

        if (t == 0).all():
            return model_mean

        posterior_var_t = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        image_size: int,
        batch_size: int,
        channels: int,
        y: torch.Tensor,
        device: str,
        clamp: bool = True,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Returns images in [0, 1] if denormalize=True.
        Assumes training images were normalized to [-1, 1].
        """
        self.model.eval()

        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y)

        if clamp:
            x = x.clamp(-1.0, 1.0)

        if denormalize:
            x = (x + 1.0) / 2.0
            x = x.clamp(0.0, 1.0)

        return x
