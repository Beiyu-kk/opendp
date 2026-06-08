from __future__ import annotations

import torch
from torch import nn


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)

        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

    def add_noise(
        self,
        clean_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, clean_samples.shape)
        sqrt_one_minus_alpha = _extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            clean_samples.shape,
        )
        return sqrt_alpha * clean_samples + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if isinstance(timestep, int):
            timesteps = torch.full(
                (sample.shape[0],),
                timestep,
                device=sample.device,
                dtype=torch.long,
            )
        else:
            timesteps = timestep.to(device=sample.device, dtype=torch.long)

        beta_t = _extract(self.betas, timesteps, sample.shape)
        sqrt_one_minus_alpha_bar_t = _extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            sample.shape,
        )
        sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, timesteps, sample.shape)
        model_mean = sqrt_recip_alpha_t * (sample - beta_t * noise_pred / sqrt_one_minus_alpha_bar_t)

        variance = _extract(self.posterior_variance, timesteps, sample.shape).clamp_min(1e-20)
        noise = torch.randn(
            sample.shape,
            device=sample.device,
            dtype=sample.dtype,
            generator=generator,
        )
        nonzero_mask = (timesteps != 0).to(sample.dtype).reshape(
            sample.shape[0],
            *((1,) * (sample.ndim - 1)),
        )
        return model_mean + nonzero_mask * torch.sqrt(variance) * noise

    def inference_timesteps(self, num_inference_steps: int | None = None) -> list[int]:
        if num_inference_steps is None or num_inference_steps >= self.num_train_timesteps:
            return list(range(self.num_train_timesteps - 1, -1, -1))
        steps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            int(num_inference_steps),
            dtype=torch.long,
        )
        return [int(t) for t in steps.tolist()]
