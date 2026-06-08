from __future__ import annotations

import torch

from opendp.diffusion.noise_scheduler import DDPMScheduler


class DDPMSampler:
    def __init__(self, scheduler: DDPMScheduler) -> None:
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        obs_history: torch.Tensor,
        action_shape: tuple[int, int, int],
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        sample = torch.randn(
            action_shape,
            device=obs_history.device,
            dtype=obs_history.dtype,
            generator=generator,
        )
        for t in self.scheduler.inference_timesteps(num_inference_steps):
            timesteps = torch.full(
                (action_shape[0],),
                t,
                device=obs_history.device,
                dtype=torch.long,
            )
            noise_pred = model(sample, timesteps, obs_history)
            sample = self.scheduler.step(noise_pred, timesteps, sample, generator=generator)
        return sample
