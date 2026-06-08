from __future__ import annotations

import torch
import torch.nn.functional as F

from opendp.diffusion.noise_scheduler import DDPMScheduler


def ddpm_loss(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    clean_actions: torch.Tensor,
    obs_history: torch.Tensor,
) -> torch.Tensor:
    batch_size = clean_actions.shape[0]
    timesteps = torch.randint(
        low=0,
        high=scheduler.num_train_timesteps,
        size=(batch_size,),
        device=clean_actions.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(clean_actions)
    noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
    noise_pred = model(noisy_actions, timesteps, obs_history)
    return F.mse_loss(noise_pred, noise)
