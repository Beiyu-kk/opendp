from __future__ import annotations

import numpy as np
import torch
from torch import nn

from opendp.diffusion.ddpm import ddpm_loss
from opendp.diffusion.noise_scheduler import DDPMScheduler
from opendp.diffusion.sampler import DDPMSampler
from opendp.models.unet1d import ConditionalUnet1D
from opendp.utils.normalizer import LinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        action_horizon: int,
        model_config: dict | None = None,
        diffusion_config: dict | None = None,
        sampling_config: dict | None = None,
        normalizer: LinearNormalizer | None = None,
    ) -> None:
        super().__init__()
        model_config = model_config or {}
        diffusion_config = diffusion_config or {}
        sampling_config = sampling_config or {}

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = sampling_config.get(
            "num_inference_steps",
            diffusion_config.get("num_train_timesteps", 100),
        )

        self.model = ConditionalUnet1D(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            obs_dim=self.obs_dim,
            obs_horizon=self.obs_horizon,
            **model_config,
        )
        self.scheduler = DDPMScheduler(**diffusion_config)
        self.normalizer = normalizer.copy() if normalizer is not None else LinearNormalizer.identity(
            self.obs_dim,
            self.action_dim,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        normalizer: LinearNormalizer | None = None,
    ) -> "DiffusionPolicy":
        task = config["task"]
        return cls(
            obs_dim=task["obs_dim"],
            action_dim=task["action_dim"],
            obs_horizon=task["obs_horizon"],
            action_horizon=task["action_horizon"],
            model_config=config.get("model", {}),
            diffusion_config=config.get("diffusion", {}),
            sampling_config=config.get("sampling", {}),
            normalizer=normalizer,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return ddpm_loss(
            model=self.model,
            scheduler=self.scheduler,
            clean_actions=batch["actions"],
            obs_history=batch["obs"],
        )

    @torch.no_grad()
    def sample_action_sequence(
        self,
        obs_history: torch.Tensor,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        obs_history = self.normalizer.normalize_obs(obs_history)
        sampler = DDPMSampler(self.scheduler)
        actions = sampler.sample(
            model=self.model,
            obs_history=obs_history,
            action_shape=(obs_history.shape[0], self.action_horizon, self.action_dim),
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            generator=generator,
        )
        actions = self.normalizer.unnormalize_action(actions)
        if was_training:
            self.train()
        return actions

    @torch.no_grad()
    def predict_action_sequence(
        self,
        obs_history: np.ndarray | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        device = torch.device(device)
        obs_tensor = torch.as_tensor(obs_history, dtype=torch.float32, device=device)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        actions = self.sample_action_sequence(obs_tensor)
        return actions[0].detach().cpu().numpy()
