from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LinearNormalizer(nn.Module):
    def __init__(
        self,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("obs_mean", obs_mean.float())
        self.register_buffer("obs_std", obs_std.float().clamp_min(1e-6))
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float().clamp_min(1e-6))

    @classmethod
    def identity(cls, obs_dim: int, action_dim: int) -> "LinearNormalizer":
        return cls(
            obs_mean=torch.zeros(obs_dim),
            obs_std=torch.ones(obs_dim),
            action_mean=torch.zeros(action_dim),
            action_std=torch.ones(action_dim),
        )

    @classmethod
    def from_data(cls, observations: np.ndarray, actions: np.ndarray) -> "LinearNormalizer":
        obs = torch.as_tensor(observations.reshape(-1, observations.shape[-1]), dtype=torch.float32)
        act = torch.as_tensor(actions.reshape(-1, actions.shape[-1]), dtype=torch.float32)
        return cls(
            obs_mean=obs.mean(dim=0),
            obs_std=obs.std(dim=0, unbiased=False),
            action_mean=act.mean(dim=0),
            action_std=act.std(dim=0, unbiased=False),
        )

    def copy(self) -> "LinearNormalizer":
        return LinearNormalizer(
            obs_mean=self.obs_mean.detach().cpu().clone(),
            obs_std=self.obs_std.detach().cpu().clone(),
            action_mean=self.action_mean.detach().cpu().clone(),
            action_std=self.action_std.detach().cpu().clone(),
        )

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs_mean = self.obs_mean.to(device=obs.device, dtype=obs.dtype)
        obs_std = self.obs_std.to(device=obs.device, dtype=obs.dtype)
        return (obs - obs_mean) / obs_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        action_std = self.action_std.to(device=action.device, dtype=action.dtype)
        return (action - action_mean) / action_std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        action_std = self.action_std.to(device=action.device, dtype=action.dtype)
        return action * action_std + action_mean
