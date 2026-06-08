from __future__ import annotations

import torch
from torch import nn

from opendp.models.condition_encoder import ConditionEncoder
from opendp.models.time_embedding import TimeEmbedding


class ConditionalUnet1D(nn.Module):
    """A compact conditional denoiser with the UNet-facing API.

    This minimal project keeps the model intentionally small: it flattens the
    action sequence, conditions on flattened state history and timestep
    embeddings, and predicts noise with an MLP.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        obs_dim: int,
        obs_horizon: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        condition_embed_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.obs_dim = int(obs_dim)
        self.obs_horizon = int(obs_horizon)
        self.action_flat_dim = self.action_dim * self.action_horizon

        self.time_encoder = TimeEmbedding(time_embed_dim, hidden_dim=hidden_dim)
        self.condition_encoder = ConditionEncoder(
            input_dim=self.obs_dim * self.obs_horizon,
            embed_dim=condition_embed_dim,
            hidden_dim=hidden_dim,
        )

        input_dim = self.action_flat_dim + time_embed_dim + condition_embed_dim
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(max(1, int(num_layers))):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.Mish()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.action_flat_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        obs_history: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        action_features = noisy_actions.reshape(batch_size, -1)
        time_features = self.time_encoder(timesteps)
        condition_features = self.condition_encoder(obs_history)
        pred = self.net(torch.cat([action_features, time_features, condition_features], dim=-1))
        return pred.reshape(batch_size, self.action_horizon, self.action_dim)


StateActionNoisePredictor = ConditionalUnet1D
