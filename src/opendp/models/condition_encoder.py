from __future__ import annotations

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Mish(),
        )

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.net(obs_history.flatten(start_dim=1))
