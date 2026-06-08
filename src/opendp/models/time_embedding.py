from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps[:, None]

        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - 1, 1)
        emb = timesteps[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.net(timesteps)
