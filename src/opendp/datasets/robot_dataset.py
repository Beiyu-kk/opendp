from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from opendp.utils.normalizer import LinearNormalizer


class RobotSequenceDataset(Dataset):
    """State/action sequence dataset for state-only Diffusion Policy."""

    def __init__(
        self,
        data_path: str | Path,
        obs_horizon: int,
        action_horizon: int,
        normalizer: LinearNormalizer | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        raw = np.load(self.data_path, allow_pickle=False)
        observations = raw["observations"] if "observations" in raw else raw["states"]
        actions = raw["actions"]

        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("Expected observations/actions with shape [episodes, steps, dim].")
        if observations.shape[:2] != actions.shape[:2]:
            raise ValueError("Observations and actions must have matching episode/step axes.")

        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.num_episodes, self.episode_len, self.obs_dim = self.observations.shape
        self.action_dim = self.actions.shape[-1]

        if self.episode_len < self.obs_horizon + self.action_horizon - 1:
            raise ValueError("Episode length is too short for requested horizons.")

        self.normalizer = normalizer or LinearNormalizer.from_data(self.observations, self.actions)
        self.indices = [
            (episode_idx, t)
            for episode_idx in range(self.num_episodes)
            for t in range(self.obs_horizon - 1, self.episode_len - self.action_horizon + 1)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_idx, t = self.indices[index]
        obs_seq = self.observations[
            episode_idx,
            t - self.obs_horizon + 1 : t + 1,
        ]
        action_seq = self.actions[
            episode_idx,
            t : t + self.action_horizon,
        ]

        obs = torch.from_numpy(obs_seq)
        actions = torch.from_numpy(action_seq)
        return {
            "obs": self.normalizer.normalize_obs(obs),
            "actions": self.normalizer.normalize_action(actions),
        }
