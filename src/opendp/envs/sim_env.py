from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from opendp.envs.base_env import BaseEnv


class ToyBookGraspEnv(BaseEnv):
    """A tiny 2D reaching task used to exercise the policy pipeline."""

    obs_dim = 4
    action_dim = 2

    def __init__(
        self,
        workspace: float = 1.0,
        max_action: float = 0.08,
        success_tolerance: float = 0.035,
        max_steps: int = 48,
    ) -> None:
        self.workspace = float(workspace)
        self.max_action = float(max_action)
        self.success_tolerance = float(success_tolerance)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng()
        self.position = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ToyBookGraspEnv":
        return cls(
            workspace=config.get("workspace", 1.0),
            max_action=config.get("max_action", 0.08),
            success_tolerance=config.get("success_tolerance", 0.035),
            max_steps=config.get("max_steps", config.get("episode_len", 48)),
        )

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        low = -0.75 * self.workspace
        high = 0.75 * self.workspace
        self.position = self.rng.uniform(low, high, size=2).astype(np.float32)
        self.target = self.rng.uniform(low, high, size=2).astype(np.float32)
        while np.linalg.norm(self.target - self.position) < 0.4 * self.workspace:
            self.target = self.rng.uniform(low, high, size=2).astype(np.float32)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_action, self.max_action)
        self.position = np.clip(self.position + action, -self.workspace, self.workspace)
        self.step_count += 1

        distance = float(np.linalg.norm(self.target - self.position))
        success = distance <= self.success_tolerance
        done = success or self.step_count >= self.max_steps
        reward = -distance
        return self._obs(), reward, done, {"distance": distance, "success": success}

    def expert_action(
        self,
        kp: float = 0.45,
        noise_std: float = 0.0,
    ) -> np.ndarray:
        delta = self.target - self.position
        action = kp * delta
        norm = float(np.linalg.norm(action))
        if norm > self.max_action:
            action = action / norm * self.max_action
        if noise_std > 0:
            action = action + self.rng.normal(0.0, noise_std, size=2)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.position, self.target]).astype(np.float32)


def collect_expert_dataset(
    output_path: str | Path,
    num_episodes: int,
    episode_len: int,
    env_config: dict[str, Any] | None = None,
    seed: int = 0,
    kp: float = 0.45,
    noise_std: float = 0.0,
) -> Path:
    env = ToyBookGraspEnv.from_config(env_config or {})
    observations = np.zeros((num_episodes, episode_len, env.obs_dim), dtype=np.float32)
    actions = np.zeros((num_episodes, episode_len, env.action_dim), dtype=np.float32)

    for episode_idx in range(num_episodes):
        obs = env.reset(seed=seed + episode_idx)
        for step_idx in range(episode_len):
            action = env.expert_action(kp=kp, noise_std=noise_std)
            observations[episode_idx, step_idx] = obs
            actions[episode_idx, step_idx] = action
            obs, _, _, _ = env.step(action)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "env": "ToyBookGraspEnv",
        "num_episodes": int(num_episodes),
        "episode_len": int(episode_len),
        "seed": int(seed),
    }
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        metadata=json.dumps(metadata),
    )
    return output_path
