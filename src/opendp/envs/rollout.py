from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from opendp.envs.base_env import BaseEnv
from opendp.policies.diffusion_policy import DiffusionPolicy


def rollout_policy(
    env: BaseEnv,
    policy: DiffusionPolicy,
    obs_horizon: int,
    max_steps: int,
    device: str,
    seed: int | None = None,
) -> dict[str, Any]:
    obs = env.reset(seed=seed)
    history = deque([obs.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
    rewards: list[float] = []
    trajectory: list[dict[str, Any]] = []
    done = False
    info: dict[str, Any] = {}

    for step_idx in range(max_steps):
        obs_history = np.stack(history, axis=0)
        action_sequence = policy.predict_action_sequence(obs_history, device=device)
        action = action_sequence[0]
        next_obs, reward, done, info = env.step(action)

        trajectory.append(
            {
                "step": step_idx,
                "obs": obs.tolist(),
                "action": action.tolist(),
                "reward": float(reward),
                "distance": float(info.get("distance", 0.0)),
                "success": bool(info.get("success", False)),
            }
        )
        rewards.append(float(reward))
        obs = next_obs
        history.append(obs.copy())
        if done:
            break

    return {
        "success": bool(info.get("success", False)),
        "steps": len(rewards),
        "return": float(sum(rewards)),
        "final_distance": float(info.get("distance", 0.0)),
        "trajectory": trajectory,
    }
