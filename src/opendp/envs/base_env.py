from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnv(ABC):
    obs_dim: int
    action_dim: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        raise NotImplementedError
