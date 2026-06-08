from __future__ import annotations

from opendp.envs.base_env import BaseEnv


class RealRobotEnv(BaseEnv):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Real robot control is not implemented in the minimal build.")
