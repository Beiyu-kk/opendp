from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from opendp.policies.diffusion_policy import DiffusionPolicy


def save_checkpoint(
    path: str | Path,
    policy: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "config": config,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_policy_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DiffusionPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = DiffusionPolicy.from_config(checkpoint["config"])
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy, checkpoint
