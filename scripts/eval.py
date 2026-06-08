from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np

from opendp.envs.rollout import rollout_policy
from opendp.envs.sim_env import ToyBookGraspEnv
from opendp.utils.checkpoint import load_policy_checkpoint
from opendp.utils.config import load_config
from opendp.utils.device import get_device
from opendp.utils.seed import set_seed


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a state-only Diffusion Policy checkpoint.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config), args.task_config)
    seed = args.seed if args.seed is not None else config.get("seed", 0) + 10_000
    set_seed(seed)

    checkpoint_path = resolve_path(
        args.checkpoint or config.get("training", {}).get("checkpoint_path", "checkpoints/state_diffusion_policy.pt")
    )
    device = get_device(config.get("device", "auto"))
    policy, checkpoint = load_policy_checkpoint(checkpoint_path, device=device)
    run_config = checkpoint["config"]

    eval_cfg = run_config.get("eval", {})
    episodes = args.episodes or eval_cfg.get("episodes", 8)
    max_steps = eval_cfg.get("max_steps", run_config["task"].get("max_steps", 48))
    output_path = resolve_path(eval_cfg.get("output_path", "outputs/eval_rollout.json"))

    results = []
    for episode_idx in range(episodes):
        env = ToyBookGraspEnv.from_config(run_config["task"])
        result = rollout_policy(
            env=env,
            policy=policy,
            obs_horizon=run_config["task"]["obs_horizon"],
            max_steps=max_steps,
            device=str(device),
            seed=seed + episode_idx,
        )
        results.append(result)

    success_rate = float(np.mean([r["success"] for r in results]))
    final_distance = float(np.mean([r["final_distance"] for r in results]))
    summary = {
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "success_rate": success_rate,
        "mean_final_distance": final_distance,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded checkpoint epoch {checkpoint.get('epoch')}: {checkpoint_path}")
    print(f"success_rate={success_rate:.3f} mean_final_distance={final_distance:.4f}")
    print(f"Saved rollout summary to {output_path}")


if __name__ == "__main__":
    main()
