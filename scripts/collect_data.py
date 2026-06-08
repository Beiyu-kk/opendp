from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from opendp.envs.sim_env import collect_expert_dataset
from opendp.utils.config import load_yaml


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect toy state-action demonstrations.")
    parser.add_argument("--task-config", default="configs/task_book_grasp.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episode-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = load_yaml(resolve_path(args.task_config))
    dataset_cfg = task.get("dataset", {})
    expert_cfg = task.get("expert", {})

    output_path = resolve_path(args.output or dataset_cfg.get("path", "data/book_grasp_state.npz"))
    num_episodes = args.episodes or dataset_cfg.get("num_episodes", 128)
    episode_len = args.episode_len or task.get("episode_len", 48)

    path = collect_expert_dataset(
        output_path=output_path,
        num_episodes=num_episodes,
        episode_len=episode_len,
        env_config=task,
        seed=args.seed,
        kp=expert_cfg.get("kp", 0.45),
        noise_std=expert_cfg.get("noise_std", 0.0),
    )
    print(f"Saved dataset to {path}")


if __name__ == "__main__":
    main()
