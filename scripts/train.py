from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opendp.datasets.robot_dataset import RobotSequenceDataset
from opendp.envs.sim_env import collect_expert_dataset
from opendp.policies.diffusion_policy import DiffusionPolicy
from opendp.utils.checkpoint import save_checkpoint
from opendp.utils.config import load_config
from opendp.utils.device import get_device
from opendp.utils.seed import set_seed


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dataset(config: dict, seed: int) -> Path:
    task = config["task"]
    dataset_cfg = task.get("dataset", {})
    expert_cfg = task.get("expert", {})
    data_path = resolve_path(dataset_cfg.get("path", "data/book_grasp_state.npz"))
    auto_generate = config.get("data", {}).get("auto_generate", True)

    if auto_generate or not data_path.exists():
        collect_expert_dataset(
            output_path=data_path,
            num_episodes=dataset_cfg.get("num_episodes", 128),
            episode_len=task.get("episode_len", 48),
            env_config=task,
            seed=seed,
            kp=expert_cfg.get("kp", 0.45),
            noise_std=expert_cfg.get("noise_std", 0.0),
        )
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal state-only Diffusion Policy.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(resolve_path(args.config), args.task_config)
    seed = int(config.get("seed", 0))
    set_seed(seed)

    task = config["task"]
    training_cfg = config.get("training", {})
    device = get_device(config.get("device", "auto"))
    data_path = ensure_dataset(config, seed=seed)

    dataset = RobotSequenceDataset(
        data_path=data_path,
        obs_horizon=task["obs_horizon"],
        action_horizon=task["action_horizon"],
    )
    loader = DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=False,
    )

    policy = DiffusionPolicy.from_config(config, normalizer=dataset.normalizer).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    epochs = args.epochs or training_cfg.get("epochs", 8)
    log_every = training_cfg.get("log_every", 20)
    checkpoint_path = resolve_path(args.checkpoint or training_cfg.get("checkpoint_path", "checkpoints/state_diffusion_policy.pt"))

    global_step = 0
    last_avg_loss = 0.0
    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = policy.compute_loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            total_loss += float(loss.item())
            if global_step % log_every == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        last_avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch={epoch} avg_loss={last_avg_loss:.6f}")

    path = save_checkpoint(
        path=checkpoint_path,
        policy=policy,
        optimizer=optimizer,
        epoch=epochs,
        config=config,
        metrics={"avg_loss": last_avg_loss, "global_step": global_step},
    )
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
