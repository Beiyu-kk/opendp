from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path, task_config_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    task_path = task_config_path or config.get("task_config")
    if task_path is not None:
        task_path = Path(task_path)
        if not task_path.is_absolute():
            task_path = config_path.parent.parent / task_path
        config["task"] = load_yaml(task_path)
        config["task_config"] = str(task_path)
    return config
