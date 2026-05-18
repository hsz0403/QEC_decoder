"""Small file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import yaml


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Union[str, Path], data: dict[str, Any]) -> None:
    """Save a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def append_jsonl(path: Union[str, Path], row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def save_json(path: Union[str, Path], data: dict[str, Any]) -> None:
    """Save a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
