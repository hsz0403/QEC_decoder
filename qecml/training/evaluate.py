"""Checkpoint evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from qecml.data.stim_dataset import load_npz_dataset
from qecml.training.trainer import build_model_from_config, evaluate_model, prepare_tensors


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    data_path: Union[str, Path],
    *,
    batch_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on a common NPZ dataset."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model_name = config.get("model", {}).get("name", "mlp")
    dense = model_name in {"mini_alphaqubit", "mini_aq", "temporal_cnn", "cnn"}
    dataset = load_npz_dataset(data_path)
    events, labels, _ = prepare_tensors(dataset, dense=dense)
    model = build_model_from_config(
        config.get("model", {}),
        train_events=events,
        train_labels=labels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    metrics = evaluate_model(model, events, labels, batch_size=batch_size, device=device)
    metrics["num_shots"] = int(labels.shape[0])
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["data"] = str(data_path)
    return metrics
