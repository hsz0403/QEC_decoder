"""Training loop for neural QEC decoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from qecml.data.coordinate_mapper import try_events_to_dense
from qecml.data.stim_dataset import load_npz_dataset
from qecml.decoders.neural.mini_alphaqubit import MiniAlphaQubitDecoder
from qecml.decoders.neural.mlp import FlatMLPDecoder
from qecml.decoders.neural.temporal_cnn import TemporalCNNDecoder
from qecml.training.losses import bce_with_logits
from qecml.training.metrics import metrics_from_logits
from qecml.utils.io import append_jsonl, save_yaml
from qecml.utils.seed import set_seed


def prepare_tensors(
    dataset: dict[str, Any],
    *,
    dense: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare event and label tensors from a loaded NPZ dataset."""
    events = dataset["events"]
    labels = dataset["labels"]
    metadata = dict(dataset.get("metadata") or {})
    if dense:
        dense_events, mask, mapping = try_events_to_dense(events, dataset.get("detector_coords"))
        if dense_events is None or mask is None or mapping is None:
            raise ValueError("dense model requested but coordinate mapping was unavailable")
        metadata["dense_shape"] = list(dense_events.shape[1:])
        metadata["num_sites"] = dense_events.shape[2]
        metadata["num_features"] = dense_events.shape[3]
        metadata["coordinate_time_axis"] = mapping.time_axis
        return torch.from_numpy(dense_events).float(), torch.from_numpy(labels).float(), metadata
    return torch.from_numpy(events).float(), torch.from_numpy(labels).float(), metadata


def build_model_from_config(
    model_config: dict[str, Any],
    *,
    train_events: torch.Tensor,
    train_labels: torch.Tensor,
) -> torch.nn.Module:
    """Instantiate a decoder model from config and dataset shapes."""
    name = model_config.get("name", "mlp")
    num_observables = int(train_labels.shape[1])
    hidden_dim = int(model_config.get("hidden_dim", 128))
    dropout = float(model_config.get("dropout", 0.1))
    if name in {"mlp", "flat_mlp"}:
        return FlatMLPDecoder(
            num_detectors=int(train_events.flatten(start_dim=1).shape[1]),
            num_observables=num_observables,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if name in {"temporal_cnn", "cnn"}:
        if train_events.ndim != 4:
            raise ValueError("temporal_cnn requires dense [B, T, S, F] events")
        return TemporalCNNDecoder(
            num_sites=int(train_events.shape[2]),
            num_features=int(train_events.shape[3]),
            num_observables=num_observables,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if name in {"mini_alphaqubit", "mini_aq"}:
        if train_events.ndim != 4:
            raise ValueError("mini_alphaqubit requires dense [B, T, S, F] events")
        return MiniAlphaQubitDecoder(
            num_sites=int(train_events.shape[2]),
            num_features=int(train_events.shape[3]),
            num_observables=num_observables,
            hidden_dim=hidden_dim,
            num_layers=int(model_config.get("num_layers", 2)),
            num_heads=int(model_config.get("num_heads", 4)),
            dropout=dropout,
        )
    raise ValueError(f"unknown model name: {name}")


def _compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    events: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 512,
    device: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    """Evaluate a model on a tensor dataset."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(events, labels), batch_size=batch_size)
    for batch_events, _ in loader:
        logits = model(batch_events.to(device))
        all_logits.append(logits.cpu())
    return metrics_from_logits(torch.cat(all_logits, dim=0), labels)


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Train a neural decoder from a parsed YAML config."""
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 0))
    set_seed(seed)
    if "num_threads" in train_cfg:
        torch.set_num_threads(int(train_cfg["num_threads"]))
    elif not torch.cuda.is_available():
        torch.set_num_threads(1)
    model_name = config.get("model", {}).get("name", "mlp")
    dense = model_name in {"mini_alphaqubit", "mini_aq", "temporal_cnn", "cnn"}

    train_ds = load_npz_dataset(config["data"]["train"])
    val_ds = load_npz_dataset(config["data"]["val"])
    train_events, train_labels, train_meta = prepare_tensors(train_ds, dense=dense)
    val_events, val_labels, _ = prepare_tensors(val_ds, dense=dense)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(
        config.get("model", {}), train_events=train_events, train_labels=train_labels
    ).to(device)

    batch_size = int(train_cfg.get("batch_size", 256))
    epochs = int(train_cfg.get("epochs", 10))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    pos_weight_cfg = train_cfg.get("pos_weight", None)
    if pos_weight_cfg == "auto":
        pos_weight = _compute_pos_weight(train_labels).to(device)
    elif pos_weight_cfg is None:
        pos_weight = None
    else:
        pos_weight = torch.tensor(pos_weight_cfg, dtype=torch.float32, device=device)

    run_dir = Path(config.get("output", {}).get("run_dir", "runs/neural"))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", config)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_loader = DataLoader(
        TensorDataset(train_events, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)

    best_val_ler = float("inf")
    checkpoint_path = run_dir / "checkpoint.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_events, batch_labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch_events = batch_events.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_events)
            loss = bce_with_logits(logits, batch_labels, pos_weight=pos_weight)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(
            model, val_events, val_labels, batch_size=batch_size, device=device
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_jsonl(metrics_path, row)
        if val_metrics["logical_error_rate"] <= best_val_ler:
            best_val_ler = float(val_metrics["logical_error_rate"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "metadata": train_meta,
                    "model_name": model_name,
                    "train_event_shape": list(train_events.shape[1:]),
                    "num_observables": int(train_labels.shape[1]),
                },
                checkpoint_path,
            )
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "best_val_ler": best_val_ler,
    }
