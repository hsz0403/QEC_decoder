#!/usr/bin/env python
"""Train a CNN decoder on surface-code code-capacity iid data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_code_capacity import SurfaceCodeCNNDecoder, load_surface_code_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(labels.dtype)
    bit_accuracy = (preds == labels).float().mean()
    logical_error_rate = (preds != labels).any(dim=1).float().mean()
    return {
        "bce": float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).cpu()),
        "bit_accuracy": float(bit_accuracy.cpu()),
        "logical_error_rate": float(logical_error_rate.cpu()),
        "label_positive_rate": float(labels.mean().cpu()),
        "prediction_positive_rate": float(preds.mean().cpu()),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    events: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits = []
    loader = DataLoader(TensorDataset(events, labels), batch_size=batch_size)
    for batch_events, _ in loader:
        logits.append(model(batch_events.to(device)).cpu())
    return _metrics(torch.cat(logits, dim=0), labels)


def _pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("training", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "num_threads" in cfg.get("training", {}):
        torch.set_num_threads(int(cfg["training"]["num_threads"]))

    train_ds = load_surface_code_dataset(cfg["data"]["train"])
    val_ds = load_surface_code_dataset(cfg["data"]["val"])
    train_events = torch.from_numpy(train_ds["events"]).long()
    train_labels = torch.from_numpy(train_ds["labels"]).float()
    val_events = torch.from_numpy(val_ds["events"]).long()
    val_labels = torch.from_numpy(val_ds["labels"]).float()

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg.get("model", {})
    model = SurfaceCodeCNNDecoder(
        detector_grid=torch.from_numpy(train_ds["detector_grid"]),
        grid_shape=tuple(int(x) for x in train_ds["metadata"]["grid_shape"]),
        logical_mask=torch.from_numpy(train_ds["logical_mask"]),
        num_observables=int(train_labels.shape[1]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        bottleneck_ratio=int(model_cfg.get("bottleneck_ratio", 4)),
        dropout=float(model_cfg.get("dropout", 0.05)),
    ).to(device)

    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 1024))
    epochs = int(train_cfg.get("epochs", 20))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    pos_weight = _pos_weight(train_labels).to(device) if train_cfg.get("pos_weight") == "auto" else None

    run_dir = Path(cfg.get("output", {}).get("run_dir", "runs/surface_cnn"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoint.pt"
    resume = bool(train_cfg.get("resume", False))
    start_epoch = 1
    best_ler = float("inf")
    if resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            start_epoch = int(rows[-1]["epoch"]) + 1
            best_ler = min(float(row["val_logical_error_rate"]) for row in rows)
    elif metrics_path.exists():
        metrics_path.unlink()

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_ler = min(best_ler, float(checkpoint.get("best_val_logical_error_rate", best_ler)))

    loader = DataLoader(
        TensorDataset(train_events, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=bool(device.type == "cuda"),
    )
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        for batch_events, batch_labels in tqdm(
            loader,
            desc=f"epoch {epoch}",
            leave=False,
            disable=not bool(train_cfg.get("progress", True)),
        ):
            batch_events = batch_events.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_events)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch_labels,
                pos_weight=pos_weight,
            )
            loss.backward()
            grad_clip = train_cfg.get("grad_clip_norm", 1.0)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(
            model,
            val_events,
            val_labels,
            batch_size=batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["logical_error_rate"] <= best_ler:
            best_ler = val_metrics["logical_error_rate"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "metadata": train_ds["metadata"],
                    "best_val_logical_error_rate": best_ler,
                },
                checkpoint_path,
            )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "best_val_logical_error_rate": best_ler,
        "final_val": evaluate(model, val_events, val_labels, batch_size=batch_size, device=device),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
