#!/usr/bin/env python
"""Plot training curves for the surface-code CNN experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/surface_cnn_d11_p005_h128",
            "runs/surface_cnn_d11_p005_h512",
        ],
        help="Run directories containing metrics.jsonl.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["H=128", "H=512"],
        help="Legend labels matching --runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/surface_code_training_curves.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict[str, float]]:
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    if len(args.runs) != len(args.labels):
        raise ValueError("--runs and --labels must have the same length")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for run, label in zip(args.runs, args.labels):
        rows = load_rows(Path(run))
        epochs = [row["epoch"] for row in rows]
        train_loss = [row["train_loss"] for row in rows]
        val_bce = [row["val_bce"] for row in rows]
        val_ler = [row["val_logical_error_rate"] for row in rows]
        pred_pos = [row["val_prediction_positive_rate"] for row in rows]

        axes[0].plot(epochs, train_loss, marker="o", markersize=2.5, label=label)
        axes[1].plot(epochs, val_bce, marker="o", markersize=2.5, label=label)
        axes[2].plot(epochs, val_ler, marker="o", markersize=2.5, label=f"{label} LER")
        axes[2].plot(
            epochs,
            pred_pos,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} pred-positive",
        )

    axes[2].axhline(0.23632, color="black", linestyle=":", linewidth=1.2, label="all-zero baseline")
    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation BCE")
    axes[2].set_title("Validation LER and Positive Rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("BCEWithLogits loss")
    axes[1].set_ylabel("BCE")
    axes[2].set_ylabel("rate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(args.out)


if __name__ == "__main__":
    main()
