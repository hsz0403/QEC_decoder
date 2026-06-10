#!/usr/bin/env python
"""Create simple visual examples from a surface-code iid dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_code_capacity import load_surface_code_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/surface_code_examples"))
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def _syndrome_grid(events: np.ndarray, detector_grid: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=np.uint8)
    for detector_idx, (y, x) in enumerate(detector_grid):
        grid[y, x] |= events[detector_idx]
    return grid


def _plot_syndrome_grid(
    grid: np.ndarray,
    *,
    title: str,
    path: Path,
) -> None:
    """Save a high-contrast detector grid.

    A raw imshow of a sparse 10x12 binary array is technically correct but looks
    nearly blank in GitHub previews. This draws active detectors as large red
    square markers so the syndrome pattern is visible at a glance.
    """
    height, width = grid.shape
    ys, xs = np.where(grid == 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(np.zeros_like(grid), cmap="gray", vmin=0, vmax=1)
    if len(xs):
        ax.scatter(xs, ys, s=190, marker="s", c="#d62728", edgecolors="black", linewidths=0.7)
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.7)
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ds = load_surface_code_dataset(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = ds["events"]
    labels = ds["labels"]
    detector_grid = ds["detector_grid"]
    grid_shape = tuple(int(x) for x in ds["metadata"]["grid_shape"])

    positive = np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]
    negative = np.where(~labels.reshape(labels.shape[0], -1).any(axis=1))[0]
    chosen = []
    chosen.extend(positive[: args.num_samples // 2].tolist())
    chosen.extend(negative[: args.num_samples - len(chosen)].tolist())
    if len(chosen) < args.num_samples:
        chosen.extend(list(range(min(args.num_samples - len(chosen), events.shape[0]))))

    rows = []
    for panel_idx, sample_idx in enumerate(chosen[: args.num_samples]):
        grid = _syndrome_grid(events[sample_idx], detector_grid, grid_shape)
        path = args.out_dir / f"sample_{panel_idx:02d}_idx_{sample_idx}.png"
        _plot_syndrome_grid(
            grid,
            title=(
                f"sample {sample_idx}, label={labels[sample_idx].astype(int).tolist()}, "
                f"active={int(events[sample_idx].sum())}"
            ),
            path=path,
        )
        rows.append(
            {
                "sample_index": int(sample_idx),
                "active_detectors": int(events[sample_idx].sum()),
                "label": labels[sample_idx].astype(int).tolist(),
                "image": str(path),
            }
        )

    summary = {
        "data": str(args.data),
        "metadata": ds["metadata"],
        "events_shape": list(events.shape),
        "labels_shape": list(labels.shape),
        "label_positive_rate": labels.mean(axis=0).astype(float).tolist(),
        "examples": rows,
    }
    with (args.out_dir / "examples_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
