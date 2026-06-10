#!/usr/bin/env python
"""Generate code-capacity surface-code iid-error datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surface_code_capacity import build_surface_code_iid_dataset, save_surface_code_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = {
        "distance": int(cfg.get("distance", 11)),
        "p": float(cfg.get("p", 0.05)),
        "basis": str(cfg.get("basis", "z")),
        "store_errors": bool(cfg.get("store_errors", False)),
    }
    seed = int(cfg.get("seed", 0))
    outputs = [
        ("train", int(cfg["shots_train"]), Path(cfg["output_train"]), seed),
        ("val", int(cfg["shots_val"]), Path(cfg["output_val"]), seed + 1),
    ]
    for split, shots, output, split_seed in outputs:
        dataset = build_surface_code_iid_dataset(shots=shots, seed=split_seed, **common)
        dataset["metadata"]["split"] = split
        save_surface_code_dataset(output, dataset)
        print(json.dumps({"split": split, "output": str(output), **dataset["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
