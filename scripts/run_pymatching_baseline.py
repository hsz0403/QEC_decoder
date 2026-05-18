#!/usr/bin/env python
"""Run a PyMatching baseline on a common NPZ dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qecml.data.stim_dataset import load_npz_dataset
from qecml.decoders.pymatching_decoder import build_matching, decode_batch, logical_error_rate
from qecml.sim.stim_surface_code import build_rotated_surface_code_circuit
from qecml.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = load_npz_dataset(args.data)
    metadata = dataset["metadata"]
    if metadata.get("source") != "stim":
        raise ValueError("PyMatching baseline currently reconstructs only Stim-generated datasets")
    circuit = build_rotated_surface_code_circuit(
        basis=metadata["basis"],
        distance=int(metadata["distance"]),
        rounds=int(metadata["rounds"]),
        p=float(metadata["p"]),
        noise_model=metadata.get("noise_model", "circuit_depolarizing"),
    )
    matching = build_matching(circuit)
    pred = decode_batch(matching, dataset["events"])
    labels = dataset["labels"]
    ler = logical_error_rate(pred, labels)
    mistakes = int(np.sum(np.any(pred != labels, axis=1)))
    result = {
        "pymatching_ler": ler,
        "num_shots": int(labels.shape[0]),
        "num_mistakes": mistakes,
        "prediction_shape": list(pred.shape),
        "label_shape": list(labels.shape),
        "data": args.data,
    }
    output = args.output
    if output is None:
        stem = Path(args.data).stem.replace("_val", "").replace("_train", "")
        output = f"runs/baselines/pymatching_{stem}.json"
    save_json(output, result)
    print(result)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
