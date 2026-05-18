#!/usr/bin/env python
"""Generate train/val Stim surface-code datasets."""

from __future__ import annotations

import argparse

import numpy as np

from qecml.data.stim_dataset import save_npz_dataset
from qecml.sim.stim_surface_code import (
    build_rotated_surface_code_circuit,
    circuit_metadata,
    get_detector_coordinates,
    sample_detector_events,
)
from qecml.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    circuit = build_rotated_surface_code_circuit(
        basis=cfg["basis"],
        distance=int(cfg["distance"]),
        rounds=int(cfg["rounds"]),
        p=float(cfg["p"]),
        noise_model=cfg.get("noise_model", "circuit_depolarizing"),
        seed=cfg.get("seed"),
    )
    coords = get_detector_coordinates(circuit)
    metadata = circuit_metadata(
        basis=cfg["basis"],
        distance=int(cfg["distance"]),
        rounds=int(cfg["rounds"]),
        p=float(cfg["p"]),
        noise_model=cfg.get("noise_model", "circuit_depolarizing"),
        extra={
            "n_detectors": circuit.num_detectors,
            "n_observables": circuit.num_observables,
            "generator_config": cfg,
        },
    )
    for split, shots_key, output_key, seed_offset in [
        ("train", "shots_train", "output_train", 0),
        ("val", "shots_val", "output_val", 1),
    ]:
        events, labels = sample_detector_events(
            circuit, shots=int(cfg[shots_key]), seed=int(cfg.get("seed", 0)) + seed_offset
        )
        split_meta = {**metadata, "split": split, "shots": int(cfg[shots_key])}
        save_npz_dataset(
            cfg[output_key],
            events=events,
            labels=labels,
            detector_coords=coords,
            metadata=split_meta,
        )
        print(f"wrote {cfg[output_key]}")
        print(f"  n_detectors: {events.shape[1]}")
        print(f"  n_observables: {labels.shape[1]}")
        print(f"  label positive rate: {float(np.mean(labels)):.6f}")
        print(f"  detector coordinates: {'yes' if coords.size else 'no'}")
        print(
            "  metadata: "
            f"basis={cfg['basis']} distance={cfg['distance']} "
            f"rounds={cfg['rounds']} p={cfg['p']}"
        )


if __name__ == "__main__":
    main()
