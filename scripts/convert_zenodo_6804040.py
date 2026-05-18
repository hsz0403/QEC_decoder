#!/usr/bin/env python
"""Convert one Zenodo 6804040 experiment into the common NPZ schema."""

from __future__ import annotations

import argparse

from qecml.data.zenodo_6804040 import convert_experiment_to_npz, experiment_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default="data/zenodo_6804040/raw/google_qec3v5_experiment_data.zip",
    )
    parser.add_argument("--experiment")
    parser.add_argument("--output")
    parser.add_argument("--skip-shots", type=int, default=0)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in experiment_names(args.zip):
            print(name)
        return

    if not args.experiment:
        raise SystemExit("--experiment is required unless --list is set")
    output = args.output or f"runs/data/zenodo_6804040_{args.experiment}.npz"
    result = convert_experiment_to_npz(
        args.zip,
        args.experiment,
        output,
        skip_shots=args.skip_shots,
        max_shots=args.max_shots,
    )
    print(result)


if __name__ == "__main__":
    main()
