"""Command-line entry point for BB-code training-data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .codes import KNOWN_BB_CODES, BBCode, build_bb_code
from .sim import SimulationResult, simulate_circuit_level, simulate_phenomenological


def _compression(name: str) -> str | None:
    return None if name == "none" else name


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _analog_channels(
    measurements: np.ndarray,
    *,
    snr: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = 1.0 / snr
    analog = measurements.astype(np.float32) + rng.normal(
        0.0,
        sigma,
        size=measurements.shape,
    ).astype(np.float32)
    logit = (2.0 * analog - 1.0) / (2.0 * sigma * sigma)
    soft = _sigmoid(logit).astype(np.float32)
    return analog, soft


def _term_json(terms: tuple[tuple[str, int], ...]) -> str:
    return json.dumps([[axis, power] for axis, power in terms])


def _write_code_group(h5: h5py.File, code: BBCode, compression: str | None) -> None:
    group = h5.create_group("code")
    group.attrs["name"] = code.name
    group.attrs["label"] = code.label
    group.attrs["ell"] = code.ell
    group.attrs["m"] = code.m
    group.attrs["a_terms"] = _term_json(code.a_terms)
    group.attrs["b_terms"] = _term_json(code.b_terms)
    group.attrs["n_data"] = code.n_data
    group.attrs["n_checks"] = code.n_checks
    group.attrs["k"] = code.k
    group.create_dataset("hx", data=code.hx, compression=compression)
    group.create_dataset("hz", data=code.hz, compression=compression)
    group.create_dataset("hx_supports", data=code.hx_supports, compression=compression)
    group.create_dataset("hz_supports", data=code.hz_supports, compression=compression)
    group.create_dataset("x_logicals", data=code.x_logicals, compression=compression)
    group.create_dataset("z_logicals", data=code.z_logicals, compression=compression)


def _create_datasets(
    h5: h5py.File,
    *,
    shots: int,
    cycles: int,
    code: BBCode,
    compression: str | None,
    chunks: int,
    store_errors: bool,
    analog_snr: float | None,
) -> dict[str, h5py.Dataset]:
    chunk_rows = min(chunks, shots)
    datasets = {
        "measurements": h5.create_dataset(
            "measurements",
            shape=(shots, cycles, code.n_checks),
            dtype="u1",
            chunks=(chunk_rows, cycles, code.n_checks),
            compression=compression,
        ),
        "events": h5.create_dataset(
            "events",
            shape=(shots, cycles, code.n_checks),
            dtype="u1",
            chunks=(chunk_rows, cycles, code.n_checks),
            compression=compression,
        ),
        "events_with_final": h5.create_dataset(
            "events_with_final",
            shape=(shots, cycles + 1, code.n_checks),
            dtype="u1",
            chunks=(chunk_rows, cycles + 1, code.n_checks),
            compression=compression,
        ),
        "final_events": h5.create_dataset(
            "final_events",
            shape=(shots, code.n_checks),
            dtype="u1",
            chunks=(chunk_rows, code.n_checks),
            compression=compression,
        ),
        "final_syndrome": h5.create_dataset(
            "final_syndrome",
            shape=(shots, code.n_checks),
            dtype="u1",
            chunks=(chunk_rows, code.n_checks),
            compression=compression,
        ),
        "logical_flips": h5.create_dataset(
            "logical_flips",
            shape=(shots, 2 * code.k),
            dtype="u1",
            chunks=(chunk_rows, 2 * code.k),
            compression=compression,
        ),
    }

    if store_errors:
        datasets["final_x_error"] = h5.create_dataset(
            "final_x_error",
            shape=(shots, code.n_data),
            dtype="u1",
            chunks=(chunk_rows, code.n_data),
            compression=compression,
        )
        datasets["final_z_error"] = h5.create_dataset(
            "final_z_error",
            shape=(shots, code.n_data),
            dtype="u1",
            chunks=(chunk_rows, code.n_data),
            compression=compression,
        )

    if analog_snr is not None:
        datasets["analog_measurements"] = h5.create_dataset(
            "analog_measurements",
            shape=(shots, cycles, code.n_checks),
            dtype="f4",
            chunks=(chunk_rows, cycles, code.n_checks),
            compression=compression,
        )
        datasets["soft_measurements"] = h5.create_dataset(
            "soft_measurements",
            shape=(shots, cycles, code.n_checks),
            dtype="f4",
            chunks=(chunk_rows, cycles, code.n_checks),
            compression=compression,
        )

    return datasets


def _write_result(
    datasets: dict[str, h5py.Dataset],
    result: SimulationResult,
    slc: slice,
    *,
    analog_snr: float | None,
    rng: np.random.Generator,
) -> None:
    datasets["measurements"][slc] = result.measurements
    datasets["events"][slc] = result.events
    datasets["events_with_final"][slc] = result.events_with_final
    datasets["final_events"][slc] = result.final_events
    datasets["final_syndrome"][slc] = result.final_syndrome
    datasets["logical_flips"][slc] = result.logical_flips

    if "final_x_error" in datasets:
        assert result.final_x_error is not None and result.final_z_error is not None
        datasets["final_x_error"][slc] = result.final_x_error
        datasets["final_z_error"][slc] = result.final_z_error

    if analog_snr is not None:
        analog, soft = _analog_channels(result.measurements, snr=analog_snr, rng=rng)
        datasets["analog_measurements"][slc] = analog
        datasets["soft_measurements"][slc] = soft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output HDF5 path.")
    parser.add_argument("--code", default="bb_144", choices=[*sorted(KNOWN_BB_CODES), "custom"])
    parser.add_argument("--ell", type=int, help="Custom BB ell.")
    parser.add_argument("--m", type=int, help="Custom BB m.")
    parser.add_argument("--a", help="Custom A polynomial, e.g. 'x3,y1,y2'.")
    parser.add_argument("--b", help="Custom B polynomial, e.g. 'y3,x1,x2'.")
    parser.add_argument("--noise", choices=["circuit", "phenomenological"], default="circuit")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument(
        "--p",
        type=float,
        default=1e-3,
        help="Circuit-level depolarizing error rate.",
    )
    parser.add_argument("--p-data", type=float, help="Phenomenological data depolarizing rate.")
    parser.add_argument("--p-meas", type=float, help="Phenomenological measurement flip rate.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument(
        "--store-errors",
        action="store_true",
        help="Store final Pauli frames for debugging.",
    )
    parser.add_argument(
        "--analog-snr",
        type=float,
        help="Add simple Gaussian analog/soft measurement channels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots < 1:
        raise ValueError("--shots must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.analog_snr is not None and args.analog_snr <= 0:
        raise ValueError("--analog-snr must be positive")

    code = build_bb_code(args.code, ell=args.ell, m=args.m, a=args.a, b=args.b)
    rng = np.random.default_rng(args.seed)
    compression = _compression(args.compression)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.out, "w") as h5:
        h5.attrs["schema_version"] = "0.1"
        h5.attrs["simulator"] = args.noise
        h5.attrs["shots"] = args.shots
        h5.attrs["cycles"] = args.cycles
        h5.attrs["seed"] = args.seed
        h5.attrs["description"] = (
            "BB-code memory-experiment samples. Check order is all X checks "
            "followed by all Z checks. logical_flips stores X-logical flips "
            "followed by Z-logical flips."
        )
        h5.attrs["p"] = args.p
        h5.attrs["p_data"] = args.p_data if args.p_data is not None else args.p
        h5.attrs["p_meas"] = args.p_meas if args.p_meas is not None else args.p
        if args.analog_snr is not None:
            h5.attrs["analog_snr"] = args.analog_snr

        _write_code_group(h5, code, compression)
        datasets = _create_datasets(
            h5,
            shots=args.shots,
            cycles=args.cycles,
            code=code,
            compression=compression,
            chunks=args.batch_size,
            store_errors=args.store_errors,
            analog_snr=args.analog_snr,
        )

        for start in range(0, args.shots, args.batch_size):
            end = min(start + args.batch_size, args.shots)
            batch_shots = end - start
            if args.noise == "circuit":
                result = simulate_circuit_level(
                    code,
                    shots=batch_shots,
                    cycles=args.cycles,
                    p=args.p,
                    rng=rng,
                    store_errors=args.store_errors,
                )
            else:
                p_data = args.p if args.p_data is None else args.p_data
                p_meas = args.p if args.p_meas is None else args.p_meas
                result = simulate_phenomenological(
                    code,
                    shots=batch_shots,
                    cycles=args.cycles,
                    p_data=p_data,
                    p_meas=p_meas,
                    rng=rng,
                    store_errors=args.store_errors,
                )
            _write_result(
                datasets,
                result,
                slice(start, end),
                analog_snr=args.analog_snr,
                rng=rng,
            )

    print(
        f"Wrote {args.shots} {args.noise} samples for {code.label} "
        f"({args.cycles} cycles, k={code.k}) to {args.out}"
    )


if __name__ == "__main__":
    main()
