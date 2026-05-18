"""Optional helpers for inspecting Zenodo record 6804040.

The first milestone does not assume a fixed real-data format. Conversion to the
common NPZ schema should be implemented after inspecting the downloaded README
and file tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile

import numpy as np
import stim
import yaml

from qecml.data.stim_dataset import save_npz_dataset
from qecml.sim.stim_surface_code import get_detector_coordinates


def list_zip_tree(zip_path: Union[str, Path]) -> list[str]:
    """Return file names contained in a Zenodo ZIP archive."""
    with ZipFile(zip_path) as zf:
        return zf.namelist()


def read_readmes_from_zip(zip_path: Union[str, Path], max_chars: int = 20_000) -> dict[str, str]:
    """Read README-like files from a ZIP archive."""
    result: dict[str, str] = {}
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if "readme" not in lower:
                continue
            with zf.open(name) as f:
                result[name] = f.read(max_chars).decode("utf-8", errors="replace")
    return result


def experiment_names(zip_path: Union[str, Path], *, code_prefix: str = "surface_code") -> list[str]:
    """Return experiment directory names from the Zenodo ZIP."""
    with ZipFile(zip_path) as zf:
        names = {
            name.split("/")[0]
            for name in zf.namelist()
            if "/" in name and name.split("/")[0].startswith(code_prefix)
        }
    return sorted(names)


def parse_experiment_name(name: str) -> dict[str, Any]:
    """Parse names like ``surface_code_bX_d3_r01_center_3_5``."""
    pattern = re.compile(
        r"^(?P<code>.+)_b(?P<basis>[XZ])_d(?P<distance>\d+)_"
        r"r(?P<rounds>\d+)_center_(?P<center_row>\d+)_(?P<center_col>\d+)$"
    )
    match = pattern.match(name)
    if not match:
        return {"experiment": name}
    result: dict[str, Any] = match.groupdict()
    for key in ["distance", "rounds", "center_row", "center_col"]:
        result[key] = int(result[key])
    return result


def unpack_b8(data: bytes, *, shots: int, bits_per_shot: int) -> np.ndarray:
    """Unpack Stim b8 result-format data into ``[shots, bits_per_shot]``."""
    row_bytes = (bits_per_shot + 7) // 8
    expected = shots * row_bytes
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes but found {len(data)}")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(shots, row_bytes)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot].astype(np.uint8)


def read_01(data: bytes, *, shots: int, observables: int = 1) -> np.ndarray:
    """Read Stim 01 result-format observable data."""
    lines = [line.strip() for line in data.decode("ascii").splitlines() if line.strip()]
    if len(lines) != shots:
        raise ValueError(f"expected {shots} lines but found {len(lines)}")
    arr = np.array([[int(ch) for ch in line] for line in lines], dtype=np.uint8)
    if arr.shape != (shots, observables):
        raise ValueError(f"expected shape {(shots, observables)} but found {arr.shape}")
    return arr


def load_experiment_from_zip(
    zip_path: Union[str, Path],
    experiment: str,
    *,
    skip_shots: int = 0,
    max_shots: int | None = None,
) -> dict[str, Any]:
    """Load one Zenodo 6804040 experiment into the common in-memory schema."""
    with ZipFile(zip_path) as zf:
        prefix = experiment.rstrip("/") + "/"
        props = yaml.safe_load(zf.read(prefix + "properties.yml"))
        total_shots = int(props["shots"])
        if skip_shots < 0 or skip_shots > total_shots:
            raise ValueError("skip_shots must be in [0, total_shots]")
        shots = total_shots - skip_shots
        if max_shots is not None:
            shots = min(shots, int(max_shots))
        n_detectors = int(props["circuit_detectors"])
        n_observables = int(props["circuit_observables"])

        row_bytes = (n_detectors + 7) // 8
        det_start = skip_shots * row_bytes
        det_stop = det_start + shots * row_bytes
        det_bytes = zf.read(prefix + "detection_events.b8")[det_start:det_stop]
        events = unpack_b8(det_bytes, shots=shots, bits_per_shot=n_detectors)

        all_label_lines = zf.read(prefix + "obs_flips_actual.01").splitlines()
        raw_label_lines = all_label_lines[skip_shots : skip_shots + shots]
        label_lines = b"\n".join(raw_label_lines) + b"\n"
        labels = read_01(label_lines, shots=shots, observables=n_observables)

        circuit = stim.Circuit(zf.read(prefix + "circuit_ideal.stim").decode("utf-8"))
        detector_coords = get_detector_coordinates(circuit)

    metadata = {
        "source": "zenodo_6804040",
        "experiment": experiment,
        "properties": props,
        "skip_shots": skip_shots,
        "shots": shots,
        **parse_experiment_name(experiment),
    }
    return {
        "events": events,
        "labels": labels,
        "detector_coords": detector_coords,
        "metadata": metadata,
    }


def convert_experiment_to_npz(
    zip_path: Union[str, Path],
    experiment: str,
    output_path: Union[str, Path],
    *,
    skip_shots: int = 0,
    max_shots: int | None = None,
) -> dict[str, Any]:
    """Convert one Zenodo 6804040 experiment to the repo's common NPZ schema."""
    dataset = load_experiment_from_zip(
        zip_path,
        experiment,
        skip_shots=skip_shots,
        max_shots=max_shots,
    )
    save_npz_dataset(
        output_path,
        events=dataset["events"],
        labels=dataset["labels"],
        detector_coords=dataset["detector_coords"],
        metadata=dataset["metadata"],
    )
    return {
        "output": str(output_path),
        "events_shape": list(dataset["events"].shape),
        "labels_shape": list(dataset["labels"].shape),
        "label_positive_rate": float(np.mean(dataset["labels"])),
        "metadata_json": json.dumps(dataset["metadata"], sort_keys=True),
    }
