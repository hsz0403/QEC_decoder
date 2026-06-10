"""Code-capacity surface-code datasets from parity-check matrices.

This module intentionally keeps the data-generation task simple:

1. Build a rotated surface-code memory circuit with Stim and no measurement noise.
2. Extract a binary detector parity-check matrix H and logical-action matrix L.
3. Sample iid binary error-mechanism vectors e ~ Bernoulli(p).
4. Compute syndrome = e H^T mod 2 and logical_action = e L^T mod 2.

The resulting supervised task is syndrome -> logical_action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import stim


def build_code_capacity_circuit(
    *,
    distance: int,
    basis: str = "z",
    topology_error_rate: float = 1e-3,
) -> stim.Circuit:
    """Build a one-round rotated surface-code circuit used only to extract H and L.

    The nonzero data depolarization probability is only used to make Stim emit error
    mechanisms in the detector error model. The actual ML dataset samples iid binary
    errors from the extracted columns using the user-specified p.
    """
    if basis not in {"x", "z"}:
        raise ValueError("basis must be 'x' or 'z'")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=1,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=topology_error_rate,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )


def _target_is_logical(target: stim.DemTarget) -> bool:
    attr = target.is_logical_observable_id
    return bool(attr() if callable(attr) else attr)


def _target_is_detector(target: stim.DemTarget) -> bool:
    attr = target.is_relative_detector_id
    return bool(attr() if callable(attr) else attr)


def detector_error_model_to_matrices(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H, L, and DEM probabilities from a flattened detector error model."""
    flat = dem.flattened()
    h = np.zeros((flat.num_detectors, flat.num_errors), dtype=np.uint8)
    logical = np.zeros((flat.num_observables, flat.num_errors), dtype=np.uint8)
    probabilities = np.zeros(flat.num_errors, dtype=np.float32)

    error_index = 0
    for instruction in flat:
        if instruction.type != "error":
            continue
        probabilities[error_index] = instruction.args_copy()[0]
        for target in instruction.targets_copy():
            if _target_is_detector(target):
                h[target.val, error_index] ^= 1
            elif _target_is_logical(target):
                logical[target.val, error_index] ^= 1
        error_index += 1
    return h, logical, probabilities


def detector_coordinates(circuit: stim.Circuit) -> np.ndarray:
    """Return detector coordinates as a dense float32 array sorted by detector index."""
    coords = circuit.get_detector_coordinates()
    if not coords:
        return np.empty((0, 0), dtype=np.float32)
    max_dim = max(len(v) for v in coords.values())
    arr = np.full((circuit.num_detectors, max_dim), np.nan, dtype=np.float32)
    for detector_idx in range(circuit.num_detectors):
        values = coords.get(detector_idx)
        if values is None:
            return np.empty((0, 0), dtype=np.float32)
        arr[detector_idx, : len(values)] = values
    if np.isnan(arr).any():
        return np.empty((0, 0), dtype=np.float32)
    return arr


def grid_from_detector_coordinates(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Map detector coordinates to integer grid indices for CNN inputs."""
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 2:
        raise ValueError("detector coordinates must have shape [num_detectors, >=2]")
    xy = coords[:, :2]
    xs = np.unique(xy[:, 0])
    ys = np.unique(xy[:, 1])
    x_map = {float(x): i for i, x in enumerate(xs)}
    y_map = {float(y): i for i, y in enumerate(ys)}
    grid = np.zeros((coords.shape[0], 2), dtype=np.int64)
    for i, (x, y) in enumerate(xy):
        grid[i, 0] = y_map[float(y)]
        grid[i, 1] = x_map[float(x)]
    return grid, (len(ys), len(xs))


def logical_mask_for_grid(grid_shape: tuple[int, int], *, basis: str) -> np.ndarray:
    """Create a simple logical-support pooling mask over the detector grid."""
    height, width = grid_shape
    mask = np.zeros((height, width), dtype=np.float32)
    if basis == "z":
        mask[:, width // 2] = 1.0
    elif basis == "x":
        mask[height // 2, :] = 1.0
    else:
        raise ValueError("basis must be 'x' or 'z'")
    return mask


def sample_iid_errors(
    h: np.ndarray,
    logical: np.ndarray,
    *,
    shots: int,
    p: float,
    rng: np.random.Generator,
    batch_size: int = 200_000,
    store_errors: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sample iid errors and compute syndrome/logical labels over GF(2)."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    events = np.empty((shots, h.shape[0]), dtype=np.uint8)
    labels = np.empty((shots, logical.shape[0]), dtype=np.uint8)
    all_errors = np.empty((shots, h.shape[1]), dtype=np.uint8) if store_errors else None
    h_t = h.T.astype(np.uint8)
    logical_t = logical.T.astype(np.uint8)
    for start in range(0, shots, batch_size):
        end = min(start + batch_size, shots)
        errors = (rng.random((end - start, h.shape[1])) < p).astype(np.uint8)
        events[start:end] = (errors @ h_t) & 1
        labels[start:end] = (errors @ logical_t) & 1
        if all_errors is not None:
            all_errors[start:end] = errors
    return events, labels, all_errors


def build_surface_code_iid_dataset(
    *,
    distance: int,
    p: float,
    shots: int,
    basis: str = "z",
    seed: int = 0,
    store_errors: bool = False,
) -> dict[str, Any]:
    """Generate a code-capacity surface-code iid-error dataset."""
    circuit = build_code_capacity_circuit(distance=distance, basis=basis)
    dem = circuit.detector_error_model(decompose_errors=True)
    h, logical, dem_probabilities = detector_error_model_to_matrices(dem)
    coords = detector_coordinates(circuit)
    grid, grid_shape = grid_from_detector_coordinates(coords)
    rng = np.random.default_rng(seed)
    events, labels, errors = sample_iid_errors(
        h,
        logical,
        shots=shots,
        p=p,
        rng=rng,
        store_errors=store_errors,
    )
    metadata = {
        "source": "surface_code_capacity_iid",
        "basis": basis,
        "distance": distance,
        "p": p,
        "shots": shots,
        "seed": seed,
        "num_detectors": int(h.shape[0]),
        "num_error_mechanisms": int(h.shape[1]),
        "num_observables": int(logical.shape[0]),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "generation_rule": "e~Bernoulli(p); syndrome=e@H.T mod2; label=e@L.T mod2",
        "note": (
            "No measurement-error sampling is used. Stim is used only to extract the "
            "rotated surface-code detector parity-check matrix and logical action."
        ),
    }
    result: dict[str, Any] = {
        "events": events,
        "labels": labels,
        "parity_check": h,
        "logical_matrix": logical,
        "dem_probabilities": dem_probabilities,
        "detector_coords": coords,
        "detector_grid": grid,
        "logical_mask": logical_mask_for_grid(grid_shape, basis=basis),
        "metadata": metadata,
    }
    if errors is not None:
        result["errors"] = errors
    return result


def save_surface_code_dataset(path: str | Path, dataset: dict[str, Any]) -> None:
    """Save a generated dataset to compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "events": dataset["events"].astype(np.uint8),
        "labels": dataset["labels"].astype(np.uint8),
        "parity_check": dataset["parity_check"].astype(np.uint8),
        "logical_matrix": dataset["logical_matrix"].astype(np.uint8),
        "dem_probabilities": dataset["dem_probabilities"].astype(np.float32),
        "detector_coords": dataset["detector_coords"].astype(np.float32),
        "detector_grid": dataset["detector_grid"].astype(np.int64),
        "logical_mask": dataset["logical_mask"].astype(np.float32),
        "metadata_json": json.dumps(dataset["metadata"], sort_keys=True),
    }
    if "errors" in dataset:
        arrays["errors"] = dataset["errors"].astype(np.uint8)
    np.savez_compressed(path, **arrays)


def load_surface_code_dataset(path: str | Path) -> dict[str, Any]:
    """Load a compressed NPZ surface-code dataset."""
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"]
        metadata_text = metadata_raw.item() if metadata_raw.shape == () else metadata_raw
        result: dict[str, Any] = {
            "events": data["events"].astype(np.uint8),
            "labels": data["labels"].astype(np.uint8),
            "parity_check": data["parity_check"].astype(np.uint8),
            "logical_matrix": data["logical_matrix"].astype(np.uint8),
            "dem_probabilities": data["dem_probabilities"].astype(np.float32),
            "detector_coords": data["detector_coords"].astype(np.float32),
            "detector_grid": data["detector_grid"].astype(np.int64),
            "logical_mask": data["logical_mask"].astype(np.float32),
            "metadata": json.loads(str(metadata_text)),
        }
        if "errors" in data:
            result["errors"] = data["errors"].astype(np.uint8)
        return result
