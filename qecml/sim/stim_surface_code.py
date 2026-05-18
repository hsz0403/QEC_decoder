"""Stim rotated surface-code circuit generation and sampling."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import stim


def build_rotated_surface_code_circuit(
    *,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    noise_model: str = "circuit_depolarizing",
    seed: Optional[int] = None,
) -> stim.Circuit:
    """Build a noisy rotated surface-code memory circuit with Stim.

    The ``seed`` argument is accepted for API symmetry with samplers. Stim's
    generated circuit itself is deterministic for these inputs.
    """
    del seed
    if basis not in {"x", "z"}:
        raise ValueError("basis must be 'x' or 'z'")
    if distance < 2:
        raise ValueError("distance must be at least 2")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if noise_model != "circuit_depolarizing":
        raise ValueError("only noise_model='circuit_depolarizing' is currently supported")

    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def sample_detector_events(
    circuit: stim.Circuit,
    shots: int,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector events and logical observable flips.

    Returns:
        ``events`` with shape ``[shots, n_detectors]`` and ``labels`` with
        shape ``[shots, n_observables]``.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    sampler = circuit.compile_detector_sampler(seed=seed)
    events, obs = sampler.sample(shots=shots, separate_observables=True)
    return events.astype(np.uint8), obs.astype(np.uint8)


def get_detector_coordinates(circuit: stim.Circuit) -> np.ndarray:
    """Return detector coordinates sorted by detector index, or an empty array."""
    coords: dict[int, list[float]] = circuit.get_detector_coordinates()
    if not coords:
        return np.empty((0, 0), dtype=np.float32)

    n = circuit.num_detectors
    max_dim = max((len(v) for v in coords.values()), default=0)
    if len(coords) != n or max_dim == 0:
        return np.empty((0, 0), dtype=np.float32)

    arr = np.full((n, max_dim), np.nan, dtype=np.float32)
    for det_idx in range(n):
        values = coords.get(det_idx)
        if values is None:
            return np.empty((0, 0), dtype=np.float32)
        arr[det_idx, : len(values)] = values
    if np.isnan(arr).any():
        return np.empty((0, 0), dtype=np.float32)
    return arr


def circuit_metadata(
    *,
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    noise_model: str = "circuit_depolarizing",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create standard metadata for a generated Stim dataset."""
    metadata: dict[str, Any] = {
        "source": "stim",
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "noise_model": noise_model,
    }
    if extra:
        metadata.update(extra)
    return metadata
