"""PyMatching MWPM baseline decoder."""

from __future__ import annotations

import numpy as np
import pymatching
import stim


def build_matching(circuit: stim.Circuit) -> pymatching.Matching:
    """Build a PyMatching decoder from a Stim detector error model."""
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_batch(matching: pymatching.Matching, events: np.ndarray) -> np.ndarray:
    """Decode detector events and return predicted observable flips."""
    events = np.asarray(events, dtype=np.uint8)
    if hasattr(matching, "decode_batch"):
        pred = matching.decode_batch(events)
    else:
        pred = np.asarray([matching.decode(row) for row in events])
    pred = np.asarray(pred, dtype=np.uint8)
    if pred.ndim == 1:
        pred = pred[:, None]
    return pred


def logical_error_rate(pred: np.ndarray, labels: np.ndarray) -> float:
    """Return sample-wise logical error rate."""
    pred = np.asarray(pred).astype(bool)
    labels = np.asarray(labels).astype(bool)
    if pred.shape != labels.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, labels {labels.shape}")
    return float(np.mean(np.any(pred != labels, axis=1)))
