"""Storage helpers for Stim-generated QEC datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def save_npz_dataset(
    path: Union[str, Path],
    *,
    events: np.ndarray,
    labels: np.ndarray,
    detector_coords: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save detector events, labels, coordinates, and metadata to a common NPZ schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "events": events.astype(np.uint8),
        "labels": labels.astype(np.uint8),
        "metadata_json": json.dumps(metadata or {}, sort_keys=True),
    }
    if detector_coords is not None and detector_coords.size > 0:
        arrays["detector_coords"] = detector_coords.astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_npz_dataset(path: Union[str, Path]) -> dict[str, Any]:
    """Load a dataset saved with :func:`save_npz_dataset`."""
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"]
        metadata_text = metadata_raw.item() if metadata_raw.shape == () else metadata_raw
        metadata = json.loads(str(metadata_text))
        detector_coords = data["detector_coords"] if "detector_coords" in data else None
        return {
            "events": data["events"].astype(np.uint8),
            "labels": data["labels"].astype(np.uint8),
            "detector_coords": (
                None if detector_coords is None else detector_coords.astype(np.float32)
            ),
            "metadata": metadata,
        }
