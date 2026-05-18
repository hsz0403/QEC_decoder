"""Map flat detector events into a dense space-time tensor."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CoordinateMapping:
    """Mapping from detector index to dense ``[time, site]`` coordinates."""

    time_values: np.ndarray
    site_keys: list[tuple[float, ...]]
    detector_to_time_site: dict[int, tuple[int, int]]
    valid_mask: np.ndarray
    time_axis: int


def build_coordinate_mapping(detector_coords: np.ndarray) -> CoordinateMapping:
    """Build a robust dense coordinate mapping from Stim detector coordinates.

    The axis with the largest number of repeated groups is selected as time,
    with a mild preference for the last coordinate axis used by Stim surface-code
    generators. If multiple detectors land on one ``(time, site)`` slot, the
    mapping is considered ambiguous and a ``ValueError`` is raised.
    """
    coords = np.asarray(detector_coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] < 2:
        raise ValueError("detector_coords must have shape [N, C] with C >= 2")
    if not np.isfinite(coords).all():
        raise ValueError("detector_coords contains non-finite values")

    axis_scores: list[tuple[int, int]] = []
    for axis in range(coords.shape[1]):
        axis_scores.append((len(np.unique(coords[:, axis])), axis))
    max_unique = max(score for score, _ in axis_scores)
    candidate_axes = [axis for score, axis in axis_scores if score == max_unique]
    time_axis = coords.shape[1] - 1 if coords.shape[1] - 1 in candidate_axes else candidate_axes[0]

    time_values = np.array(sorted(np.unique(coords[:, time_axis])), dtype=np.float32)
    time_index = {float(v): i for i, v in enumerate(time_values)}

    spatial_axes = [i for i in range(coords.shape[1]) if i != time_axis]
    site_keys = sorted({tuple(float(x) for x in row[spatial_axes]) for row in coords})
    site_index = {key: i for i, key in enumerate(site_keys)}

    valid_mask = np.zeros((len(time_values), len(site_keys)), dtype=bool)
    detector_to_time_site: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    for det_idx, row in enumerate(coords):
        t = time_index[float(row[time_axis])]
        site_key = tuple(float(x) for x in row[spatial_axes])
        s = site_index[site_key]
        slot = (t, s)
        if slot in occupied:
            raise ValueError("multiple detectors map to the same dense time/site slot")
        occupied.add(slot)
        detector_to_time_site[det_idx] = slot
        valid_mask[t, s] = True

    return CoordinateMapping(
        time_values=time_values,
        site_keys=site_keys,
        detector_to_time_site=detector_to_time_site,
        valid_mask=valid_mask,
        time_axis=time_axis,
    )


def events_to_dense(
    events: np.ndarray,
    detector_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, CoordinateMapping]:
    """Convert flat events ``[B, N]`` to dense events ``[B, T, S, 1]``."""
    flat = np.asarray(events)
    if flat.ndim != 2:
        raise ValueError("events must have shape [B, N]")
    mapping = build_coordinate_mapping(detector_coords)
    if flat.shape[1] != len(mapping.detector_to_time_site):
        raise ValueError("events detector dimension does not match coordinate count")

    dense = np.zeros(
        (flat.shape[0], len(mapping.time_values), len(mapping.site_keys), 1),
        dtype=flat.dtype,
    )
    for det_idx, (t, s) in mapping.detector_to_time_site.items():
        dense[:, t, s, 0] = flat[:, det_idx]
    return dense, mapping.valid_mask, mapping


def try_events_to_dense(
    events: np.ndarray,
    detector_coords: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[CoordinateMapping]]:
    """Try dense conversion and warn instead of failing when coordinates are ambiguous."""
    if detector_coords is None or detector_coords.size == 0:
        warnings.warn("detector coordinates unavailable; using flat representation", stacklevel=2)
        return None, None, None
    try:
        return events_to_dense(events, detector_coords)
    except ValueError as exc:
        warnings.warn(f"coordinate mapping failed: {exc}; using flat representation", stacklevel=2)
        return None, None, None
