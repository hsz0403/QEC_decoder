import numpy as np

from qecml.data.coordinate_mapper import events_to_dense, try_events_to_dense
from qecml.sim.stim_surface_code import (
    build_rotated_surface_code_circuit,
    get_detector_coordinates,
    sample_detector_events,
)


def test_manual_coordinate_mapping_dense_shape():
    events = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    dense, mask, mapping = events_to_dense(events, coords)
    assert dense.shape == (2, 2, 2, 1)
    assert mask.shape == (2, 2)
    assert mask.all()
    assert mapping.time_axis == 1


def test_stim_coordinate_mapping_or_clean_fallback():
    circuit = build_rotated_surface_code_circuit(basis="z", distance=3, rounds=3, p=0.001)
    events, _ = sample_detector_events(circuit, shots=4, seed=4)
    coords = get_detector_coordinates(circuit)
    dense, mask, mapping = try_events_to_dense(events, coords)
    if dense is not None:
        assert dense.shape[0] == 4
        assert dense.ndim == 4
        assert mask is not None
        assert mapping is not None
