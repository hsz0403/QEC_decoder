from qecml.data.stim_dataset import load_npz_dataset, save_npz_dataset
from qecml.sim.stim_surface_code import (
    build_rotated_surface_code_circuit,
    circuit_metadata,
    get_detector_coordinates,
    sample_detector_events,
)


def test_stim_generation_shapes_and_metadata(tmp_path):
    circuit = build_rotated_surface_code_circuit(basis="z", distance=3, rounds=3, p=0.001)
    events, labels = sample_detector_events(circuit, shots=16, seed=1)
    coords = get_detector_coordinates(circuit)
    assert circuit.num_detectors > 0
    assert events.shape == (16, circuit.num_detectors)
    assert labels.shape == (16, circuit.num_observables)
    metadata = circuit_metadata(basis="z", distance=3, rounds=3, p=0.001)
    assert metadata["distance"] == 3
    assert metadata["rounds"] == 3
    assert metadata["basis"] == "z"
    assert metadata["p"] == 0.001

    path = tmp_path / "data.npz"
    save_npz_dataset(path, events=events, labels=labels, detector_coords=coords, metadata=metadata)
    loaded = load_npz_dataset(path)
    assert loaded["events"].shape == events.shape
    assert loaded["labels"].shape == labels.shape
    assert loaded["metadata"]["distance"] == 3
