import numpy as np

from qecml.decoders.pymatching_decoder import build_matching, decode_batch, logical_error_rate
from qecml.sim.stim_surface_code import build_rotated_surface_code_circuit, sample_detector_events


def test_zero_noise_ler_is_zero():
    circuit = build_rotated_surface_code_circuit(basis="z", distance=3, rounds=3, p=0.0)
    events, labels = sample_detector_events(circuit, shots=32, seed=2)
    pred = decode_batch(build_matching(circuit), events)
    assert pred.shape == labels.shape
    assert logical_error_rate(pred, labels) == 0.0


def test_nonzero_noise_ler_is_finite():
    circuit = build_rotated_surface_code_circuit(basis="z", distance=3, rounds=3, p=0.01)
    events, labels = sample_detector_events(circuit, shots=64, seed=3)
    pred = decode_batch(build_matching(circuit), events)
    ler = logical_error_rate(pred, labels)
    assert pred.shape == labels.shape
    assert np.isfinite(ler)
    assert 0.0 <= ler <= 1.0
