import numpy as np

from bbcode_dataset.codes import build_bb_code
from bbcode_dataset.sim import simulate_circuit_level, simulate_phenomenological


def test_bb_72_code_parameters_and_checks():
    code = build_bb_code("bb_72")
    assert code.n_data == 72
    assert code.n_checks == 72
    assert code.k == 12
    assert np.all(code.hx.sum(axis=1) == 6)
    assert np.all(code.hz.sum(axis=1) == 6)
    assert not np.any((code.hx @ code.hz.T) & 1)
    assert np.array_equal((code.x_logicals @ code.z_logicals.T) & 1, np.eye(code.k, dtype=np.uint8))


def test_zero_noise_phenomenological_is_trivial():
    code = build_bb_code("bb_72")
    result = simulate_phenomenological(
        code,
        shots=8,
        cycles=3,
        p_data=0.0,
        p_meas=0.0,
        rng=np.random.default_rng(1),
    )
    assert not result.measurements.any()
    assert not result.events_with_final.any()
    assert not result.logical_flips.any()


def test_zero_noise_circuit_level_is_trivial():
    code = build_bb_code("bb_72")
    result = simulate_circuit_level(
        code,
        shots=8,
        cycles=3,
        p=0.0,
        rng=np.random.default_rng(1),
    )
    assert not result.measurements.any()
    assert not result.events_with_final.any()
    assert not result.logical_flips.any()
