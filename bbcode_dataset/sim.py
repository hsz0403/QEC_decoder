"""Simulation backends for BB-code neural-decoder training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codes import BBCode
from .gf2 import matmul_mod2


@dataclass
class SimulationResult:
    measurements: np.ndarray
    events: np.ndarray
    final_events: np.ndarray
    events_with_final: np.ndarray
    final_syndrome: np.ndarray
    logical_flips: np.ndarray
    final_x_error: np.ndarray | None = None
    final_z_error: np.ndarray | None = None


def _empty_errors(batch_size: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((batch_size, n), dtype=np.uint8),
        np.zeros((batch_size, n), dtype=np.uint8),
    )


def _sample_depolarizing(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    faulty = rng.random(shape) < p
    pauli = rng.integers(0, 3, size=shape, dtype=np.uint8)
    x = faulty & (pauli != 2)  # X or Y
    z = faulty & (pauli != 0)  # Y or Z
    return x.astype(np.uint8), z.astype(np.uint8)


def _xor_subset(target: np.ndarray, columns: np.ndarray, values: np.ndarray) -> None:
    view = target[:, columns]
    view ^= values.astype(np.uint8)
    target[:, columns] = view


def _parity_from_supports(errors: np.ndarray, supports: np.ndarray) -> np.ndarray:
    gathered = errors[:, supports]
    return np.bitwise_xor.reduce(gathered, axis=2).astype(np.uint8)


def _current_syndrome(code: BBCode, x_error: np.ndarray, z_error: np.ndarray) -> np.ndarray:
    x_check_syndrome = _parity_from_supports(z_error, code.hx_supports)
    z_check_syndrome = _parity_from_supports(x_error, code.hz_supports)
    return np.concatenate([x_check_syndrome, z_check_syndrome], axis=1)


def _logical_flips(code: BBCode, x_error: np.ndarray, z_error: np.ndarray) -> np.ndarray:
    x_logical_flips = matmul_mod2(z_error, code.x_logicals.T)
    z_logical_flips = matmul_mod2(x_error, code.z_logicals.T)
    return np.concatenate([x_logical_flips, z_logical_flips], axis=1).astype(np.uint8)


def _events_from_measurements(
    measurements: np.ndarray,
    final_syndrome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events = measurements.copy()
    if measurements.shape[1] > 1:
        events[:, 1:, :] ^= measurements[:, :-1, :]
    final_events = final_syndrome ^ measurements[:, -1, :]
    events_with_final = np.concatenate([events, final_events[:, None, :]], axis=1)
    return events, final_events, events_with_final


def simulate_phenomenological(
    code: BBCode,
    *,
    shots: int,
    cycles: int,
    p_data: float,
    p_meas: float,
    rng: np.random.Generator,
    store_errors: bool = False,
) -> SimulationResult:
    """Simulate repeated noisy syndrome measurements with data and readout noise."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    x_error, z_error = _empty_errors(shots, code.n_data)
    measurements = np.zeros((shots, cycles, code.n_checks), dtype=np.uint8)

    for cycle in range(cycles):
        dx, dz = _sample_depolarizing(rng, x_error.shape, p_data)
        x_error ^= dx
        z_error ^= dz

        syndrome = _current_syndrome(code, x_error, z_error)
        measurement_flips = (rng.random(syndrome.shape) < p_meas).astype(np.uint8)
        measurements[:, cycle, :] = syndrome ^ measurement_flips

    final_syndrome = _current_syndrome(code, x_error, z_error)
    events, final_events, events_with_final = _events_from_measurements(
        measurements,
        final_syndrome,
    )
    logical_flips = _logical_flips(code, x_error, z_error)

    return SimulationResult(
        measurements=measurements,
        events=events,
        final_events=final_events,
        events_with_final=events_with_final,
        final_syndrome=final_syndrome,
        logical_flips=logical_flips,
        final_x_error=x_error.copy() if store_errors else None,
        final_z_error=z_error.copy() if store_errors else None,
    )


class _CircuitFrame:
    def __init__(self, code: BBCode, shots: int, p: float, rng: np.random.Generator):
        self.code = code
        self.shots = shots
        self.p = p
        self.rng = rng
        self.lm = code.lm
        self.qx = np.arange(0, self.lm, dtype=np.int64)
        self.ql = np.arange(self.lm, 2 * self.lm, dtype=np.int64)
        self.qr = np.arange(2 * self.lm, 3 * self.lm, dtype=np.int64)
        self.qz = np.arange(3 * self.lm, 4 * self.lm, dtype=np.int64)
        self.x, self.z = _empty_errors(shots, 4 * self.lm)

    def init_x(self, qubits: np.ndarray) -> None:
        self.x[:, qubits] = 0
        self.z[:, qubits] = 0
        faults = (self.rng.random((self.shots, qubits.size)) < self.p).astype(np.uint8)
        self.z[:, qubits] = faults

    def init_z(self, qubits: np.ndarray) -> None:
        self.x[:, qubits] = 0
        self.z[:, qubits] = 0
        faults = (self.rng.random((self.shots, qubits.size)) < self.p).astype(np.uint8)
        self.x[:, qubits] = faults

    def idle(self, qubits: np.ndarray) -> None:
        dx, dz = _sample_depolarizing(self.rng, (self.shots, qubits.size), self.p)
        _xor_subset(self.x, qubits, dx)
        _xor_subset(self.z, qubits, dz)

    def cnot(self, controls: np.ndarray, targets: np.ndarray) -> None:
        target_x = self.x[:, targets]
        target_x ^= self.x[:, controls]
        self.x[:, targets] = target_x

        control_z = self.z[:, controls]
        control_z ^= self.z[:, targets]
        self.z[:, controls] = control_z

        faulty = self.rng.random((self.shots, controls.size)) < self.p
        pauli = self.rng.integers(1, 16, size=(self.shots, controls.size), dtype=np.uint8)
        control_pauli = pauli // 4
        target_pauli = pauli % 4

        control_x = faulty & ((control_pauli == 1) | (control_pauli == 2))
        control_z = faulty & ((control_pauli == 2) | (control_pauli == 3))
        target_x = faulty & ((target_pauli == 1) | (target_pauli == 2))
        target_z = faulty & ((target_pauli == 2) | (target_pauli == 3))

        _xor_subset(self.x, controls, control_x)
        _xor_subset(self.z, controls, control_z)
        _xor_subset(self.x, targets, target_x)
        _xor_subset(self.z, targets, target_z)

    def measure_z(self, qubits: np.ndarray) -> np.ndarray:
        outcome = self.x[:, qubits].copy()
        flips = self.rng.random(outcome.shape) < self.p
        outcome ^= flips.astype(np.uint8)
        return outcome

    def measure_x(self, qubits: np.ndarray) -> np.ndarray:
        outcome = self.z[:, qubits].copy()
        flips = self.rng.random(outcome.shape) < self.p
        outcome ^= flips.astype(np.uint8)
        return outcome

    def data_errors(self) -> tuple[np.ndarray, np.ndarray]:
        x_data = np.concatenate([self.x[:, self.ql], self.x[:, self.qr]], axis=1)
        z_data = np.concatenate([self.z[:, self.ql], self.z[:, self.qr]], axis=1)
        return x_data, z_data

    def _l(self, columns: np.ndarray) -> np.ndarray:
        return self.ql[columns]

    def _r(self, columns: np.ndarray) -> np.ndarray:
        return self.qr[columns]

    def run_cycle(self) -> np.ndarray:
        c = self.code
        i = np.arange(self.lm, dtype=np.int64)

        # Round 1
        self.init_x(self.qx)
        self.cnot(self._r(c.a_t_maps[0]), self.qz[i])
        self.idle(self.ql)

        # Round 2
        self.cnot(self.qx[i], self._l(c.a_maps[1]))
        self.cnot(self._r(c.a_t_maps[2]), self.qz[i])

        # Round 3
        self.cnot(self.qx[i], self._r(c.b_maps[1]))
        self.cnot(self._l(c.b_t_maps[0]), self.qz[i])

        # Round 4
        self.cnot(self.qx[i], self._r(c.b_maps[0]))
        self.cnot(self._l(c.b_t_maps[1]), self.qz[i])

        # Round 5
        self.cnot(self.qx[i], self._r(c.b_maps[2]))
        self.cnot(self._l(c.b_t_maps[2]), self.qz[i])

        # Round 6
        self.cnot(self.qx[i], self._l(c.a_maps[0]))
        self.cnot(self._r(c.a_t_maps[1]), self.qz[i])

        # Round 7
        self.cnot(self.qx[i], self._l(c.a_maps[2]))
        z_check_measurements = self.measure_z(self.qz)
        self.idle(self.qr)

        # Round 8
        x_check_measurements = self.measure_x(self.qx)
        self.init_z(self.qz)
        self.idle(self.ql)
        self.idle(self.qr)

        return np.concatenate([x_check_measurements, z_check_measurements], axis=1)


def simulate_circuit_level(
    code: BBCode,
    *,
    shots: int,
    cycles: int,
    p: float,
    rng: np.random.Generator,
    store_errors: bool = False,
) -> SimulationResult:
    """Simulate the 2308.07915 BB syndrome cycle with standard depolarizing faults."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    frame = _CircuitFrame(code, shots, p, rng)
    frame.init_z(frame.qz)

    measurements = np.zeros((shots, cycles, code.n_checks), dtype=np.uint8)
    for cycle in range(cycles):
        measurements[:, cycle, :] = frame.run_cycle()

    x_error, z_error = frame.data_errors()
    final_syndrome = _current_syndrome(code, x_error, z_error)
    events, final_events, events_with_final = _events_from_measurements(
        measurements,
        final_syndrome,
    )
    logical_flips = _logical_flips(code, x_error, z_error)

    return SimulationResult(
        measurements=measurements,
        events=events,
        final_events=final_events,
        events_with_final=events_with_final,
        final_syndrome=final_syndrome,
        logical_flips=logical_flips,
        final_x_error=x_error.copy() if store_errors else None,
        final_z_error=z_error.copy() if store_errors else None,
    )
