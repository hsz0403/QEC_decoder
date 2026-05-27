"""Small GF(2) linear-algebra helpers used to construct CSS logical bases."""

from __future__ import annotations

import numpy as np


def as_binary(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.uint8) & 1


def matmul_mod2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (as_binary(left) @ as_binary(right)) & 1


def row_reduce(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return reduced row-echelon form over GF(2) and pivot columns."""
    a = as_binary(matrix).copy()
    if a.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = a.shape
    pivot_cols: list[int] = []
    r = 0

    for c in range(cols):
        candidates = np.flatnonzero(a[r:, c])
        if candidates.size == 0:
            continue

        p = r + int(candidates[0])
        if p != r:
            a[[r, p]] = a[[p, r]]

        other_rows = np.flatnonzero(a[:, c])
        other_rows = other_rows[other_rows != r]
        if other_rows.size:
            a[other_rows] ^= a[r]

        pivot_cols.append(c)
        r += 1
        if r == rows:
            break

    return a[:r], pivot_cols


def row_space_basis(matrix: np.ndarray) -> np.ndarray:
    rref, _ = row_reduce(matrix)
    if rref.size == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    nonzero = np.any(rref, axis=1)
    return rref[nonzero].astype(np.uint8, copy=False)


def rank(matrix: np.ndarray) -> int:
    _, pivots = row_reduce(matrix)
    return len(pivots)


def nullspace(matrix: np.ndarray) -> np.ndarray:
    """Return a row basis for ker(matrix) over GF(2)."""
    a = as_binary(matrix)
    if a.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = a.shape
    if cols == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if rows == 0:
        return np.eye(cols, dtype=np.uint8)

    rref, pivots = row_reduce(a)
    pivot_set = set(pivots)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    basis = np.zeros((len(free_cols), cols), dtype=np.uint8)

    for row, free_col in enumerate(free_cols):
        basis[row, free_col] = 1
        for pivot_row, pivot_col in enumerate(pivots):
            basis[row, pivot_col] = rref[pivot_row, free_col]

    return basis


def inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a square binary matrix over GF(2)."""
    a = as_binary(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")

    n = a.shape[0]
    aug = np.concatenate([a.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    r = 0
    for c in range(n):
        candidates = np.flatnonzero(aug[r:, c])
        if candidates.size == 0:
            raise ValueError("matrix is singular over GF(2)")
        p = r + int(candidates[0])
        if p != r:
            aug[[r, p]] = aug[[p, r]]

        other_rows = np.flatnonzero(aug[:, c])
        other_rows = other_rows[other_rows != r]
        if other_rows.size:
            aug[other_rows] ^= aug[r]
        r += 1

    return aug[:, n:].astype(np.uint8, copy=False)


def quotient_basis(super_basis: np.ndarray, sub_basis: np.ndarray) -> np.ndarray:
    """Select vectors that extend subspace(sub_basis) inside subspace(super_basis)."""
    super_basis = as_binary(super_basis)
    sub_basis = as_binary(sub_basis)
    if super_basis.ndim != 2 or sub_basis.ndim != 2:
        raise ValueError("bases must be two-dimensional")
    if super_basis.shape[1] != sub_basis.shape[1]:
        raise ValueError("basis widths differ")

    current = row_space_basis(sub_basis)
    current_rank = current.shape[0]
    selected: list[np.ndarray] = []

    for vector in super_basis:
        candidate = np.vstack([current, vector])
        candidate_rank = rank(candidate)
        if candidate_rank > current_rank:
            selected.append(vector.copy())
            current = row_space_basis(candidate)
            current_rank = candidate_rank

    if not selected:
        return np.zeros((0, super_basis.shape[1]), dtype=np.uint8)
    return np.asarray(selected, dtype=np.uint8)


def css_logical_bases(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired X- and Z-logical bases for a CSS code.

    Rows of ``x_logicals`` are in ker(HZ) modulo row(HX). Rows of ``z_logicals``
    are transformed so that ``x_logicals @ z_logicals.T == I`` over GF(2).
    """
    hx = as_binary(hx)
    hz = as_binary(hz)
    if hx.shape[1] != hz.shape[1]:
        raise ValueError("HX and HZ must act on the same number of data qubits")

    x_logicals = quotient_basis(nullspace(hz), row_space_basis(hx))
    z_logicals = quotient_basis(nullspace(hx), row_space_basis(hz))
    if x_logicals.shape[0] != z_logicals.shape[0]:
        raise ValueError("X and Z logical dimensions differ")

    k = x_logicals.shape[0]
    if k == 0:
        return x_logicals, z_logicals

    pairing = matmul_mod2(x_logicals, z_logicals.T)
    pairing_inv = inverse(pairing)
    z_dual = matmul_mod2(pairing_inv.T, z_logicals)

    check = matmul_mod2(x_logicals, z_dual.T)
    if not np.array_equal(check, np.eye(k, dtype=np.uint8)):
        raise ValueError("failed to build paired logical bases")

    return x_logicals.astype(np.uint8), z_dual.astype(np.uint8)
