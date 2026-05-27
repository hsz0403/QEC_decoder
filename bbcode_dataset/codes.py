"""Bivariate bicycle code construction from Bravyi et al. arXiv:2308.07915."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .gf2 import css_logical_bases, matmul_mod2

Term = tuple[str, int]


KNOWN_BB_CODES: dict[str, dict[str, object]] = {
    "bb_72": {
        "label": "[[72,12,6]]",
        "ell": 6,
        "m": 6,
        "a": "x3,y1,y2",
        "b": "y3,x1,x2",
    },
    "bb_90": {
        "label": "[[90,8,10]]",
        "ell": 15,
        "m": 3,
        "a": "x9,y1,y2",
        "b": "1,x2,x7",
    },
    "bb_108": {
        "label": "[[108,8,10]]",
        "ell": 9,
        "m": 6,
        "a": "x3,y1,y2",
        "b": "y3,x1,x2",
    },
    "bb_144": {
        "label": "[[144,12,12]]",
        "ell": 12,
        "m": 6,
        "a": "x3,y1,y2",
        "b": "y3,x1,x2",
    },
    "bb_288": {
        "label": "[[288,12,18]]",
        "ell": 12,
        "m": 12,
        "a": "x3,y2,y7",
        "b": "y3,x1,x2",
    },
    "bb_360": {
        "label": "[[360,12,<=24]]",
        "ell": 30,
        "m": 6,
        "a": "x9,y1,y2",
        "b": "y3,x25,x26",
    },
    "bb_756": {
        "label": "[[756,16,<=34]]",
        "ell": 21,
        "m": 18,
        "a": "x3,y10,y17",
        "b": "y5,x3,x19",
    },
}


def parse_polynomial(text: str | Iterable[str | Term]) -> list[Term]:
    """Parse terms like ``x3,y1,y2`` or ``x^3 + y + y^2``."""
    if not isinstance(text, str):
        terms: list[Term] = []
        for term in text:
            if isinstance(term, tuple):
                axis, power = term
                terms.append((axis.lower(), int(power)))
            else:
                terms.extend(parse_polynomial(term))
        return terms

    cleaned = text.replace(" ", "").replace("+", ",")
    terms = []
    for raw in cleaned.split(","):
        if not raw:
            continue
        token = raw.lower()
        if token in {"1", "i", "id", "identity"}:
            terms.append(("i", 0))
            continue
        axis = token[0]
        if axis not in {"x", "y"}:
            raise ValueError(f"unknown polynomial term: {raw!r}")
        power_text = token[1:]
        if power_text.startswith("^"):
            power_text = power_text[1:]
        power = 1 if power_text == "" else int(power_text)
        terms.append((axis, power))
    return terms


def _term_columns(ell: int, m: int, term: Term, *, transpose: bool = False) -> np.ndarray:
    """Column index of the nonzero entry for each row of a permutation term."""
    axis, power = term
    if transpose:
        power = -power
    lm = ell * m
    rows = np.arange(lm, dtype=np.int64)
    x_coord = rows // m
    y_coord = rows % m

    if axis == "i":
        pass
    elif axis == "x":
        x_coord = (x_coord + power) % ell
    elif axis == "y":
        y_coord = (y_coord + power) % m
    else:
        raise ValueError(f"unknown term axis {axis!r}")

    return (x_coord * m + y_coord).astype(np.int64)


def _supports_from_terms(
    ell: int,
    m: int,
    terms: list[Term],
    *,
    offset: int,
    transpose: bool = False,
) -> np.ndarray:
    columns = [_term_columns(ell, m, term, transpose=transpose) + offset for term in terms]
    return np.stack(columns, axis=1)


def _matrix_from_supports(num_rows: int, num_cols: int, supports: np.ndarray) -> np.ndarray:
    matrix = np.zeros((num_rows, num_cols), dtype=np.uint8)
    for row in range(num_rows):
        for col in supports[row]:
            matrix[row, col] ^= 1
    return matrix


@dataclass(frozen=True)
class BBCode:
    name: str
    label: str
    ell: int
    m: int
    a_terms: tuple[Term, ...]
    b_terms: tuple[Term, ...]
    hx_supports: np.ndarray
    hz_supports: np.ndarray
    hx: np.ndarray
    hz: np.ndarray
    x_logicals: np.ndarray
    z_logicals: np.ndarray
    a_maps: np.ndarray
    b_maps: np.ndarray
    a_t_maps: np.ndarray
    b_t_maps: np.ndarray

    @property
    def lm(self) -> int:
        return self.ell * self.m

    @property
    def n_data(self) -> int:
        return 2 * self.lm

    @property
    def n_checks(self) -> int:
        return 2 * self.lm

    @property
    def k(self) -> int:
        return int(self.x_logicals.shape[0])

    def validate(self) -> None:
        if self.hx.shape != (self.lm, self.n_data):
            raise ValueError("HX has unexpected shape")
        if self.hz.shape != (self.lm, self.n_data):
            raise ValueError("HZ has unexpected shape")
        if not np.all(self.hx.sum(axis=1) == 6):
            raise ValueError("expected every X-check to have weight 6")
        if not np.all(self.hz.sum(axis=1) == 6):
            raise ValueError("expected every Z-check to have weight 6")
        commutator = matmul_mod2(self.hx, self.hz.T)
        if np.any(commutator):
            raise ValueError("HX and HZ do not commute")


def build_bb_code(
    name: str = "bb_144",
    *,
    ell: int | None = None,
    m: int | None = None,
    a: str | Iterable[str | Term] | None = None,
    b: str | Iterable[str | Term] | None = None,
    compute_logicals: bool = True,
) -> BBCode:
    """Build a BB code from a known name or custom polynomial data."""
    if name != "custom":
        if name not in KNOWN_BB_CODES:
            choices = ", ".join(sorted(KNOWN_BB_CODES))
            raise ValueError(f"unknown code {name!r}; choices: {choices}")
        spec = KNOWN_BB_CODES[name]
        ell = int(spec["ell"])
        m = int(spec["m"])
        a = str(spec["a"])
        b = str(spec["b"])
        label = str(spec["label"])
    else:
        if ell is None or m is None or a is None or b is None:
            raise ValueError("custom code requires ell, m, a, and b")
        label = f"custom [[{2 * ell * m},k,?]]"

    assert ell is not None and m is not None and a is not None and b is not None
    a_terms = parse_polynomial(a)
    b_terms = parse_polynomial(b)
    if len(a_terms) != 3 or len(b_terms) != 3:
        raise ValueError("this generator expects weight-6 BB codes with three A and three B terms")

    lm = ell * m
    n_data = 2 * lm
    a_maps = np.stack([_term_columns(ell, m, term) for term in a_terms], axis=0)
    b_maps = np.stack([_term_columns(ell, m, term) for term in b_terms], axis=0)
    a_t_maps = np.stack([_term_columns(ell, m, term, transpose=True) for term in a_terms], axis=0)
    b_t_maps = np.stack([_term_columns(ell, m, term, transpose=True) for term in b_terms], axis=0)

    hx_supports = np.concatenate(
        [
            _supports_from_terms(ell, m, a_terms, offset=0),
            _supports_from_terms(ell, m, b_terms, offset=lm),
        ],
        axis=1,
    )
    hz_supports = np.concatenate(
        [
            _supports_from_terms(ell, m, b_terms, offset=0, transpose=True),
            _supports_from_terms(ell, m, a_terms, offset=lm, transpose=True),
        ],
        axis=1,
    )
    hx = _matrix_from_supports(lm, n_data, hx_supports)
    hz = _matrix_from_supports(lm, n_data, hz_supports)

    if compute_logicals:
        x_logicals, z_logicals = css_logical_bases(hx, hz)
    else:
        x_logicals = np.zeros((0, n_data), dtype=np.uint8)
        z_logicals = np.zeros((0, n_data), dtype=np.uint8)

    code = BBCode(
        name=name,
        label=label,
        ell=ell,
        m=m,
        a_terms=tuple(a_terms),
        b_terms=tuple(b_terms),
        hx_supports=hx_supports.astype(np.int64),
        hz_supports=hz_supports.astype(np.int64),
        hx=hx,
        hz=hz,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        a_maps=a_maps.astype(np.int64),
        b_maps=b_maps.astype(np.int64),
        a_t_maps=a_t_maps.astype(np.int64),
        b_t_maps=b_t_maps.astype(np.int64),
    )
    code.validate()
    return code
