import numpy as np

from qecml.data.zenodo_6804040 import read_01, unpack_b8


def test_unpack_b8_little_endian_rows():
    packed = bytes([0b00000101, 0b00000010])
    arr = unpack_b8(packed, shots=2, bits_per_shot=4)
    assert arr.dtype == np.uint8
    assert arr.tolist() == [[1, 0, 1, 0], [0, 1, 0, 0]]


def test_read_01_labels():
    labels = read_01(b"0\n1\n0\n", shots=3, observables=1)
    assert labels.shape == (3, 1)
    assert labels.tolist() == [[0], [1], [0]]
