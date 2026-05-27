# BB-code ML decoder dataset generator

This workspace contains a small dataset-generation base for training neural decoders on the
bivariate bicycle (BB) quantum LDPC codes from Bravyi et al.,
[arXiv:2308.07915](https://arxiv.org/abs/2308.07915). The generator follows the same
high-level data strategy used by the recurrent surface-code decoder work in Bausch et al.,
[arXiv:2310.05900](https://arxiv.org/abs/2310.05900): simulate memory experiments, record
repeated stabilizer measurements and detection events, and train the network to predict final
logical flips.

## What is implemented

- BB code construction for the weight-6 examples in Table 3 of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
- CSS check matrices `HX = [A | B]` and `HZ = [B.T | A.T]`.
- Paired X/Z logical bases computed over GF(2), used to label final logical flips.
- Two simulation backends:
  - `circuit`: a Pauli-frame simulator for the depth-8 BB syndrome cycle from Table 5 of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915), with standard circuit-level depolarizing faults on CNOTs, initialization, measurement, and idle locations.
  - `phenomenological`: repeated data depolarizing noise plus noisy syndrome measurements.
- HDF5 output with measurement channels, detection-event channels, clean final syndrome, final detection event, and logical labels.
- Optional simple Gaussian analog/soft measurement channels for experiments similar in spirit to the I/Q-soft-input setting of [arXiv:2310.05900](https://arxiv.org/abs/2310.05900).

This is a dataset base, not a finished decoder. It intentionally does not include BP-OSD,
MWPM, leakage, crosstalk, or a full hardware-calibrated analog readout model.

## Setup

```bash
python -m pip install -e ".[dev]"
```

If `numpy`, `h5py`, and `pytest` are already installed, no extra setup is required for the
standalone generator.

## Quick start

Generate a small circuit-level dataset for the `[[72,12,6]]` BB code:

```bash
python -m bbcode_dataset.generate \
  --code bb_72 \
  --noise circuit \
  --cycles 6 \
  --shots 1000 \
  --p 0.001 \
  --seed 1 \
  --out data/bb72_circuit_p001.h5
```

Generate a larger phenomenological dataset:

```bash
python -m bbcode_dataset.generate \
  --code bb_144 \
  --noise phenomenological \
  --cycles 12 \
  --shots 100000 \
  --p-data 0.001 \
  --p-meas 0.001 \
  --batch-size 2048 \
  --out data/bb144_pheno_p001.h5
```

Generate circuit-level samples with simple analog/soft measurement channels:

```bash
python -m bbcode_dataset.generate \
  --code bb_144 \
  --noise circuit \
  --cycles 12 \
  --shots 20000 \
  --p 0.001 \
  --analog-snr 10 \
  --out data/bb144_circuit_p001_soft.h5
```

## Supported BB codes

| CLI name | Code | `ell,m` | A | B |
| --- | --- | --- | --- | --- |
| `bb_72` | `[[72,12,6]]` | `6,6` | `x3 + y + y2` | `y3 + x + x2` |
| `bb_90` | `[[90,8,10]]` | `15,3` | `x9 + y + y2` | `1 + x2 + x7` |
| `bb_108` | `[[108,8,10]]` | `9,6` | `x3 + y + y2` | `y3 + x + x2` |
| `bb_144` | `[[144,12,12]]` | `12,6` | `x3 + y + y2` | `y3 + x + x2` |
| `bb_288` | `[[288,12,18]]` | `12,12` | `x3 + y2 + y7` | `y3 + x + x2` |
| `bb_360` | `[[360,12,<=24]]` | `30,6` | `x9 + y + y2` | `y3 + x25 + x26` |
| `bb_756` | `[[756,16,<=34]]` | `21,18` | `x3 + y10 + y17` | `y5 + x3 + x19` |

Custom weight-6 BB codes are also supported:

```bash
python -m bbcode_dataset.generate \
  --code custom --ell 12 --m 6 \
  --a "x3,y1,y2" \
  --b "y3,x1,x2" \
  --noise circuit \
  --shots 1000 \
  --cycles 12 \
  --out data/custom.h5
```

## HDF5 schema

All binary arrays are stored as `uint8`.

- `measurements`: shape `(shots, cycles, n_checks)`.
- `events`: shape `(shots, cycles, n_checks)`, where `events[t] = measurements[t] XOR measurements[t-1]` and the previous measurement before cycle 0 is zero.
- `events_with_final`: shape `(shots, cycles + 1, n_checks)`, appending the final clean data-readout event.
- `final_events`: shape `(shots, n_checks)`, equal to `final_syndrome XOR measurements[-1]`.
- `final_syndrome`: clean final syndrome from the accumulated final data Pauli frame.
- `logical_flips`: shape `(shots, 2*k)`. The first `k` bits are X-logical flips caused by final Z errors; the last `k` bits are Z-logical flips caused by final X errors.
- `code/hx`, `code/hz`: dense CSS check matrices.
- `code/hx_supports`, `code/hz_supports`: weight-6 check supports for fast custom loaders.
- `code/x_logicals`, `code/z_logicals`: paired logical bases used for labels.
- `analog_measurements`, `soft_measurements`: present only when `--analog-snr` is set.

Check ordering is always all X-check outcomes first, followed by all Z-check outcomes.

## Suggested training inputs

A good first neural-decoder input tensor is:

```python
inputs = np.stack([measurements, events], axis=-1)
targets = logical_flips
```

For a recurrent model, feed one cycle at a time with shape `(batch, n_checks, channels)`.
For a transformer or temporal convolution model, use `(batch, cycles, n_checks, channels)`.
Train with binary cross entropy on all `2*k` logical-flip bits.

For experiments closer to arXiv:2310.05900, include `events_with_final` and either hard
measurements or `soft_measurements` as additional channels. Use separate train/dev/test
files generated with different seeds, and sweep error rates or use a simple curriculum such
as `p in {0.0005, 0.00075, 0.001, 0.0015, 0.002}`.

## Verification

Run the tests:

```bash
python -m pytest tests/test_bbcode_dataset.py
```

Generate a tiny sample and inspect the stored shapes:

```bash
python -m bbcode_dataset.generate --code bb_72 --noise circuit --cycles 3 --shots 16 --p 0.001 --out runs/data/bb_smoke.h5
python - <<'PY'
import h5py
with h5py.File("runs/data/bb_smoke.h5", "r") as f:
    for key in ["measurements", "events_with_final", "final_syndrome", "logical_flips"]:
        print(key, f[key].shape, f[key].dtype)
    print(dict(f["code"].attrs))
PY
```

## Development notes

The `circuit` backend is the most BB-specific generator. It follows the Table 5 schedule:
X-check ancillas are initialized in the X basis and measured in the X basis; Z-check ancillas
are initialized in the Z basis and measured in the Z basis. CNOT faults are sampled uniformly
from the 15 non-identity two-qubit Paulis, idle faults uniformly from `X/Y/Z`, initialization
faults prepare the orthogonal basis state, and measurement faults flip the classical outcome.

The next useful extensions are leakage states, crosstalk faults that trigger correlated check
events, and calibration-derived nonuniform circuit error rates.
