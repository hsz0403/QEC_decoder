# qec-neural-decoder

Minimal AlphaQubit-1-style surface-code neural decoder scaffold. This repo starts with
Stim-generated rotated surface-code memory data, compares against a PyMatching MWPM baseline,
and trains small neural decoders.

This is research infrastructure, not an official AlphaQubit reproduction.

## Quickstart

```bash
make setup
make test
python scripts/generate_stim_dataset.py --config configs/stim_d3_smoke.yaml
python scripts/run_pymatching_baseline.py --data runs/data/stim_d3_smoke_val.npz
python scripts/train.py --config configs/train_mlp_d3_smoke.yaml
python scripts/train.py --config configs/train_mini_aq_d3_smoke.yaml
python scripts/evaluate.py --checkpoint runs/mini_aq_d3_smoke/checkpoint.pt --data runs/data/stim_d3_smoke_val.npz
```

On clusters where GPU access requires `sbatch`, keep the smoke tests on CPU and submit larger
training jobs through the scheduler. The training script automatically uses CUDA only when a GPU
is visible to the job. The default smoke configs set `training.num_threads: 1` because many-login
node CPU thread pools can be slower than a single thread for these tiny models.

## Concepts

A physical qubit is a hardware-level qubit that can suffer errors. A logical qubit is encoded
across many physical qubits so that errors can be detected and corrected. Stabilizers are
commuting measurements that reveal error information without directly measuring the logical
state. A syndrome is the collection of stabilizer measurement outcomes. A detection event is a
change in syndrome information that signals a likely error in space-time. A logical observable
flip is an encoded logical failure label, usually the target output of a decoder in a memory
experiment.

A surface-code memory experiment repeatedly measures stabilizers for a distance-`d` code and asks
whether the stored logical qubit was flipped. Stim generates fast synthetic stabilizer circuits and
detector samples. PyMatching implements minimum-weight perfect matching, a standard strong
baseline for graphlike detector error models. The AlphaQubit-style neural decoder here is a small
recurrent transformer-like model over detector events, intended as a readable starting point for
later adaptation experiments.

## Repo Structure

```text
configs/          YAML configs for data generation and training
qecml/sim/        Stim circuit generation and sampling
qecml/data/       Common NPZ schema, coordinate mapping, Zenodo inspection helpers
qecml/decoders/   PyMatching and neural decoders
qecml/training/   Metrics, losses, trainer, checkpoint evaluation
scripts/          CLI entry points
tests/            Unit and smoke tests
notebooks/        Starter notebooks
```

## Generate Data

```bash
python scripts/generate_stim_dataset.py --config configs/stim_d3_smoke.yaml
```

This writes common-schema `.npz` files with:

```text
events: [shots, n_detectors]
labels: [shots, n_observables]
detector_coords: optional [n_detectors, coord_dim]
metadata_json: JSON metadata
```

## Run PyMatching

```bash
python scripts/run_pymatching_baseline.py --data runs/data/stim_d3_smoke_val.npz
```

The script reconstructs the Stim circuit from dataset metadata, builds a detector error model,
runs PyMatching, prints logical error rate, and saves metrics under `runs/baselines/`.

## Train Neural Decoders

Flat MLP:

```bash
python scripts/train.py --config configs/train_mlp_d3_smoke.yaml
```

Mini AlphaQubit-style recurrent transformer:

```bash
python scripts/train.py --config configs/train_mini_aq_d3_smoke.yaml
```

Each run saves:

```text
runs/<run_name>/checkpoint.pt
runs/<run_name>/metrics.jsonl
runs/<run_name>/config.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint runs/mini_aq_d3_smoke/checkpoint.pt \
  --data runs/data/stim_d3_smoke_val.npz
```

Metrics include BCE, logical error rate at threshold 0.5, accuracy, label positive rate,
prediction positive rate, ROC-AUC when both classes are present, and calibration error.

## Inspect Zenodo 6804040

Zenodo record 6804040 is optional and not required for the synthetic pipeline.

```bash
python scripts/download_zenodo_6804040.py
python scripts/inspect_zenodo_6804040.py
```

The inspection script prints the archive tree, extracts README-like files, identifies likely
sample/circuit/prediction/metadata files, and writes `runs/zenodo_6804040_inspection.md`.

## Data Examples

The examples below come from the downloaded Zenodo 6804040 experiment:

```text
surface_code_bX_d3_r01_center_3_5
```

This is a surface-code memory experiment with:

```text
basis: X
distance: 3
rounds: 1
shots: 50000
circuit_detectors: 8
circuit_observables: 1
```

The raw archive stores detector events in Stim `b8` format, where each shot is bit-packed and
byte-aligned. For this experiment there are 8 detector bits per shot, so each shot is exactly one
byte in `detection_events.b8`. The first 16 raw bytes are:

```python
[96, 66, 0, 0, 0, 68, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0]
```

Labels are stored in `obs_flips_actual.01`, one logical observable flip per line. The first 12
raw label lines are:

```python
["0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"]
```

The Zenodo archive also provides PyMatching predictions. The first 12 raw PyMatching prediction
lines for the same experiment are:

```python
["0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0"]
```

After conversion to this repo's common `.npz` schema, the validation split used for the first ML
run is:

```text
file: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
shots: 10000
source shots: 40000 through 49999
events shape: [10000, 8]
labels shape: [10000, 1]
detector_coords shape: [8, 3]
label positive rate: 0.0795
```

The first 8 converted validation detector-event rows are:

```python
[
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
]
```

The corresponding ground-truth logical observable flip labels are:

```python
[0, 0, 0, 0, 0, 0, 0, 1]
```

The detector coordinates extracted from `circuit_ideal.stim` are:

```python
[
    [1.0, 4.0, 0.0],
    [3.0, 4.0, 0.0],
    [3.0, 6.0, 0.0],
    [5.0, 6.0, 0.0],
    [1.0, 4.0, 1.0],
    [3.0, 4.0, 1.0],
    [3.0, 6.0, 1.0],
    [5.0, 6.0, 1.0],
]
```

For these same 8 validation shots, the current trained `FlatMLPDecoder` produced:

```python
logits = [-2.7008, -6.3271, -0.0952, -0.0649, -6.3271, -6.3271, -2.5803, 5.5866]
probs  = [0.0629, 0.0018, 0.4762, 0.4838, 0.0018, 0.0018, 0.0704, 0.9963]
preds  = [0, 0, 0, 0, 0, 0, 0, 1]
```

The Zenodo-provided PyMatching predictions on the same 8 validation shots are:

```python
[0, 0, 0, 0, 0, 0, 0, 1]
```

On the full 10,000-shot validation split, the current flat MLP reached logical error rate
`0.0491`, while the provided PyMatching predictions reached `0.0139`.

## Known Limitations

This is not an official AlphaQubit reproduction.
This does not use Google/DeepMind code or weights.
This starts with simplified synthetic noise.
The first models are intentionally small.
Online adaptation/RL is future work.
Coordinate mapping is intentionally conservative and falls back instead of silently reshaping
ambiguous detector coordinates.

## Next Steps Toward Online Adaptation

Add synthetic drift benchmarks, supervised fine-tuning baselines, parameter-efficient adapters,
partial-feedback experiments, and richer nonuniform or biased noise models after the synthetic
Stim/PyMatching/neural pipeline is stable.
