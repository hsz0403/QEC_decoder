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
