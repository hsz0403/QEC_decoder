# Result Report: AlphaQubit-Style Neural Decoder Scaffold

Generated on 2026-05-15.

This report summarizes the current state of the repository, the real dataset that was
downloaded and inspected, the converted data used for the first machine-learning run, the
current ML model architecture, and the measured results.

## 1. Repository Location

Repository:

```text
/scratch/gpfs/MENGDIW/sh4809/QEC/qec-neural-decoder
```

Main implemented components:

```text
qecml/sim/stim_surface_code.py          Stim synthetic surface-code generation
qecml/data/stim_dataset.py              Common .npz save/load schema
qecml/data/coordinate_mapper.py         Flat detector events -> dense [B, T, S, F]
qecml/data/zenodo_6804040.py            Zenodo 6804040 inspection/conversion helpers
qecml/decoders/pymatching_decoder.py    PyMatching baseline
qecml/decoders/neural/mlp.py            FlatMLPDecoder
qecml/decoders/neural/temporal_cnn.py   TemporalCNNDecoder
qecml/decoders/neural/mini_alphaqubit.py MiniAlphaQubitDecoder
qecml/training/trainer.py               Training loop
qecml/training/evaluate.py              Checkpoint evaluation
```

Validation status:

```text
make test: 11 passed
ruff: passed
mypy: passed
```

The server default Python is 3.9, so the repo is Python 3.9+ compatible. The smoke configs use
`training.num_threads: 1` because small PyTorch CPU jobs were much faster and more stable with a
single thread on the login node.

## 2. Real Dataset

Dataset:

```text
Zenodo record: 6804040
Title: Data for "Suppressing quantum errors by scaling a surface code logical qubit"
DOI: 10.5281/zenodo.6804040
Provider: Google Quantum AI Team
```

Downloaded file:

```text
data/zenodo_6804040/raw/google_qec3v5_experiment_data.zip
```

Download/checksum:

```text
File size: 315,490,804 bytes
Human-readable size: about 301 MiB
MD5: a7fd8b481c3087090093106382dc217d
MD5 status: matches Zenodo API metadata
```

Inspection report:

```text
runs/zenodo_6804040_inspection.md
```

ZIP inventory:

```text
Total ZIP entries: 2095
README-like files found: 1
Candidate data/metadata files identified by inspection script: 916
Surface-code experiment directories: 130
Repetition-code experiment directories: 1
```

Surface-code directory breakdown:

```text
Basis X experiments: 65
Basis Z experiments: 65
Distance d=3 experiments: 104
Distance d=5 experiments: 26
Round counts present: r01, r03, r05, r07, r09, r11, r13, r15, r17, r19, r21, r23, r25
```

Important: this is the Google Sycamore surface-code / repetition-code dataset requested for
milestone 3. It is not the Willow color-code dataset.

## 3. Dataset File Format

The README in the ZIP describes one subdirectory per experiment. Directory names follow:

```text
{code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}
```

Example:

```text
surface_code_bX_d3_r01_center_3_5
```

Each experiment directory contains files such as:

```text
properties.yml
layout.svg
circuit_ideal.stim
circuit_noisy.stim
circuit_detector_error_model.dem
pij_from_even_for_odd.dem
pij_from_odd_for_even.dem
sweep.b8
measurements.b8
detection_events.b8
obs_flips_actual.01
obs_flips_predicted_by_pymatching.01
obs_flips_predicted_by_correlated_matching.01
```

For surface-code experiments, many directories also include:

```text
obs_flips_predicted_by_tensor_network_contraction.01
obs_flips_predicted_by_belief_matching.01
```

Relevant formats:

```text
detection_events.b8:
  Bit-packed detector event data. Shape after unpacking is [shots, circuit_detectors].
  Bits are little-endian inside each byte.

obs_flips_actual.01:
  Ground-truth logical observable flip labels.
  One text line per shot.
  All experiments inspected have one observable, so shape is [shots, 1].

obs_flips_predicted_by_pymatching.01:
  Provided PyMatching predictions from the dataset authors.
```

## 4. Example Experiment Used for ML

The first real-data ML run used:

```text
Experiment: surface_code_bX_d3_r01_center_3_5
Code: surface_code
Basis: X
Distance: 3
Rounds: 1
Center data qubit: row 3, col 5
```

Original experiment properties:

```yaml
type: surface_code_memory_experiment
basis: X
rounds: 1
distance: 3
data_qubits: 9
measure_qubits: 8
shots: 50000
center_data_qubit_row: 3
center_data_qubit_col: 5
circuit_measurements: 17
circuit_sweep_bits: 9
circuit_detectors: 8
circuit_observables: 1
circuit_qubits: 17
```

Stim circuit coordinate data:

```text
detector_coords shape: [8, 3]
```

For comparison, a larger d=5 example in the ZIP has:

```yaml
experiment: surface_code_bX_d5_r25_center_5_5
basis: X
rounds: 25
distance: 5
data_qubits: 25
measure_qubits: 24
shots: 50000
circuit_measurements: 625
circuit_sweep_bits: 25
circuit_detectors: 600
circuit_observables: 1
circuit_qubits: 49
```

The repetition-code directory has:

```yaml
experiment: repetition_code_bZ_d25_r50_center_5_5
basis: Z
rounds: 50
distance: 25
shots: 500000
circuit_measurements: 1225
circuit_detectors: 1224
circuit_observables: 1
```

## 5. Conversion to Common Schema

Implemented converter:

```text
scripts/convert_zenodo_6804040.py
qecml/data/zenodo_6804040.py
```

Example commands used:

```bash
python scripts/convert_zenodo_6804040.py \
  --experiment surface_code_bX_d3_r01_center_3_5 \
  --max-shots 40000 \
  --output runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_train.npz

python scripts/convert_zenodo_6804040.py \
  --experiment surface_code_bX_d3_r01_center_3_5 \
  --skip-shots 40000 \
  --max-shots 10000 \
  --output runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
```

Common `.npz` schema:

```text
events: uint8 array, shape [shots, n_detectors]
labels: uint8 array, shape [shots, n_observables]
detector_coords: float32 array, shape [n_detectors, coord_dim]
metadata_json: JSON string
```

Converted train split:

```text
Path: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_train.npz
File size: 27,441 bytes
Shots: 40,000
events shape: [40000, 8]
labels shape: [40000, 1]
detector_coords shape: [8, 3]
skip_shots: 0
label positive rate: 0.079375
```

Converted validation split:

```text
Path: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
File size: 7,974 bytes
Shots: 10,000
events shape: [10000, 8]
labels shape: [10000, 1]
detector_coords shape: [8, 3]
skip_shots: 40000
label positive rate: 0.0795
```

Earlier 1k ingestion smoke file:

```text
Path: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_1k.npz
File size: 2,017 bytes
Shots: 1,000
events shape: [1000, 8]
labels shape: [1000, 1]
detector_coords shape: [8, 3]
label positive rate: 0.098
```

The converted `.npz` files are very small because this particular d=3, r=1 experiment has only
8 detector bits per shot.

## 6. Current ML Model

The real-data ML result below used `FlatMLPDecoder`.

Source:

```text
qecml/decoders/neural/mlp.py
```

Input:

```text
Flat detector event vector with shape [B, N]
For this run: N = 8
```

Output:

```text
Logical observable flip logits with shape [B, K]
For this run: K = 1
```

Architecture:

```text
Linear(num_detectors, hidden_dim)
GELU
LayerNorm(hidden_dim)
Dropout(dropout)
Linear(hidden_dim, hidden_dim)
GELU
Linear(hidden_dim, num_observables)
```

Concrete architecture for this run:

```text
num_detectors: 8
hidden_dim: 64
num_observables: 1
dropout: 0.1
```

Parameter count:

```text
Total parameters: 4,929
Trainable parameters: 4,929
```

Parameter breakdown by layer shape:

```text
Linear(8 -> 64):        8*64 + 64 = 576
LayerNorm(64):          64 weight + 64 bias = 128
Linear(64 -> 64):       64*64 + 64 = 4,160
Linear(64 -> 1):        64*1 + 1 = 65
Total:                  4,929
```

This model is a flat baseline. It does not explicitly use lattice geometry, temporal recurrence,
attention, or local matching structure.

Other implemented models:

```text
TemporalCNNDecoder:
  Expects dense [B, T, S, F] detector events.
  Flattens site/features per time and uses 1D convolutions over time.

MiniAlphaQubitDecoder:
  Expects dense [B, T, S, F] detector events.
  Maintains per-site hidden state.
  Uses GRUCell-style recurrent updates per site.
  Uses MultiheadAttention over sites.
  Uses feed-forward blocks and final site pooling.
```

MiniAlphaQubitDecoder smoke results were later run on real Zenodo data:

```text
Experiment: surface_code_bX_d3_r01_center_3_5
Dense validation shape: [10000, 3, 4, 1]
hidden_dim: 64
num_layers: 1
num_heads: 4
dropout: 0.1
parameters: 79,681
epochs: 10
loss: binary cross entropy with logits
optimizer: AdamW
validation LER: 0.0446
validation BCE: 0.10686307400465012
validation accuracy: 0.9554
validation ROC-AUC: 0.9893060580283479
```

This is slightly better than the flat MLP on the same r=1 split (`0.0491` LER), but still worse
than the provided PyMatching predictions (`0.0139` LER).

A harder real-data smoke run was also attempted:

```text
Experiment: surface_code_bX_d3_r25_center_3_5
Dense shape: [B, 26, 8, 1]
MiniAlphaQubit-style epochs: 5
MiniAlphaQubit-style best LER: 0.4855
FlatMLP best LER: 0.4839
Provided PyMatching LER: 0.4258
```

The r=25 result did not learn beyond a trivial baseline. It should be treated as a negative CPU
smoke result, not as evidence that the architecture is competitive.

## 7. Training Configuration

Config file:

```text
configs/train_mlp_zenodo_d3_r01_smoke.yaml
```

Training config:

```yaml
data:
  train: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_train.npz
  val: runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
model:
  name: mlp
  hidden_dim: 64
  dropout: 0.1
training:
  batch_size: 512
  epochs: 10
  lr: 0.0005
  weight_decay: 0.00001
  grad_clip_norm: 1.0
  pos_weight: auto
  seed: 0
  num_threads: 1
output:
  run_dir: runs/mlp_zenodo_d3_r01_smoke
```

Loss:

```text
Binary cross entropy with logits
pos_weight: auto
```

The automatic positive-class weight is used because logical flips are less common than non-flips.
For the train split, the positive label rate is about 7.94%.

Checkpoint and logs:

```text
Checkpoint: runs/mlp_zenodo_d3_r01_smoke/checkpoint.pt
Checkpoint size: 24 KiB
Run config copy: runs/mlp_zenodo_d3_r01_smoke/config.yaml
Metrics: runs/mlp_zenodo_d3_r01_smoke/metrics.jsonl
Metrics file size: 2.9 KiB
```

## 8. ML Training Metrics

Validation metrics by epoch:

| Epoch | Train Loss | Val BCE | Val LER | Val Accuracy | Val ROC-AUC | Val Pred Positive Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8952 | 0.3475 | 0.1270 | 0.8730 | 0.9353 | 0.2013 |
| 2 | 0.4592 | 0.2468 | 0.1253 | 0.8747 | 0.9736 | 0.1996 |
| 3 | 0.3629 | 0.2008 | 0.1242 | 0.8758 | 0.9833 | 0.1973 |
| 4 | 0.3196 | 0.1796 | 0.0895 | 0.9105 | 0.9853 | 0.1612 |
| 5 | 0.2980 | 0.1510 | 0.0732 | 0.9268 | 0.9867 | 0.1433 |
| 6 | 0.2775 | 0.1416 | 0.0673 | 0.9327 | 0.9882 | 0.1370 |
| 7 | 0.2648 | 0.1366 | 0.0669 | 0.9331 | 0.9888 | 0.1368 |
| 8 | 0.2564 | 0.1352 | 0.0534 | 0.9466 | 0.9895 | 0.1229 |
| 9 | 0.2478 | 0.1247 | 0.0491 | 0.9509 | 0.9900 | 0.1186 |
| 10 | 0.2497 | 0.1290 | 0.0631 | 0.9369 | 0.9903 | 0.1342 |

Best checkpoint selection:

```text
Best validation LER: 0.0491
Best epoch: 9
```

Final evaluated checkpoint metrics on the 10k-shot validation split:

```text
BCE: 0.12466351687908173
Logical error rate at threshold 0.5: 0.0491
Accuracy: 0.9509
Positive label rate: 0.0795
Prediction positive rate: 0.1186
Expected calibration error: 0.06505155321611092
ROC-AUC: 0.9900398675863199
Validation shots: 10000
```

ML confusion matrix on validation split:

```text
True positives: 745
False positives: 441
True negatives: 8764
False negatives: 50
Total mistakes: 491
LER: 491 / 10000 = 0.0491
```

## 9. PyMatching Baseline on Same Validation Segment

The Zenodo ZIP contains official PyMatching predictions for the same experiment:

```text
surface_code_bX_d3_r01_center_3_5/obs_flips_predicted_by_pymatching.01
```

Using the same validation segment:

```text
Shots: 10000
Segment: shots 40000 through 49999
Ground truth: obs_flips_actual.01
Prediction: obs_flips_predicted_by_pymatching.01
```

PyMatching result:

```text
Logical error rate: 0.0139
Mistakes: 139 / 10000
```

PyMatching confusion matrix:

```text
True positives: 698
False positives: 42
True negatives: 9163
False negatives: 97
```

Comparison:

| Decoder | Validation Shots | LER | Mistakes | Accuracy |
|---|---:|---:|---:|---:|
| FlatMLPDecoder | 10000 | 0.0491 | 491 | 0.9509 |
| Provided PyMatching | 10000 | 0.0139 | 139 | 0.9861 |

Conclusion:

```text
The current MLP is learning meaningful signal from the real detector events, but it is much worse
than PyMatching on logical error rate. This is expected: the MLP is a sanity-check baseline, not a
QEC-structured decoder.
```

## 10. Synthetic Stim Smoke Results

Synthetic data was also generated from Stim:

```text
Train: runs/data/stim_d3_smoke_train.npz
Val: runs/data/stim_d3_smoke_val.npz
```

Synthetic d=3 smoke config:

```text
basis: z
distance: 3
rounds: 3
p: 0.005
shots_train: 20000
shots_val: 5000
```

Synthetic generated data:

```text
n_detectors: 24
n_observables: 1
train label positive rate: 0.104950
val label positive rate: 0.110400
detector coordinates: available
```

PyMatching on synthetic validation set:

```text
Path: runs/baselines/pymatching_stim_d3_smoke.json
pymatching_ler: 0.02
num_shots: 5000
num_mistakes: 100
prediction_shape: [5000, 1]
label_shape: [5000, 1]
```

MLP on synthetic validation set:

```text
Checkpoint: runs/mlp_d3_smoke/checkpoint.pt
LER: 0.08
Accuracy: 0.92
BCE: 0.19049444794654846
ROC-AUC: 0.9789233151522262
```

## 11. Current Interpretation

What has been demonstrated:

```text
1. The Zenodo 6804040 dataset can be downloaded and checksum-verified.
2. The archive can be inspected programmatically.
3. Real surface-code detection_events.b8 can be unpacked into dense binary detector-event arrays.
4. Real obs_flips_actual.01 labels can be loaded into the common label schema.
5. Stim detector coordinates can be extracted from circuit_ideal.stim.
6. The common .npz schema works for both synthetic Stim data and real Zenodo data.
7. A small neural decoder can train on real Google surface-code data and achieve nontrivial ROC-AUC.
8. The current flat MLP does not beat PyMatching.
```

Main limitation of the current ML result:

```text
The model was trained only on one very small d=3, r=1 experiment with 8 detectors.
This is a first ingestion and sanity-check result, not a competitive AlphaQubit-style result.
```

Why the MLP is insufficient:

```text
1. It treats detector events as a flat vector.
2. It does not explicitly model space-time structure.
3. It does not use the detector error model.
4. It does not use matching graph structure.
5. It does not use temporal recurrence because r=1 is almost non-temporal.
```

Expected next ML step:

```text
Train on a richer real-data experiment, e.g. d=3 with more rounds or d=5/r25, and use the dense
coordinate mapping with MiniAlphaQubitDecoder.
```

## 12. Recommended Next Experiments

Short-term:

```text
1. Convert a d=3, r25 experiment and train MLP vs MiniAlphaQubit.
2. Convert a d=5, r25 experiment and test memory/runtime.
3. Evaluate the provided PyMatching predictions on every converted split.
4. Add a script that automatically creates train/val splits from any Zenodo experiment.
5. Add a result table over multiple center locations and both X/Z bases.
```

Modeling:

```text
1. Run MiniAlphaQubitDecoder on dense [B, T, S, F] real data.
2. Compare FlatMLPDecoder, TemporalCNNDecoder, MiniAlphaQubitDecoder, and PyMatching.
3. Add calibration plots and threshold sweeps.
4. Add per-round/per-distance ablations.
```

Cluster/GPU:

```text
1. Use sbatch for MiniAlphaQubit d=5/r25 jobs.
2. Keep CPU smoke configs small.
3. Save every config, checkpoint, and metrics JSONL for reproducibility.
```

## 13. Reproduction Commands

Verify code:

```bash
cd /scratch/gpfs/MENGDIW/sh4809/QEC/qec-neural-decoder
make test
make lint
```

Inspect Zenodo:

```bash
python scripts/inspect_zenodo_6804040.py
```

Convert real d=3/r1 train and val split:

```bash
python scripts/convert_zenodo_6804040.py \
  --experiment surface_code_bX_d3_r01_center_3_5 \
  --max-shots 40000 \
  --output runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_train.npz

python scripts/convert_zenodo_6804040.py \
  --experiment surface_code_bX_d3_r01_center_3_5 \
  --skip-shots 40000 \
  --max-shots 10000 \
  --output runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
```

Train MLP:

```bash
python scripts/train.py --config configs/train_mlp_zenodo_d3_r01_smoke.yaml
```

Evaluate MLP:

```bash
python scripts/evaluate.py \
  --checkpoint runs/mlp_zenodo_d3_r01_smoke/checkpoint.pt \
  --data runs/data/zenodo_6804040_surface_code_bX_d3_r01_center_3_5_val.npz
```
