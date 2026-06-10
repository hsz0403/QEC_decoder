# Surface-Code CNN Decoder

This branch contains a focused surface-code CNN decoder prototype.

The experiment is intentionally simple: it uses code-capacity surface-code data with no
measurement error and trains a CNN to map binary syndrome events to a binary logical-action
label.

```text
code:        rotated surface code
distance:    d = 11
error rate:  p = 0.05
noise:       iid data-error mechanisms
input:       syndrome / detector events, shape [shots, 120]
label:       logical_action, shape [shots, 1]
```

The main writeup is:

```text
docs/surface_code_cnn.md
```

It includes the dataset schema, visual examples, model architecture, training configs, loss,
training curves, baseline comparison, and current results.

## Results

Validation logical error rate:

```text
all-zero baseline: 0.23632
H=128 best:        0.23026
H=512 best:        0.22180
```

The H=512 checkpoint is the best current model, but its final epoch regresses, so evaluation should
use the saved best checkpoint rather than the final epoch.

## Visuals

Detector-event examples:

```text
reports/surface_iid_d11_p005_examples/montage_high_contrast.png
```

Training curves:

```text
reports/surface_code_training_curves.png
```

## Code Layout

```text
surface_code_capacity/
  data.py                         Surface-code iid dataset generation
  model.py                        SurfaceCodeCNNDecoder

scripts/
  generate_surface_code_iid.py    Generate train/validation data
  visualize_surface_code_samples.py
                                  Generate detector-event visualizations
  plot_surface_code_training.py   Plot training curves
  train_surface_code_cnn.py       Train CNN decoder

configs/
  surface_iid_d11_p005.yaml
  train_surface_cnn_d11_h128.yaml
  train_surface_cnn_d11_h512.yaml

docs/
  surface_code_cnn.md
```

## Reproduce

Generate data:

```bash
python scripts/generate_surface_code_iid.py --config configs/surface_iid_d11_p005.yaml
```

Visualize samples:

```bash
python scripts/visualize_surface_code_samples.py \
  --data runs/data/surface_iid_d11_p005_val.npz \
  --out-dir reports/surface_iid_d11_p005_examples \
  --num-samples 8
```

Train:

```bash
python scripts/train_surface_code_cnn.py --config configs/train_surface_cnn_d11_h128.yaml
python scripts/train_surface_code_cnn.py --config configs/train_surface_cnn_d11_h512.yaml
```

Plot curves:

```bash
python scripts/plot_surface_code_training.py
```

On Della, use the Slurm scripts:

```bash
sbatch slurm/surface_code_train_h128.sbatch
sbatch slurm/surface_code_train_h512.sbatch
```

## Notes

Large generated artifacts such as `.npz` datasets and `.pt` checkpoints are intentionally not
tracked. The repo keeps configs, code, small metrics, summaries, figures, and documentation.
