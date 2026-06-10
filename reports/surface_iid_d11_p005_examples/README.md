# Surface-Code Detector-Event Examples

These figures visualize validation samples from the `d=11`, `p=0.05`, no-measurement-error
surface-code dataset.

Each image shows a `10 x 12` detector grid:

```text
red square = active detector event / syndrome bit 1
black cell  = inactive detector event / syndrome bit 0
label 0     = no logical flip
label 1     = logical flip
```

The `active detectors` count in the title is computed from the original 120-bit syndrome vector.
If multiple detector indices map to the same 2D display cell, the number of visible red cells can
be slightly smaller than the active-detector count.

Files:

```text
montage_high_contrast.png  Combined view with four label-1 and four label-0 examples
examples_summary.json      Metadata for the selected examples
sample_00_idx_2.png        label=1, active detectors=9
sample_01_idx_7.png        label=1, active detectors=9
sample_02_idx_8.png        label=1, active detectors=9
sample_03_idx_18.png       label=1, active detectors=11
sample_04_idx_0.png        label=0, active detectors=6
sample_05_idx_1.png        label=0, active detectors=15
sample_06_idx_3.png        label=0, active detectors=4
sample_07_idx_4.png        label=0, active detectors=12
```

The label is not determined by active-detector count alone. It depends on whether the sampled
error has nontrivial logical action.
