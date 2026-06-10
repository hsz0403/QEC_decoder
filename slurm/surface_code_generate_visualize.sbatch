#!/bin/bash
#SBATCH --job-name=sc-d11-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/surface_code_generate_%j.out
#SBATCH --error=logs/surface_code_generate_%j.err

set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/QEC/qec-neural-decoder
PYTHON=/scratch/gpfs/MENGDIW/sh4809/QEC/ml_bb_decoding/.conda_envs/mlbb-gpu/bin/python

echo "Started at: $(date)"
"${PYTHON}" scripts/generate_surface_code_iid.py --config configs/surface_iid_d11_p005.yaml
"${PYTHON}" scripts/visualize_surface_code_samples.py \
  --data runs/data/surface_iid_d11_p005_val.npz \
  --out-dir reports/surface_iid_d11_p005_examples \
  --num-samples 8
echo "Finished at: $(date)"
