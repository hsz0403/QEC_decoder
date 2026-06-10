#!/bin/bash
#SBATCH --job-name=sc-cnn-h128
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/surface_code_train_h128_%j.out
#SBATCH --error=logs/surface_code_train_h128_%j.err

set -euo pipefail
cd /scratch/gpfs/MENGDIW/sh4809/QEC/qec-neural-decoder
PYTHON=/scratch/gpfs/MENGDIW/sh4809/QEC/ml_bb_decoding/.conda_envs/mlbb-gpu/bin/python

echo "Started at: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
"${PYTHON}" scripts/train_surface_code_cnn.py --config configs/train_surface_cnn_d11_h128.yaml
echo "Finished at: $(date)"
