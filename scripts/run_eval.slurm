#!/usr/bin/env bash
#SBATCH --job-name=psgjepa1d_eval
#SBATCH --account=bfid-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
set -euo pipefail

module purge
module load default-rh96
module load miniforge3-python
source $(conda info --base)/etc/profile.d/conda.sh
set +u   # env's activate.d hooks assume nounset off
conda activate /u/sobh/.conda/envs/mri_mrf_pytorch_env
set -u
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

cd /u/sobh/psgjepa1d
python -u eval/run_eval.py --ckpt-dir ckpt --out eval_results_v3.json --cache window_cache.npz
