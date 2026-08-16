#!/usr/bin/env bash
#SBATCH --job-name=psgjepa1d_arms
#SBATCH --account=bfid-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
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
mkdir -p ckpt logs

CACHE=/u/sobh/psgjepa1d/window_cache.npz
N_WIN=2000000
N_EP=30

# Build cache if missing (~5 min for 2M windows on GPU-node CPUs).
if [ ! -f "${CACHE}" ]; then
  echo "=== building window cache (${N_WIN} windows) ==="
  python -u -c "
import numpy as np, time
from psgjepa1d.data import gen_windows
t0=time.time()
S, U = gen_windows(${N_WIN}, 7, seed=3072)
np.savez_compressed('${CACHE}', S=S, U=U)
print(f'wrote ${CACHE}  S={S.shape}  U={U.shape}  in {time.time()-t0:.0f}s', flush=True)
"
fi

# 2M windows x 30 epochs -- config default, more capacity.
# lr=3e-3 from the config was unstable (loss blew up 0.9 -> 9000); keep 1e-3.
COMMON="data.n_windows=${N_WIN} trainer.max_epochs=${N_EP} optimizer.lr=1.0e-3"

for entry in \
  "baseline|loss.grounding.weight=0.0" \
  "static|loss.grounding.use_transition=false loss.grounding.use_velocity=false" \
  "transition|loss.grounding.use_static=false" \
  "full|"
do
  NAME=${entry%%|*}
  OVR=${entry#*|}
  OUT=ckpt/${NAME}.pt
  if [ -f "${OUT}" ]; then
    echo "=== ${NAME}: already trained at ${OUT}, skipping ==="
    continue
  fi
  echo "=== training arm=${NAME}  overrides='${OVR}' ==="
  python -u train.py --cache ${CACHE} --out ${OUT} ${OVR} ${COMMON}
  echo "=== saved ${OUT} ==="
done

echo "=== all arms complete ==="
ls -la ckpt/
