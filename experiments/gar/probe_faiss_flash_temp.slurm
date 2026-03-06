#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --job-name=probe-faiss-flash
#SBATCH --output=/scratch/afw8937/logs/probe-faiss-flash-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --partition=h100_tandon

set -euo pipefail

VENV="/scratch/afw8937/efficient-search-agents/venv-temp"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: venv-temp not found at $VENV"
  exit 1
fi

echo "=== Node / GPU Info ==="
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi
$PYTHON -V
$PIP --version

# Ensure baseline deps inside temp env
$PIP install -q --upgrade pip setuptools wheel
$PIP install -q numpy

run_faiss_probe () {
  local ver="$1"
  echo
  echo "=== Probing faiss-gpu-cu12==$ver ==="
  $PIP uninstall -y faiss-gpu-cu12 faiss-cpu >/dev/null 2>&1 || true
  if ! $PIP install -q "faiss-gpu-cu12==$ver"; then
    echo "INSTALL_FAIL faiss-gpu-cu12==$ver"
    return 0
  fi

  $PYTHON - <<'PY'
import traceback
import numpy as np
import faiss
print('faiss_version', getattr(faiss, '__version__', None))
print('has_gpu_api', hasattr(faiss,'StandardGpuResources'), hasattr(faiss,'index_cpu_to_gpu'))
try:
    print('num_gpus', faiss.get_num_gpus())
except Exception as e:
    print('num_gpus_error', repr(e))
try:
    d = 64
    xb = np.random.rand(20000, d).astype('float32')
    xq = np.random.rand(8, d).astype('float32')
    cpu = faiss.IndexFlatIP(d)
    cpu.add(xb)
    res = faiss.StandardGpuResources()
    gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
    D, I = gpu.search(xq, 5)
    print('GPU_SMOKE_OK', D.shape, I.shape)
except Exception as e:
    print('GPU_SMOKE_FAIL', type(e).__name__, str(e))
    traceback.print_exc()
PY
}

run_faiss_probe "1.14.0"
run_faiss_probe "1.13.2"

# Optional flash-attn compatibility probe (no impact on FAISS)
echo
echo "=== Probing flash-attn install/import ==="
$PIP uninstall -y flash-attn >/dev/null 2>&1 || true
if timeout 900 $PIP install -q --no-build-isolation flash-attn; then
  $PYTHON - <<'PY'
import traceback
try:
    import flash_attn
    print('FLASH_ATTN_IMPORT_OK', getattr(flash_attn, '__file__', 'unknown'))
except Exception as e:
    print('FLASH_ATTN_IMPORT_FAIL', type(e).__name__, str(e))
    traceback.print_exc()
PY
else
  echo "FLASH_ATTN_INSTALL_FAIL"
fi

echo
echo "=== Probe Complete ==="
