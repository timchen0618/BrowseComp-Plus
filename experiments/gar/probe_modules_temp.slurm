#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --job-name=probe-modules
#SBATCH --output=/scratch/afw8937/logs/probe-modules-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:05:00
#SBATCH --partition=cpu_short
set -euo pipefail

echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"

# try module command
if command -v module >/dev/null 2>&1; then
  echo "module command exists"
else
  echo "module command missing in shell, trying source init"
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi

if command -v module >/dev/null 2>&1; then
  module --version || true
  module avail 2>&1 | head -120 || true
  module avail cuda 2>&1 | head -120 || true
else
  echo "module command still unavailable"
fi

which nvcc || true
whereis nvcc || true
find /share/apps -maxdepth 4 -name nvcc 2>/dev/null | head -40 || true
