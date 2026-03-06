#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --job-name=probe-faiss-prereq
#SBATCH --output=/scratch/afw8937/logs/probe-faiss-prereq-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
#SBATCH --partition=h100_tandon

set -euo pipefail

echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi | head -30

for cmd in gcc g++ cmake make ninja swig git python3; do
  echo "--- $cmd ---"
  which $cmd || true
  $cmd --version 2>/dev/null | head -2 || true
done

echo "--- nvcc ---"
which nvcc || true
nvcc --version || true

echo "--- cuda dirs ---"
ls -d /usr/local/cuda* 2>/dev/null || true

