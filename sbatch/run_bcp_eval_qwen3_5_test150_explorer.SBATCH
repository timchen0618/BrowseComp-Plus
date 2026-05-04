#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=eval_qwen3.5-122b-a10b_test150_explorer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afw8937@nyu.edu
#SBATCH --output=/scratch/afw8937/browsecomp-plus/logs/eval_qwen3.5-122b-a10b_test150_explorer_%j.out
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:2

# LLM-as-judge eval for any Qwen3.5-122B-A10B test150 explorer-prepended run.
# Mirrors run_bcp_eval_minimax_test150_explorer.SBATCH.
# Submit with: sbatch --export=ALL,RUN_NAME=qwen3.5-4b_vanilla_traj_orig_seed0 sbatch/run_bcp_eval_qwen3_5_test150_explorer.SBATCH

set -euo pipefail

if [[ -z "${RUN_NAME:-}" ]]; then echo "FATAL: must set RUN_NAME env var"; exit 1; fi

REPO_DIR=/home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus
VENV=/scratch/afw8937/efficient-search-agents/venv

cd "$REPO_DIR"
mkdir -p /scratch/afw8937/tmp
export TMPDIR=/scratch/afw8937/tmp

MODEL_NAME="qwen3.5-122b-a10b"
INPUT_DIR="runs/bcp/Qwen3-Embedding-8B/test150/${MODEL_NAME}/${RUN_NAME}"
EVAL_DIR="evals"

echo "=== Input: $INPUT_DIR ==="
test -d "$INPUT_DIR" || { echo "FATAL: input dir missing: $INPUT_DIR"; exit 1; }

module load openjdk/21.0.3
source "$VENV/bin/activate"

export HF_HOME=/scratch/afw8937/.huggingface

python scripts_evaluation/evaluate_run.py \
    --input_dir "$INPUT_DIR" \
    --eval_dir "$EVAL_DIR" \
    --ground_truth "data/browsecomp_plus_decrypted.jsonl" \
    --qrel_evidence "topics-qrels/bcp/qrel_evidence.txt" \
    --model "Qwen/Qwen3-32B" \
    --tensor_parallel_size 2 \
    --batch_size 64 \
    --force

echo "=== Done ==="
