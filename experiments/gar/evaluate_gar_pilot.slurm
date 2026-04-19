#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --job-name=gar-eval
#SBATCH --output=/scratch/afw8937/logs/gar-eval-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --partition=h100_tandon

set -e

echo "================================================"
echo "GAR Pilot Evaluation (LLM Judge)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "================================================"

nvidia-smi

# ---- Environment Setup ----
VENV="/scratch/afw8937/efficient-search-agents/venv"
PYTHON="${VENV}/bin/python"
export PATH="${VENV}/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/afw8937/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus

GT="data/browsecomp_plus_decrypted.jsonl"
PILOT_DIR="/scratch/afw8937/efficient-search-agents/runs/gar_pilot"
EVAL_DIR="/scratch/afw8937/efficient-search-agents/evals/gar_pilot"
mkdir -p "${EVAL_DIR}"

# Evaluate each condition that has outputs
for CONDITION in baseline gar_no_gate gar_no_gate_alpha09; do
    INPUT="${PILOT_DIR}/${CONDITION}"
    if [ ! -d "${INPUT}" ]; then
        echo "Skipping ${CONDITION}: directory not found"
        continue
    fi
    NFILES=$(ls "${INPUT}"/*.json 2>/dev/null | wc -l)
    if [ "${NFILES}" -eq 0 ]; then
        echo "Skipping ${CONDITION}: no JSON files"
        continue
    fi

    echo ""
    echo "================================================"
    echo "Evaluating: ${CONDITION} (${NFILES} files)"
    echo "================================================"

    $PYTHON scripts_evaluation/evaluate_run.py \
        --input_dir "${INPUT}" \
        --ground_truth "${GT}" \
        --eval_dir "${EVAL_DIR}" \
        --model "Qwen/Qwen3-32B" \
        --tensor_parallel_size 1 \
        --batch_size 32 \
        --max_output_tokens 2048

    echo "Done: ${CONDITION}"
done

echo ""
echo "================================================"
echo "All evaluations complete!"
echo "Results in: ${EVAL_DIR}"
echo "End time: $(date)"
echo "================================================"
