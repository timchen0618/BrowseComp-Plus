#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=glm-4.7-flash_summarize_test300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afw8937@nyu.edu
#SBATCH --output=/scratch/afw8937/browsecomp-plus/logs/glm-4.7-flash_summarize_test300_%j.out
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2

# Self-summarize first-run GLM-4.7-Flash trajectories for test300 qids.
# Uses summarize_trajectories.py reading from full/glm-4.7-flash/seed0/,
# filtered to the 150 test qids via --queries-tsv.
# Output: summaries/glm-4.7-flash_test300.jsonl (resumable — appends).
# Submit with: sbatch sbatch/run_bcp_summarize_glm_test300.SBATCH

set -euo pipefail

MODEL_NAME="glm-4.7-flash"
model_path="zai-org/GLM-4.7-Flash"
TENSOR_PARALLEL_SIZE=2
EMBED_RESERVE_GB=5
MAX_MODEL_LEN=131072

REPO_DIR=/home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus
VENV=/scratch/afw8937/efficient-search-agents/venv

SIF=/scratch/afw8937/singularity_images/vllm_nightly.sif
OVERLAY=/scratch/afw8937/singularity_images/vllm_nightly_overlay.img
SINGULARITY=/share/apps/apptainer/bin/singularity

dataset=bcp
seed=0

cd "$REPO_DIR"
mkdir -p /scratch/afw8937/tmp summaries
export TMPDIR=/scratch/afw8937/tmp

PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "=== Container sanity check ==="
test -f "$SIF"     || { echo "FATAL: SIF missing at $SIF"; exit 1; }
test -f "$OVERLAY" || { echo "FATAL: overlay missing at $OVERLAY"; exit 1; }

echo "=== Computing dynamic GPU_UTIL ==="
GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_TOTAL_GB=$((GPU_TOTAL_MIB / 1024))
GPU_UTIL=$(python3 -c "print(round(($GPU_TOTAL_GB - $EMBED_RESERVE_GB) / $GPU_TOTAL_GB, 2))")
echo "    detected: ${GPU_TOTAL_GB}GB/GPU, reserve: ${EMBED_RESERVE_GB}GB → GPU_UTIL=$GPU_UTIL"

echo "=== Starting vLLM server (GLM-4.7-Flash, TP=$TENSOR_PARALLEL_SIZE) ==="
$SINGULARITY exec --nv \
    --overlay "$OVERLAY":ro \
    --bind /scratch/afw8937/.huggingface:/hf_cache \
    "$SIF" /bin/bash -c "
        export HF_HOME=/hf_cache
        vllm serve '$model_path' \
            --trust-remote-code \
            --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            --gpu-memory-utilization $GPU_UTIL \
            --max-model-len $MAX_MODEL_LEN \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-auto-tool-choice \
            --port $PORT
    " &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

echo "=== Waiting for vLLM to be ready ==="
for i in $(seq 1 90); do
    if curl -sf http://localhost:${PORT}/v1/models >/dev/null 2>&1; then
        echo "vLLM ready after ${i}0 seconds"
        break
    fi
    sleep 10
done

if ! curl -sf http://localhost:${PORT}/v1/models >/dev/null 2>&1; then
    echo "FATAL: vLLM did not come up within 15 min — aborting"
    exit 1
fi

source "$VENV/bin/activate"

TRAJ_DIR="runs/${dataset}/Qwen3-Embedding-8B/test300/${MODEL_NAME}/seed${seed}"
OUT_FILE="summaries/${MODEL_NAME}_test300.jsonl"
QUERIES_TSV="topics-qrels/${dataset}/queries_test300.tsv"

echo "=== Running summarize_trajectories.py ==="
echo "    traj-dir: $TRAJ_DIR"
echo "    queries : $QUERIES_TSV"
echo "    output  : $OUT_FILE"

python summarize_trajectories.py \
    --trajectory-dir "$TRAJ_DIR" \
    --queries-tsv "$QUERIES_TSV" \
    --output-file "$OUT_FILE" \
    --model "$model_path" \
    --model-url http://localhost:${PORT}/v1 \
    --num-threads 16 \
    --max-tokens 8192 \
    --verbose

echo "=== Done ==="
