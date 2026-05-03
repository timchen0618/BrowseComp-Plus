#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=glm-4.7-flash_test150_budget5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afw8937@nyu.edu
#SBATCH --output=/scratch/afw8937/browsecomp-plus/logs/glm-4.7-flash_test150_budget5_%j.out
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=h100_tandon
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2

# Round-1 GLM-4.7-Flash on test150 with --search-budget 5 (≤5 tool calls).
# Output: runs/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/budget5_seed0/
# This becomes the explorer trajectory for the self-prompted Task 2 round-2.
# Submit with: sbatch sbatch/run_bcp_test150_glm_budget5.SBATCH

set -euo pipefail

MODEL_NAME="glm-4.7-flash"
model_path="zai-org/GLM-4.7-Flash"
TENSOR_PARALLEL_SIZE=2
EMBED_RESERVE_GB=25
MAX_MODEL_LEN=131072

REPO_DIR=/home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus
VENV=/scratch/afw8937/efficient-search-agents/venv
INDEX_PATH="/scratch/afw8937/efficient-search-agents/indexes/qwen3-embedding-8b/corpus.shard*.pkl"
EMBED_MODEL_NAME="Qwen/Qwen3-Embedding-8B"

SIF=/scratch/afw8937/singularity_images/vllm_nightly.sif
OVERLAY=/scratch/afw8937/singularity_images/vllm_nightly_overlay.img
SINGULARITY=/share/apps/apptainer/bin/singularity

dataset=bcp
seed=0
split=test150
run_name=budget5_seed${seed}

cd "$REPO_DIR"
mkdir -p /scratch/afw8937/tmp
export TMPDIR=/scratch/afw8937/tmp

test -f "$SIF"     || { echo "FATAL: SIF missing"; exit 1; }
test -f "$OVERLAY" || { echo "FATAL: overlay missing"; exit 1; }

PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "=== Using vLLM port $PORT ==="

GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_TOTAL_GB=$((GPU_TOTAL_MIB / 1024))
GPU_UTIL=$(python3 -c "print(round(($GPU_TOTAL_GB - $EMBED_RESERVE_GB) / $GPU_TOTAL_GB, 2))")
echo "    GPU_UTIL=$GPU_UTIL (${GPU_TOTAL_GB}GB - ${EMBED_RESERVE_GB}GB reserve)"

echo "=== Starting vLLM (GLM-4.7-Flash, TP=$TENSOR_PARALLEL_SIZE) ==="
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

for i in $(seq 1 90); do
    if curl -sf http://localhost:${PORT}/v1/models >/dev/null 2>&1; then
        echo "vLLM ready after ${i}0 seconds"; break
    fi
    sleep 10
done
curl -sf http://localhost:${PORT}/v1/models >/dev/null || { echo "FATAL: vLLM not up"; exit 1; }

module load openjdk/21.0.3
source "$VENV/bin/activate"

OUT_BASE="runs/${dataset}/Qwen3-Embedding-8B/${split}/${MODEL_NAME}"
output_dir="${OUT_BASE}/${run_name}"

echo "=== Running glm_client.py (budget=5 round-1, will become Task 2 round-1 input) ==="
echo "    output: $output_dir"

python search_agent/glm_client.py \
    --output-dir "$output_dir" \
    --searcher-type faiss \
    --index-path "$INDEX_PATH" \
    --model-name "$EMBED_MODEL_NAME" \
    --normalize \
    --num-threads 16 \
    --query "topics-qrels/${dataset}/queries_test150.tsv" \
    --model "$model_path" \
    --model-url http://localhost:${PORT}/v1 \
    --max-model-len $MAX_MODEL_LEN \
    --search-budget 5 \
    --save-raw-messages

echo "=== Done ==="
