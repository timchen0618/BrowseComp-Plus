#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=minimax-m2.5_test300_traj_orig
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afw8937@nyu.edu
#SBATCH --output=/scratch/afw8937/browsecomp-plus/logs/minimax-m2.5_test300_traj_orig_%j.out
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:2

# GLM-4.7-Flash test300 round-2 with full baseline trajectory prepended.
# SHARD env var: A or B. Depends on test300 baseline being done for those qids.

set -euo pipefail

if [[ -z "${SHARD:-}" ]]; then echo "FATAL: must set SHARD env var (A or B)"; exit 1; fi
if [[ "$SHARD" != "A" && "$SHARD" != "B" ]]; then echo "FATAL: SHARD must be A or B"; exit 1; fi

MODEL_NAME="minimax-m2.5"
model_path="MiniMaxAI/MiniMax-M2.5"
TENSOR_PARALLEL_SIZE=2
EMBED_RESERVE_GB=5
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=16

REPO_DIR=/home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus
VENV=/scratch/afw8937/efficient-search-agents/venv
INDEX_PATH="/scratch/afw8937/efficient-search-agents/indexes/qwen3-embedding-8b/corpus.shard*.pkl"
EMBED_MODEL_NAME="Qwen/Qwen3-Embedding-8B"

SIF=/scratch/afw8937/singularity_images/vllm_nightly.sif
OVERLAY=/scratch/afw8937/singularity_images/vllm_nightly_overlay.img
SINGULARITY=/share/apps/apptainer/bin/singularity

dataset=bcp
split=test300
run_name=traj_orig_ext_seed0

cd "$REPO_DIR"
mkdir -p /scratch/afw8937/tmp
export TMPDIR=/scratch/afw8937/tmp

QUERY_FILE="topics-qrels/${dataset}/queries_test300_shard${SHARD}.tsv"
TRAJ_DIR="runs/${dataset}/Qwen3-Embedding-8B/${split}/${MODEL_NAME}/seed0"
test -f "$QUERY_FILE" || { echo "FATAL: query file missing: $QUERY_FILE"; exit 1; }
test -d "$TRAJ_DIR"   || { echo "FATAL: baseline traj dir missing: $TRAJ_DIR"; exit 1; }
test -f "$SIF"     || { echo "FATAL: SIF missing"; exit 1; }
test -f "$OVERLAY" || { echo "FATAL: overlay missing"; exit 1; }

PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "=== shard=$SHARD query=$QUERY_FILE traj=$TRAJ_DIR ($(ls $TRAJ_DIR | wc -l) files) ==="

GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_TOTAL_GB=$((GPU_TOTAL_MIB / 1024))
GPU_UTIL=$(python3 -c "print(round(($GPU_TOTAL_GB - $EMBED_RESERVE_GB) / $GPU_TOTAL_GB, 2))")

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
            --max-num-seqs $MAX_NUM_SEQS \
            --reasoning-parser minimax_m2_append_think \
            --tool-call-parser minimax_m2 \
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

export CUDA_VISIBLE_DEVICES=-1

OUT_BASE="runs/${dataset}/Qwen3-Embedding-8B/${split}/${MODEL_NAME}"
output_dir="${OUT_BASE}/${run_name}"

python search_agent/minimax_client.py \
    --output-dir "$output_dir" \
    --searcher-type faiss \
    --index-path "$INDEX_PATH" \
    --model-name "$EMBED_MODEL_NAME" \
    --normalize \
    --num-threads 16 \
    --query "$QUERY_FILE" \
    --model "$model_path" \
    --model-url http://localhost:${PORT}/v1 \
    --max-model-len $MAX_MODEL_LEN \
            --max-num-seqs $MAX_NUM_SEQS \
    --planning-trigger traj_orig_ext \
    --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY \
    --trajectory-dir "$TRAJ_DIR" \
    --max-chars 500000 \
    --reasoning-max-chars 3000 \
    --tool-output-max-chars 5000 \
    --save-raw-messages

echo "=== Done ==="
