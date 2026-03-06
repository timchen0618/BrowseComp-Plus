#!/bin/bash
#SBATCH -A torch_pr_62_tandon_priority
#SBATCH --job-name=gar-pilot
#SBATCH --output=/scratch/afw8937/logs/gar-pilot-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
# GPU count set via sbatch --gres override (baseline=1, gar=2)
#SBATCH --gres=gpu:h100:2
#SBATCH --time=24:00:00
#SBATCH --partition=h100_tandon

set -e

# ---- Configuration ----
CONDITION="${CONDITION:-baseline}"  # baseline / gar_no_gate / gar_gated
GAR_ALPHA="${GAR_ALPHA:-0.7}"
AGENT_PORT=6008
GEN_PORT=6009
AGENT_MODEL="hkust-nlp/WebExplorer-8B"
GEN_MODEL="Qwen/Qwen3-8B"

echo "================================================"
echo "GAR Pilot Experiment"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Condition: ${CONDITION}"
echo "Alpha: ${GAR_ALPHA}"
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

# Load API keys
if [ -f /home/afw8937/projects/efficient-search-agents/.env ]; then
    set -a
    source /home/afw8937/projects/efficient-search-agents/.env
    set +a
fi

# ---- Output Directory ----
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_DIR="/scratch/afw8937/efficient-search-agents/runs/gar_pilot/${CONDITION}${OUTPUT_SUFFIX}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p /scratch/afw8937/logs

PILOT_TSV="${PILOT_TSV:-/home/afw8937/projects/efficient-search-agents/hung-ting-browse-comp/BrowseComp-Plus/experiments/gar/pilot_100.tsv}"

# ---- Start vLLM Servers ----
echo "Starting agent vLLM server on GPU 0, port ${AGENT_PORT}..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -m vllm.entrypoints.openai.api_server \
    --model "${AGENT_MODEL}" \
    --port ${AGENT_PORT} \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    &> /scratch/afw8937/logs/gar-agent-vllm-${SLURM_JOB_ID}.log &
AGENT_PID=$!

if [ "${CONDITION}" != "baseline" ]; then
    echo "Starting generation vLLM server on GPU 1, port ${GEN_PORT}..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "${GEN_MODEL}" \
        --port ${GEN_PORT} \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.45 \
        --trust-remote-code \
        &> /scratch/afw8937/logs/gar-gen-vllm-${SLURM_JOB_ID}.log &
    GEN_PID=$!
fi

# ---- Health Check ----
echo "Waiting for agent vLLM server..."
for i in $(seq 1 300); do
    if curl -s http://127.0.0.1:${AGENT_PORT}/health > /dev/null 2>&1; then
        echo "Agent server ready after ${i}s"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "ERROR: Agent server did not start within 300s"
        kill $AGENT_PID 2>/dev/null
        [ -n "${GEN_PID}" ] && kill $GEN_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

if [ "${CONDITION}" != "baseline" ]; then
    echo "Waiting for generation vLLM server..."
    for i in $(seq 1 300); do
        if curl -s http://127.0.0.1:${GEN_PORT}/health > /dev/null 2>&1; then
            echo "Generation server ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "ERROR: Generation server did not start within 300s"
            kill $AGENT_PID 2>/dev/null
            kill $GEN_PID 2>/dev/null
            exit 1
        fi
        sleep 1
    done
fi

# ---- Build CLI Args ----
COMMON_ARGS=(
    --model "${AGENT_MODEL}"
    --port ${AGENT_PORT}
    --query "${PILOT_TSV}"
    --output-dir "${OUTPUT_DIR}"
    --searcher-type faiss
    --index-path "/scratch/afw8937/efficient-search-agents/indexes/qwen3-embedding-8b/corpus.shard*_of_4.pkl"
    --model-name "Qwen/Qwen3-Embedding-8B"
    --normalize
    --k 5
    --snippet-max-tokens 512
    --num-threads 1
    --store-raw
    --attn_implementation sdpa
)

GAR_ARGS=()
if [ "${CONDITION}" == "gar_no_gate" ]; then
    GAR_ARGS=(
        --gar-mode query_expansion
        --gar-alpha "${GAR_ALPHA}"
        --gar-generation-port ${GEN_PORT}
        --gar-generation-model "${GEN_MODEL}"
        --generation-port ${GEN_PORT}
        --generation-model "${GEN_MODEL}"
    )
elif [ "${CONDITION}" == "gar_gated" ]; then
    GAR_ARGS=(
        --gar-mode query_expansion_gated
        --gar-alpha "${GAR_ALPHA}"
        --gar-generation-port ${GEN_PORT}
        --gar-generation-model "${GEN_MODEL}"
        --generation-port ${GEN_PORT}
        --generation-model "${GEN_MODEL}"
    )
fi

# ---- Run Agent ----
echo "Running agent with condition: ${CONDITION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "CLI args: ${COMMON_ARGS[*]} ${GAR_ARGS[*]}"

CUDA_VISIBLE_DEVICES=1 $PYTHON search_agent/tongyi_client.py \
    "${COMMON_ARGS[@]}" \
    "${GAR_ARGS[@]}"

echo "================================================"
echo "Experiment complete!"
echo "Condition: ${CONDITION}"
echo "Output: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "================================================"

# ---- Cleanup ----
kill $AGENT_PID 2>/dev/null
[ -n "${GEN_PID}" ] && kill $GEN_PID 2>/dev/null
wait
