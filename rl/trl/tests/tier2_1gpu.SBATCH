#!/bin/bash
#SBATCH --job-name=rl_test_tier2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=200GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline/sbatch_outputs/rl_test_tier2_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hc3337@nyu.edu

# ---------------------------------------------------------------------------
# Tier 2 — 1-GPU daemon-only integration test
#
# Layout (1 H200):
#   GPU 0  main agent vLLM  (Qwen3.5-0.8B, port 8010, util=0.25)
#   GPU 0  rollout daemon   (explorer + judge, port 8011, util=0.25, --max-iterations 1)
#
# Validates:
#   - Daemon loads FAISS index + embedding model
#   - Explorer vLLM starts and generates trajectories
#   - Main agent completes (or gracefully fails)
#   - Judge swaps in and scores answers
#   - ≥4 sample files appear in the buffer dir
#   - Daemon exits with code 0
# ---------------------------------------------------------------------------

PROJECT_DIR=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline
SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
CONFIG=${PROJECT_DIR}/rl/trl/config_2gpu_val.yaml
BUFFER_DIR=${PROJECT_DIR}/rl/trl/tmp/val_2gpu_buffer
MAIN_AGENT_PORT=8010

mkdir -p ${PROJECT_DIR}/sbatch_outputs
rm -rf ${BUFFER_DIR}
mkdir -p ${BUFFER_DIR}

echo "=== Tier 2: 1-GPU daemon test — $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"

# ---------------------------------------------------------------------------
# 1. Start main agent on GPU 0, port 8010
# ---------------------------------------------------------------------------
echo "--- Starting main agent (GPU 0, port ${MAIN_AGENT_PORT}) ---"
singularity exec --nv \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3.5-0.8B \
            --port ${MAIN_AGENT_PORT} \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.25
    " &
MAIN_AGENT_PID=$!
echo "Main agent PID: ${MAIN_AGENT_PID}"

# Wait up to 5 min for main agent health
echo "Waiting for main agent on port ${MAIN_AGENT_PORT} ..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:${MAIN_AGENT_PORT}/health > /dev/null 2>&1; then
        echo "Main agent ready after $((i * 5))s"
        break
    fi
    sleep 5
done

# ---------------------------------------------------------------------------
# 2. Run daemon with --max-iterations 1 (exits after 1 rollout iteration)
# ---------------------------------------------------------------------------
echo "--- Starting rollout daemon (GPU 0, port 8011, max-iterations=1) ---"
singularity exec --nv \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        cd ${PROJECT_DIR}
        HF_TOKEN=${HF_TOKEN} \
        python rl/trl/rollout_daemon.py \
            --config ${CONFIG} \
            --gpu-id 0 \
            --gpu-memory-utilization 0.25 \
            --max-iterations 1
    "
DAEMON_EXIT=$?
echo "Daemon exit code: ${DAEMON_EXIT}"

# ---------------------------------------------------------------------------
# 3. Cleanup main agent
# ---------------------------------------------------------------------------
kill ${MAIN_AGENT_PID} 2>/dev/null
wait ${MAIN_AGENT_PID} 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------
N_SAMPLES=$(ls ${BUFFER_DIR}/sample_*.jsonl 2>/dev/null | wc -l)
echo "Buffer samples found: ${N_SAMPLES}"

if [ ${DAEMON_EXIT} -ne 0 ]; then
    echo "TIER2_FAIL: daemon exited with code ${DAEMON_EXIT}"
    exit 1
fi

if [ "${N_SAMPLES}" -lt 4 ]; then
    echo "TIER2_FAIL: expected >=4 buffer samples, got ${N_SAMPLES}"
    exit 1
fi

echo "TIER2_PASS: daemon exited cleanly with ${N_SAMPLES} buffer samples"
echo "=== Tier 2 complete — $(date) ==="
exit 0
