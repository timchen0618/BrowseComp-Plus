#!/bin/bash
#SBATCH --job-name=rl_test_tier3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=400GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline/sbatch_outputs/rl_test_tier3_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hc3337@nyu.edu

# ---------------------------------------------------------------------------
# Tier 3 — 2-GPU full integration test (mirrors VALIDATION_2GPU.md)
#
# Layout:
#   GPU 0  main agent vLLM (Qwen3.5-0.8B, port 8010, util=0.4)
#   GPU 1  rollout daemon  (explorer + judge, port 8011, time-muxed)
#   GPU 0  GRPO learner    (Qwen3.5-0.8B + LoRA, 3 steps, accelerate 1-GPU)
#
# Validates:
#   - Daemon generates trajectories and fills buffer
#   - Learner runs 3 training steps with finite loss
#   - Checkpoint saved to rl/trl/tmp/val_2gpu_buffer/ckpt_latest/
#   - Final model saved to rl/trl/runs/val_2gpu/
# ---------------------------------------------------------------------------

PROJECT_DIR=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline
SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
CONFIG=${PROJECT_DIR}/rl/trl/config_2gpu_val.yaml
BUFFER_DIR=${PROJECT_DIR}/rl/trl/tmp/val_2gpu_buffer
RUNS_DIR=${PROJECT_DIR}/rl/trl/runs/val_2gpu
MAIN_AGENT_PORT=8010

mkdir -p ${PROJECT_DIR}/sbatch_outputs
rm -rf ${BUFFER_DIR}
mkdir -p ${BUFFER_DIR}
rm -rf ${RUNS_DIR}

echo "=== Tier 3: 2-GPU full integration test — $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"

# ---------------------------------------------------------------------------
# 1. Start main agent on GPU 0, port 8010 (persistent for the whole job)
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
            --gpu-memory-utilization 0.4
    " &
MAIN_AGENT_PID=$!
echo "Main agent PID: ${MAIN_AGENT_PID}"

# Wait up to 5 min for main agent
echo "Waiting for main agent on port ${MAIN_AGENT_PORT} ..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:${MAIN_AGENT_PORT}/health > /dev/null 2>&1; then
        echo "Main agent ready after $((i * 5))s"
        break
    fi
    sleep 5
done

# ---------------------------------------------------------------------------
# 2. Start rollout daemon on GPU 1, port 8011 (runs until learner exits)
# ---------------------------------------------------------------------------
echo "--- Starting rollout daemon (GPU 1, port 8011) ---"
singularity exec --nv \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        cd ${PROJECT_DIR}
        HF_TOKEN=${HF_TOKEN} \
        python rl/trl/rollout_daemon.py \
            --config ${CONFIG} \
            --gpu-id 1
    " &
DAEMON_PID=$!
echo "Rollout daemon PID: ${DAEMON_PID}"

# Give daemon time to load FAISS + start vLLM before starting learner
echo "Waiting 30s for daemon to begin filling buffer ..."
sleep 30

# ---------------------------------------------------------------------------
# 3. Start GRPO learner on GPU 0 (foreground — wait for it to finish)
# ---------------------------------------------------------------------------
echo "--- Starting GRPO learner (GPU 0, accelerate 1-process) ---"
singularity exec --nv \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        cd ${PROJECT_DIR}
        HF_TOKEN=${HF_TOKEN} \
        CUDA_VISIBLE_DEVICES=0 \
        accelerate launch \
            --config_file rl/trl/accelerate_1gpu.yaml \
            rl/trl/grpo_train.py \
            --config ${CONFIG}
    "
LEARNER_EXIT=$?
echo "Learner exit code: ${LEARNER_EXIT}"

# ---------------------------------------------------------------------------
# 4. Cleanup daemon + main agent
# ---------------------------------------------------------------------------
echo "Shutting down daemon and main agent ..."
kill ${DAEMON_PID} ${MAIN_AGENT_PID} 2>/dev/null
wait ${DAEMON_PID} ${MAIN_AGENT_PID} 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Validate artifacts
# ---------------------------------------------------------------------------
if [ ${LEARNER_EXIT} -ne 0 ]; then
    echo "TIER3_FAIL: learner exited with code ${LEARNER_EXIT}"
    exit 1
fi

if [ ! -d "${RUNS_DIR}" ]; then
    echo "TIER3_FAIL: final model dir not found: ${RUNS_DIR}"
    exit 1
fi

if [ ! -d "${BUFFER_DIR}/ckpt_latest" ]; then
    echo "TIER3_FAIL: checkpoint dir not found: ${BUFFER_DIR}/ckpt_latest"
    exit 1
fi

N_SAMPLES=$(ls ${BUFFER_DIR}/sample_*.jsonl 2>/dev/null | wc -l)
echo "Remaining buffer samples: ${N_SAMPLES}"

echo "TIER3_PASS: learner completed, model at ${RUNS_DIR}, checkpoint at ${BUFFER_DIR}/ckpt_latest"
echo "=== Tier 3 complete — $(date) ==="
exit 0
