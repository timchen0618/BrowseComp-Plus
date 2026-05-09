#!/bin/bash
#SBATCH --job-name=rl_test_tier1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mem=16GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline/sbatch_outputs/rl_test_tier1_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hc3337@nyu.edu

# ---------------------------------------------------------------------------
# Tier 1 — CPU pipeline tests
#
# Runs: test_pipeline_cpu.py
#   - _explorer_system_prompt() date freshness
#   - config_2gpu_val.yaml correctness
#   - compute_log_probs shape regression (the [B,T]×[B,T-1] crash fix)
#   - Full tokenize→collate→log_probs→grpo_loss CPU end-to-end
# No GPU required. ~1 minute.
# ---------------------------------------------------------------------------

PROJECT_DIR=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline
SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3

mkdir -p ${PROJECT_DIR}/sbatch_outputs

echo "=== Tier 1: CPU pipeline tests — $(date) ==="
echo "Node: $(hostname)"

singularity exec \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        cd ${PROJECT_DIR}
        python -m pytest rl/trl/tests/test_pipeline_cpu.py -v --tb=short 2>&1
    "
EXIT_CODE=$?

echo "=== Tier 1 exit code: ${EXIT_CODE} — $(date) ==="
exit ${EXIT_CODE}
