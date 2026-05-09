#!/bin/bash
#SBATCH --job-name=rl_test_tier0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --mem=16GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline/sbatch_outputs/rl_test_tier0_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hc3337@nyu.edu

# ---------------------------------------------------------------------------
# Tier 0 — CPU unit tests (existing pytest suite)
#
# Runs: test_smoke.py, test_buffer.py, test_reward.py, test_rollout_worker.py
# No GPU required. ~1-2 minutes.
# ---------------------------------------------------------------------------

PROJECT_DIR=/scratch/hc3337/projects/BrowseComp-Plus/.claude/worktrees/rl-test-pipeline
SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3

mkdir -p ${PROJECT_DIR}/sbatch_outputs

echo "=== Tier 0: CPU unit tests — $(date) ==="
echo "Node: $(hostname)"

singularity exec \
    --overlay ${OVERLAY_FILE}:ro \
    ${SINGULARITY_IMAGE} /bin/bash -c "
        source /ext3/env.sh
        cd ${PROJECT_DIR}
        python -m pytest rl/trl/tests/test_smoke.py \
                         rl/trl/tests/test_buffer.py \
                         rl/trl/tests/test_reward.py \
                         rl/trl/tests/test_rollout_worker.py \
                         -v --tb=short 2>&1
    "
EXIT_CODE=$?

echo "=== Tier 0 exit code: ${EXIT_CODE} — $(date) ==="
exit ${EXIT_CODE}
