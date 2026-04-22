#!/bin/bash
# Run the auto-pipeline.
#
# From the login node: starts locally AND submits a SLURM backup job.
# The SLURM job watches auto_pipeline.pid; when the login-node process dies
# (e.g. login-node reboot) it takes over with --resume.
#
# Usage (login node):
#   bash auto_pipeline.sh --submit
#   bash auto_pipeline.sh --submit --resume
#   tail -f auto_pipeline.log
#   cat auto_pipeline_state.json | jq .
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

PID_FILE="auto_pipeline.pid"

# ── Inside SLURM: watch the PID file, take over when the process dies ─────────
# Use a dedicated marker so this branch only fires when launched BY the SBATCH
# template below, not when a user runs the script inside an interactive SLURM job.
if [ "${_AUTO_PIPELINE_BACKUP:-}" = "1" ]; then
    echo "SLURM job ${SLURM_JOB_ID} started — monitoring login-node process..."
    while true; do
        # If pipeline finished cleanly, nothing to do
        if [ -f auto_pipeline_state.json ]; then
            phase=$(python3.12 -c \
                "import json; print(json.load(open('auto_pipeline_state.json')).get('phase',''))" \
                2>/dev/null || echo "")
            if [ "$phase" = "done" ]; then
                echo "Pipeline already done; SLURM backup exiting."
                exit 0
            fi
        fi
        # If login-node process is still alive, keep waiting
        if [ -f "$PID_FILE" ]; then
            pid=$(cat "$PID_FILE")
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                sleep 60
                continue
            fi
        fi
        break
    done
    echo "Login-node process gone — taking over with --resume."
    rm -f "$PID_FILE"
    exec python3.12 auto_pipeline.py --resume "$@"
fi

# ── Login node: start locally first, then submit SLURM backup ─────────────────
mkdir -p sbatch_outputs

# Singleton guard — refuse to start if a live instance already holds the PID file.
if [ -f "$PID_FILE" ]; then
    existing_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Error: auto_pipeline already running (PID $existing_pid)."
        echo "Kill it first or use '--resume' to restart a dead instance."
        exit 1
    fi
fi

# Start local process first so its PID is in the lock file before SLURM job begins
nohup python3.12 auto_pipeline.py "$@" >> auto_pipeline.log 2>&1 &
LOCAL_PID=$!
echo "$LOCAL_PID" > "$PID_FILE"
echo "Local PID  : ${LOCAL_PID}"

PROJECT_DIR="$(pwd)"
ARGS_QUOTED=$(printf '%q ' "$@")

SBATCH_SCRIPT=$(mktemp /tmp/auto_pipeline_XXXXXX.sbatch)
cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=auto_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=${PROJECT_DIR}/sbatch_outputs/auto_pipeline_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hc3337@nyu.edu

cd ${PROJECT_DIR}
export _AUTO_PIPELINE_BACKUP=1
exec bash auto_pipeline.sh ${ARGS_QUOTED}
SBATCH_EOF

sbatch_result=$(sbatch "$SBATCH_SCRIPT")
rm -f "$SBATCH_SCRIPT"
echo "$sbatch_result  (backup; takes over if this login session dies)"
echo "Follow logs: tail -f auto_pipeline.log"
echo "Check state: cat auto_pipeline_state.json | jq ."
