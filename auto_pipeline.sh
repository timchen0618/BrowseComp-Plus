#!/bin/bash
# Run the auto-pipeline entirely inside SLURM (no login-node process).
#
# From the login node:
#   bash auto_pipeline.sh --submit           # fresh start
#   bash auto_pipeline.sh --submit --resume  # continue from saved state
#   tail -f auto_pipeline.log
#   cat auto_pipeline_state.json | jq .
#
# The SLURM job chains a new job via --resume when approaching its time limit,
# so the pipeline runs uninterrupted across multiple 2h45m slots.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

mkdir -p sbatch_outputs

PROJECT_DIR="$(pwd)"
ARGS_QUOTED=$(printf '%q ' "$@")

TMPDIR="/scratch/hc3337/tmp"
echo ${ARGS_QUOTED}
echo "hey"
echo $TMPDIR
# ── Inside SLURM job ───────────────────────────────────────────────────────────
if [ "${_AUTO_PIPELINE_SLURM:-}" = "1" ]; then
    echo "SLURM job ${SLURM_JOB_ID} started at $(date)"

    PIPELINE_ARGS=("$@")
    CHILD_PID=""

    _pipeline_phase() {
        [ -f auto_pipeline_state.json ] || { echo ""; return; }
        python3.12 -c \
            "import json; print(json.load(open('auto_pipeline_state.json')).get('phase',''))" \
            2>/dev/null || echo ""
    }

    _pipeline_done() {
        local p; p=$(_pipeline_phase)
        [ "$p" = "done" ] || [ "$p" = "stuck" ]
    }

    _submit_next() {
        if _pipeline_done; then
            echo "Pipeline done/stuck; not chaining."
            return 0
        fi
        echo "Submitting chained job at $(date)..."
        local tmp args_quoted filtered_args=()
        for arg in "${PIPELINE_ARGS[@]}"; do
            [ "$arg" != "--resume" ] && filtered_args+=("$arg")
        done
        args_quoted=$(printf '%q ' "${filtered_args[@]}")
        tmp=$(mktemp $TMPDIR/auto_pipeline_XXXXXX.sbatch)
        cat > "$tmp" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=auto_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:45:00
#SBATCH --mem=8GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=${PROJECT_DIR}/sbatch_outputs/auto_pipeline_%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hc3337@nyu.edu

cd ${PROJECT_DIR}
export _AUTO_PIPELINE_SLURM=1
# Always resume when chaining — state file holds progress
exec bash auto_pipeline.sh --resume ${args_quoted}
SBATCH_EOF
        sbatch "$tmp"
        rm -f "$tmp"
    }

    # USR1 = SLURM pre-termination warning (300s before time limit)
    # TERM = SLURM kill at time limit
    _CHAIN_SUBMITTED=0

    _submit_next_once() {
        if [ "$_CHAIN_SUBMITTED" = "1" ]; then
            echo "Chain already submitted; skipping duplicate."
            return 0
        fi
        _submit_next || echo "WARNING: chain submission failed"
        _CHAIN_SUBMITTED=1
    }

    _on_signal() {
        echo "Signal received at $(date); chaining next job..."
        _submit_next_once
        [ -n "$CHILD_PID" ] && kill "$CHILD_PID" 2>/dev/null
        wait "$CHILD_PID" 2>/dev/null || true
        exit 0
    }
    trap '_on_signal' USR1 TERM

    # Run the pipeline (not exec so we can chain after normal exit)
    python3.12 auto_pipeline.py "${PIPELINE_ARGS[@]}" &
    CHILD_PID=$!
    wait "$CHILD_PID" || true
    CHILD_PID=""

    # Python exited normally — chain if not done
    _submit_next_once || true
    exit 0
fi

# ── Login node: just submit the initial SLURM job ─────────────────────────────
TMP=$(mktemp $TMPDIR/auto_pipeline_XXXXXX.sbatch)
cat > "$TMP" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=auto_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:45:00
#SBATCH --mem=8GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=${PROJECT_DIR}/sbatch_outputs/auto_pipeline_%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hc3337@nyu.edu

cd ${PROJECT_DIR}
export _AUTO_PIPELINE_SLURM=1
exec bash auto_pipeline.sh ${ARGS_QUOTED}
SBATCH_EOF

sbatch_result=$(sbatch "$TMP")
rm -f "$TMP"
echo "$sbatch_result"
echo "Logs : tail -f ${PROJECT_DIR}/sbatch_outputs/auto_pipeline_<jobid>.out"
echo "State: cat auto_pipeline_state.json | jq ."
