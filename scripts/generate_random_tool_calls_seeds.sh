#!/usr/bin/env bash
# Generate N random_tool_calls JSONLs (one per seed) for a given (model, dataset, split).
#
# Usage:
#   scripts/generate_random_tool_calls_seeds.sh <model> <traj_dir> <input_jsonl> <out_prefix> [seeds...]
#
# Args:
#   model         model name (used only for log lines)
#   traj_dir      directory of baseline trajectory *.json files
#   input_jsonl   input JSONL with {query_id, source_file} per row. If it does not
#                 exist, we synthesize it from traj_dir via build_random_tool_calls_input.py
#   out_prefix    output JSONL prefix; per seed we write "<prefix>_seed<N>.jsonl"
#   seeds         space-separated seed list (default: 42 43 44 45)
#
# Example (BCP, GLM, 4 seeds reusing existing seed42 jsonl as the input map):
#   scripts/generate_random_tool_calls_seeds.sh \
#       glm \
#       runs/bcp/Qwen3-Embedding-8B/full/glm-4.7-flash/seed0 \
#       selected_tool_calls/glm_random_tools_calls.jsonl \
#       selected_tool_calls/glm_random_tools_calls \
#       42 43 44 45
set -euo pipefail

if [ $# -lt 4 ]; then
    sed -n '/^# Usage:/,/^set -euo/p' "$0" | head -n 25
    exit 1
fi

MODEL=$1
TRAJ_DIR=$2
INPUT_JSONL=$3
OUT_PREFIX=$4
shift 4
SEEDS=("${@:-42 43 44 45}")
# If no positional seeds, fall back to default array. The braces above produce a
# single string when no args remain, so re-split:
if [ ${#SEEDS[@]} -eq 1 ] && [ "${SEEDS[0]}" = "42 43 44 45" ]; then
    SEEDS=(42 43 44 45)
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [ ! -d "$TRAJ_DIR" ]; then
    echo "FATAL: trajectory dir missing: $TRAJ_DIR" >&2
    exit 1
fi

if [ ! -f "$INPUT_JSONL" ]; then
    echo "[$MODEL] input jsonl missing — synthesizing from $TRAJ_DIR"
    python scripts/build_random_tool_calls_input.py \
        --trajectory-dir "$TRAJ_DIR" \
        --output "$INPUT_JSONL"
fi

for seed in "${SEEDS[@]}"; do
    out="${OUT_PREFIX}_seed${seed}.jsonl"
    echo "[$MODEL seed=$seed] -> $out"
    python src_select_tool_calls/random_select_tool_calls.py \
        --input-jsonl "$INPUT_JSONL" \
        --trajectory-dir "$TRAJ_DIR" \
        --output-jsonl "$out" \
        --k 5 --seed "$seed" --format glm --force
done

echo "Done. Generated ${#SEEDS[@]} files for $MODEL."
