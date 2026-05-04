#!/bin/bash
# End-to-end Axolotl SFT for search-trajectory following.
#
#   1. Convert a selected-tool-calls JSONL into Axolotl messages JSONL
#   2. Run Axolotl's preprocess step (tokenize + cache)
#   3. Launch multi-GPU training via accelerate
#
# Required environment variables:
#   INPUT              path to the selected-tool-calls JSONL
#                      (records with {source_file, excerpt, ...}).
#   TRAJECTORY_FOLDER  folder holding the source trajectory JSON files
#                      referenced by each record's `source_file` field.
#
# Optional env vars:
#   RUN_NAME   identifier used to derive DATA_DIR, PREPARED_DIR, output_dir,
#              and wandb_name (default: data_qwen)
#   DATA_DIR   where prepare_dataset.py writes train.jsonl / val.jsonl
#              (default: sft/axolotl/data/raw/${RUN_NAME})
#   CONFIG     Axolotl YAML config
#              (default: sft/axolotl/qwen3.5_4b_search_sft.yaml)
#   SPLIT      prepare_dataset --split (default: bcp-train680-test150)
#   VAL_SIZE   only for --split random: val fraction (default: 0.1)
#   SEED       only for --split random: shuffle seed (default: 42)
#
# Example:
#   INPUT=selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
#   TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
#   RUN_NAME=my_run \
#       bash sft/axolotl/run_axolotl.sh

set -euo pipefail

export AXOLOTL_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_FSDP=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${INPUT:?set INPUT to a selected-tool-calls JSONL}"
: "${TRAJECTORY_FOLDER:?set TRAJECTORY_FOLDER to the folder holding source_file trajectories}"

RUN_NAME="${RUN_NAME:-data_qwen}"
DATA_DIR="${DATA_DIR:-sft/axolotl/data/raw/${RUN_NAME}}"
PREPARED_DIR="${DATA_DIR/\/raw\//\/prepared\/}"
# CONFIG="${CONFIG:-sft/axolotl/qwen3_30b_a3b_search_sft.yaml}"
CONFIG="${CONFIG:-sft/axolotl/qwen3.5_4b_search_sft.yaml}"
SPLIT="${SPLIT:-bcp-train680-test150}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"

if [[ "${CONFIG}" == *30b* ]]; then
    export FSDP_TRANSFORMER_CLS_TO_WRAP=Qwen3MoeDecoderLayer
else
    export FSDP_TRANSFORMER_CLS_TO_WRAP=Qwen3_5DecoderLayer
fi

export RUN_NAME DATA_DIR PREPARED_DIR

RENDERED_CONFIG="/scratch/hc3337/tmp/axolotl_config_$$.yaml"
envsubst '${DATA_DIR} ${PREPARED_DIR} ${RUN_NAME}' < "${CONFIG}" > "${RENDERED_CONFIG}"
trap "rm -f ${RENDERED_CONFIG}" EXIT

echo "[1/3] Converting excerpts: ${INPUT} -> ${DATA_DIR}"
python sft/axolotl/prepare_dataset.py \
    --input "${INPUT}" \
    --trajectory-folder "${TRAJECTORY_FOLDER}" \
    --output-dir "${DATA_DIR}" \
    --split "${SPLIT}" \
    --val-size "${VAL_SIZE}" \
    --seed "${SEED}" \
    --template "qwen"

echo "[2/3] Axolotl preprocess (tokenize + cache)"
axolotl preprocess "${RENDERED_CONFIG}"

echo "[3/3] Axolotl train"
# Debug: PEFT FSDP wrap class resolution (NDJSON -> .cursor/debug-3e9a99.log). No-op overhead: meta load only.
python sft/axolotl/debug_peft_fsdp_wrap_probe.py "${RENDERED_CONFIG}"
# Pass through any extra CLI args (e.g. --num_epochs=5) to axolotl.cli.train
accelerate launch -m axolotl.cli.train "${RENDERED_CONFIG}" "$@"



# RUN_NAME=my_run INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 bash sft/axolotl/run_axolotl.sh 