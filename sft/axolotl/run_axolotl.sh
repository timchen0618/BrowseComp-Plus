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
#   DATA_DIR   where prepare_dataset.py writes train.jsonl / val.jsonl
#              (default: sft/axolotl/data)
#   CONFIG     Axolotl YAML config
#              (default: sft/axolotl/qwen3_30b_a3b_search_sft.yaml)
#   VAL_SIZE   val split fraction (default: 0.1)
#   SEED       shuffle seed (default: 42)
#
# Example:
#   INPUT=selected_tool_calls/selected_tool_calls.jsonl \
#   TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
#       bash sft/axolotl/run_axolotl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${INPUT:?set INPUT to a selected-tool-calls JSONL}"
: "${TRAJECTORY_FOLDER:?set TRAJECTORY_FOLDER to the folder holding source_file trajectories}"

DATA_DIR="${DATA_DIR:-sft/axolotl/data}"
CONFIG="${CONFIG:-sft/axolotl/qwen3_30b_a3b_search_sft.yaml}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"

echo "[1/3] Converting excerpts: ${INPUT} -> ${DATA_DIR}"
python sft/axolotl/prepare_dataset.py \
    --input "${INPUT}" \
    --trajectory-folder "${TRAJECTORY_FOLDER}" \
    --output-dir "${DATA_DIR}" \
    --val-size "${VAL_SIZE}" \
    --seed "${SEED}"

echo "[2/3] Axolotl preprocess (tokenize + cache)"
axolotl preprocess "${CONFIG}"

echo "[3/3] Axolotl train"
# Pass through any extra CLI args (e.g. --num_epochs=5) to axolotl.cli.train
accelerate launch -m axolotl.cli.train "${CONFIG}" "$@"
