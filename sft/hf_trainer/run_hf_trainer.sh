#!/bin/bash
# Hugging Face Trainer SFT for search-trajectory following (Axolotl-compatible JSONL).
#
#   1. Convert a selected-tool-calls JSONL into messages JSONL (reuses axolotl/prepare_dataset.py)
#   2. Launch training via accelerate + sft/hf_trainer/train_sft.py
#
# Required environment variables:
#   INPUT              path to the selected-tool-calls JSONL
#                      (records with {source_file, excerpt, ...}).
#   TRAJECTORY_FOLDER  folder holding the source trajectory JSON files
#                      referenced by each record's `source_file` field.
#
# Optional env vars:
#   DATA_DIR   where prepare_dataset.py writes train.jsonl / val.jsonl
#              (default: sft/hf_trainer/data)
#   SPLIT      prepare_dataset --split (default: bcp-train680-test150)
#   VAL_SIZE   only for --split random (default: 0.1)
#   SEED       only for --split random (default: 42)
#   TEMPLATE   prepare_dataset --template (default: qwen)
#
# Any additional arguments are forwarded to train_sft.py (after --).
#
# Example:
#   INPUT=selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
#   TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
#       bash sft/hf_trainer/run_hf_trainer.sh -- --output_dir sft/checkpoints/my-hf-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${INPUT:?set INPUT to a selected-tool-calls JSONL}"
: "${TRAJECTORY_FOLDER:?set TRAJECTORY_FOLDER to the folder holding source_file trajectories}"

DATA_DIR="${DATA_DIR:-sft/hf_trainer/data}"
SPLIT="${SPLIT:-bcp-train680-test150}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"
TEMPLATE="${TEMPLATE:-qwen}"

echo "[1/2] Converting excerpts: ${INPUT} -> ${DATA_DIR}"
python sft/axolotl/prepare_dataset.py \
    --input "${INPUT}" \
    --trajectory-folder "${TRAJECTORY_FOLDER}" \
    --output-dir "${DATA_DIR}" \
    --split "${SPLIT}" \
    --val-size "${VAL_SIZE}" \
    --seed "${SEED}" \
    --template "${TEMPLATE}"

TRAIN_FILE="${DATA_DIR}/train.jsonl"
EVAL_FILE="${DATA_DIR}/val.jsonl"

EXTRA=()
if [[ -f "${EVAL_FILE}" ]]; then
  EXTRA+=(--eval_file "${EVAL_FILE}")
fi

echo "[2/2] HF Trainer (accelerate launch)"
# Allow `bash run_hf_trainer.sh -- --output_dir ...` (strip a lone `--`)
if [[ "${1:-}" == "--" ]]; then
  shift
fi
exec accelerate launch sft/hf_trainer/train_sft.py \
    --train_file "${TRAIN_FILE}" \
    "${EXTRA[@]}" \
    "$@"
