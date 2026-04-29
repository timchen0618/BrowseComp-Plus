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
#              (default: sft/axolotl/data/raw/data_qwen)
#   CONFIG     Axolotl YAML config
#              (default: sft/axolotl/qwen3.5_4b_search_sft.yaml)
#   SPLIT      prepare_dataset --split (default: bcp-train680-test150)
#   VAL_SIZE   only for --split random: val fraction (default: 0.1)
#   SEED       only for --split random: shuffle seed (default: 42)
#
# Example:
#   INPUT=selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
#   TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
#       bash sft/axolotl/run_axolotl.sh

set -euo pipefail

export AXOLOTL_DO_NOT_TRACK=1
export FSDP_TRANSFORMER_CLS_TO_WRAP=Qwen3_5DecoderLayer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${INPUT:?set INPUT to a selected-tool-calls JSONL}"
: "${TRAJECTORY_FOLDER:?set TRAJECTORY_FOLDER to the folder holding source_file trajectories}"

DATA_DIR="${DATA_DIR:-sft/axolotl/data/raw/data_qwen}"
# CONFIG="${CONFIG:-sft/axolotl/qwen3_30b_a3b_search_sft.yaml}"
CONFIG="${CONFIG:-sft/axolotl/qwen3.5_4b_search_sft.yaml}"
SPLIT="${SPLIT:-bcp-train680-test150}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"

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
axolotl preprocess "${CONFIG}"

echo "[3/3] Axolotl train"
# Debug: PEFT FSDP wrap class resolution (NDJSON -> .cursor/debug-3e9a99.log). No-op overhead: meta load only.
python sft/axolotl/debug_peft_fsdp_wrap_probe.py "${CONFIG}"
# Pass through any extra CLI args (e.g. --num_epochs=5) to axolotl.cli.train
accelerate launch -m axolotl.cli.train "${CONFIG}" "$@"



# [2026-04-18 15:22:47,226] [WARNING] [axolotl.prompt_strategies.chat_template] Last turn is not trainable, skipping having to find the turn indices. This may cause         
#   incorrect last EOT/EOS token to be unmasked.This is likely a dataset design issue. Please ensure last turn is trainable.                                                                        
                                                                                                                                                                                                  
# ● This warning means the last message in some training examples has its loss masked out (not trained on), when it should be trainable.                                                            
                                                                                                                                                                                                
#   In chat-template SFT, axolotl only computes loss on "trainable" turns — typically assistant turns. The warning fires when the final turn in a conversation is a user/tool turn or otherwise     
#   excluded from training.
                                                                                                                                                                                                  
#   Why it matters: The model needs to learn to produce the final response (including the EOT/EOS token). If the last turn is masked, the model never learns when to stop — and the EOS token       
#   masking may be incorrect.
                                                                                                                                                                                                  
#   Likely cause in your case: Some examples in your train.jsonl/val.jsonl end with a tool result or user message rather than an assistant message. Check prepare_dataset.py — the last message in  
#   each messages list should always be role: assistant.



# INPUT=selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4  bash sft/axolotl/run_axolotl.sh 