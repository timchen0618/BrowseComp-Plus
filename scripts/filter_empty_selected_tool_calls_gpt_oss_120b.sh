#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IN="$ROOT_DIR/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.jsonl"
OUT="$ROOT_DIR/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.nonempty.jsonl"
DROPPED="$ROOT_DIR/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.dropped.jsonl"

python "$ROOT_DIR/src_utils/filter_empty_selected_tool_calls.py" \
  --input "$IN" \
  --output "$OUT" \
  --dropped "$DROPPED"

