#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IN="$ROOT_DIR/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.jsonl"
OUT="$ROOT_DIR/selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl"

python "$ROOT_DIR/src_utils/repair_selected_tool_calls_jsonl.py" --input "$IN" --output "$OUT"

