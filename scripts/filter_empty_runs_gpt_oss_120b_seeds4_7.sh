#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE="$ROOT_DIR/runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b"

python "$ROOT_DIR/src_utils/filter_empty_runs.py" \
  --input-dirs \
    "$BASE/seed4" \
    "$BASE/seed5" \
    "$BASE/seed6" \
    "$BASE/seed7" \
  --mode list \
  --write-lists

echo
echo "If you want a cleaned copy directory (non-empty only), run:"
echo "  python \"$ROOT_DIR/src_utils/filter_empty_runs.py\" --input-dirs \"$BASE/seed4\" \"$BASE/seed5\" \"$BASE/seed6\" \"$BASE/seed7\" --mode copy-nonempty --output-root \"$BASE/filtered_nonempty\" --write-lists"
echo
echo "If you want to move empties aside, run:"
echo "  python \"$ROOT_DIR/src_utils/filter_empty_runs.py\" --input-dirs \"$BASE/seed4\" \"$BASE/seed5\" \"$BASE/seed6\" \"$BASE/seed7\" --mode move-empty --output-root \"$BASE/empties\" --write-lists"

