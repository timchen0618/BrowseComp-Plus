#!/usr/bin/env bash
# Rename existing random_tools_seed0/ dirs to random_tools_seed42/ to match
# the actual selection seed used by random_select_tool_calls.py (the "seed0"
# label was misleading — there is no agent-level seed; the only seed actually
# used is the selection seed = 42 in the existing JSONLs).
#
# This affects both runs/ and evals/ and all 3 BCP models (test150 split).
#
# Usage:
#   scripts/rename_random_tools_dirs.sh         # dry run (default)
#   scripts/rename_random_tools_dirs.sh --do    # actually rename
set -euo pipefail

DO_RENAME=0
if [ "${1:-}" = "--do" ]; then DO_RENAME=1; fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

OLD_NAME="random_tools_seed0"
NEW_NAME="random_tools_seed42"

for parent in runs evals; do
    for model in glm-4.7-flash minimax-m2.5 qwen3.5-122b-a10b; do
        src="$parent/bcp/Qwen3-Embedding-8B/test150/$model/$OLD_NAME"
        dst="$parent/bcp/Qwen3-Embedding-8B/test150/$model/$NEW_NAME"
        if [ ! -d "$src" ]; then
            echo "[skip]  $src (does not exist)"
            continue
        fi
        if [ -d "$dst" ]; then
            echo "[skip]  $dst already exists"
            continue
        fi
        if [ "$DO_RENAME" -eq 1 ]; then
            mv "$src" "$dst"
            echo "[done]  $src -> $dst"
        else
            echo "[plan]  mv $src $dst"
        fi
    done
done

if [ "$DO_RENAME" -eq 0 ]; then
    echo
    echo "(dry run — re-invoke with --do to rename)"
fi
