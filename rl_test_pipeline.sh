#!/bin/bash
# rl_test_pipeline.sh — thin wrapper around rl_test_pipeline.py.
#
# Runs the orchestrator directly on the login node (no SLURM job needed —
# the orchestrator itself is lightweight; only the tiered tests need SLURM).
#
# Usage:
#   bash rl_test_pipeline.sh --submit          # check status / advance
#   bash rl_test_pipeline.sh --submit --reset  # fresh start
#   bash rl_test_pipeline.sh --submit --retry  # retry failed tier
#   bash rl_test_pipeline.sh                   # dry-run
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
exec python3.12 rl_test_pipeline.py "$@"
