#!/bin/bash
# Run the auto-pipeline in the background.
# Usage:
#   nohup bash auto_pipeline.sh --submit > auto_pipeline.log 2>&1 &
#   tail -f auto_pipeline.log
#   cat auto_pipeline_state.json | jq .
set -euo pipefail
cd "$(dirname "$0")"
exec python auto_pipeline.py "$@"
