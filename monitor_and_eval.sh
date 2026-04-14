#!/bin/bash
# Monitor run completion and submit eval.SBATCH when all runs are done.
# Usage: nohup bash monitor_and_eval.sh &
# Check progress: tail -40 monitor_eval.log

PROJECT_DIR="/scratch/hc3337/projects/BrowseComp-Plus"
BASE="${PROJECT_DIR}/runs/bcp/Qwen3-Embedding-8B"
SBATCH_FILE="${PROJECT_DIR}/eval.SBATCH"
LOG="${PROJECT_DIR}/monitor_eval.log"
INTERVAL=300  # 5 minutes
MAX_CHECKS=2016  # 7 days

RUNS=(
  "full/gpt-oss-120b/traj_orig_ext_gpt-oss-120b_seed0"
  "full/gpt-oss-120b/traj_summary_orig_ext_gpt-oss-120b_seed0"
  "full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0"
  "first50/gpt-oss-120b/traj_orig_ext_gpt-oss-120b_seed0"
  "first50/gpt-oss-120b/traj_summary_orig_ext_gpt-oss-120b_seed0"
  "first50/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0"
)
EXPECTED=(830 830 830 50 50 50)
SHARD_REFS=(
  "topics-qrels/bcp/bcp_10_shards/q_*"
  "topics-qrels/bcp/bcp_10_shards/q_*"
  "topics-qrels/bcp/bcp_10_shards/q_*"
  "topics-qrels/bcp/queries_first50.tsv"
  "topics-qrels/bcp/queries_first50.tsv"
  "topics-qrels/bcp/queries_first50.tsv"
)

check_count=0
while [ $check_count -lt $MAX_CHECKS ]; do
  check_count=$((check_count + 1))
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  all_complete=true

  echo "[$timestamp] Check #$check_count" >> "$LOG"

  for i in "${!RUNS[@]}"; do
    count=$(ls "$BASE/${RUNS[$i]}"/*.json 2>/dev/null | wc -l)
    if [ "$count" -ne "${EXPECTED[$i]}" ]; then
      echo "  INCOMPLETE: ${RUNS[$i]} ($count/${EXPECTED[$i]})" >> "$LOG"
      shard_output=$(cd "$PROJECT_DIR" && python src_utils/find_missing_ids.py \
        --input_dir "$BASE/${RUNS[$i]}" \
        --reference_file "${SHARD_REFS[$i]}" 2>/dev/null \
        | grep -E "^\*\*\*Missing")
      if [ -n "$shard_output" ]; then
        while IFS= read -r line; do
          echo "    $line" >> "$LOG"
        done <<< "$shard_output"
      fi
      all_complete=false
    fi
  done

  if [ "$all_complete" = true ]; then
    echo "[$timestamp] ALL COMPLETE. Submitting eval.SBATCH..." >> "$LOG"
    sbatch_output=$(cd "$PROJECT_DIR" && sbatch "$SBATCH_FILE" 2>&1)
    echo "[$timestamp] sbatch: $sbatch_output" >> "$LOG"
    exit 0
  fi

  sleep $INTERVAL
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Timed out after 7 days." >> "$LOG"
exit 1
