#!/bin/bash
# Monitor run completion and submit eval.SBATCH when all 14 runs are done.
# Usage: nohup bash monitor_and_eval.sh &
# Checks every 5 minutes. Exits after submitting or after 7 days.

BASE="/scratch/hc3337/projects/BrowseComp-Plus/runs/bcp/Qwen3-Embedding-8B"
SBATCH_FILE="/scratch/hc3337/projects/BrowseComp-Plus/eval.SBATCH"
LOG="/scratch/hc3337/projects/BrowseComp-Plus/monitor_eval.log"
INTERVAL=300  # 5 minutes
MAX_CHECKS=2016  # 7 days worth

FULL_RUNS=(
  planning_retrospective_reinject_every_5_seed0
  planning_retrospective_seed0
  planning_v1_after_steps_5_seed3
  planning_v1_start_and_after_steps_5_seed3
  planning_v3_start_ext_gemini_2.5_pro_reinject_every_5_seed0
  planning_v3_start_ext_gemini_2.5_pro_seed0
  planning_v4_start_ext_gemini_2.5_pro_reinject_every_5_seed0
  planning_v4_start_ext_gemini_2.5_pro_seed0
  traj_ext_gpt-oss-120b_seed0
  traj_summary_ext_gpt-oss-120b_seed0
  traj_summary_ext_selected_tools_gpt-oss-120b_seed0
)

FIRST50_RUNS=(
  planning_v4_start_ext_gemini_2.5_pro_reinject_every_5_seed0
  planning_v4_start_ext_gemini_2.5_pro_seed0
  traj_summary_ext_selected_tools_gpt-oss-120b_seed0
)

check_count=0

while [ $check_count -lt $MAX_CHECKS ]; do
  check_count=$((check_count + 1))
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  all_complete=true

  echo "[$timestamp] Check #$check_count" >> "$LOG"

  for run in "${FULL_RUNS[@]}"; do
    count=$(ls "$BASE/full/gpt-oss-120b/$run"/*.json 2>/dev/null | wc -l)
    if [ "$count" -ne 830 ]; then
      echo "  INCOMPLETE: $run ($count/830)" >> "$LOG"
      all_complete=false
    fi
  done

  for run in "${FIRST50_RUNS[@]}"; do
    count=$(ls "$BASE/first50/gpt-oss-120b/$run"/*.json 2>/dev/null | wc -l)
    if [ "$count" -ne 50 ]; then
      echo "  INCOMPLETE: $run ($count/50) [first50]" >> "$LOG"
      all_complete=false
    fi
  done

  if [ "$all_complete" = true ]; then
    echo "[$timestamp] ALL RUNS COMPLETE. Submitting eval.SBATCH..." >> "$LOG"
    sbatch_output=$(sbatch "$SBATCH_FILE" 2>&1)
    echo "[$timestamp] sbatch output: $sbatch_output" >> "$LOG"
    echo "[$timestamp] Monitor exiting." >> "$LOG"
    exit 0
  fi

  sleep $INTERVAL
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max checks reached (7 days). Exiting without submitting." >> "$LOG"
exit 1
