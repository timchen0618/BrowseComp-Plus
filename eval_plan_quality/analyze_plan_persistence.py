"""
Analyze how often the executor references the plan throughout its trajectory,
and whether plan persistence correlates with correctness.

Usage:
    python eval_plan_quality/analyze_plan_persistence.py \
        --traj-dir runs/Qwen3-Embedding-8B/gpt-oss-120b/gpt-oss-120b_planning_start_ext_seed2 \
        --eval-csv evals/Qwen3-Embedding-8B/gpt-oss-120b_planning_start_ext_seed2/detailed_judge_results.csv

Optional:
    --patterns      Comma-separated regex patterns to detect plan references
                    (default: "the plan|our plan|per the plan|phase \\d|according to|following the plan|planner|plan says|plan suggests")
    --n-buckets     Number of trajectory segments to split into (default: 4)
    --min-steps     Skip trajectories with fewer steps (default: 10)
    --output-csv    Save per-trajectory stats to CSV
"""

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_PATTERNS = (
    r"the plan|our plan|per the plan|phase \d|according to|"
    r"following the plan|planner|plan says|plan suggests"
)


def extract_reasoning_texts(result):
    """Yield (step_index, text) for each reasoning entry."""
    for i, r in enumerate(result):
        if r.get("type") != "reasoning":
            continue
        out = r.get("output", "")
        if isinstance(out, list):
            out = str(out[0]) if out else ""
        else:
            out = str(out)
        yield i, out


def analyze_trajectory(result, pattern, n_buckets):
    """
    Split a trajectory into n_buckets equal segments and count plan references
    in each. Returns dict with per-bucket stats.
    """
    n_steps = len(result)
    if n_steps == 0:
        return None

    bucket_size = n_steps / n_buckets
    bucket_refs = [0] * n_buckets
    bucket_reasoning = [0] * n_buckets

    for step_idx, text in extract_reasoning_texts(result):
        bucket = min(int(step_idx / bucket_size), n_buckets - 1)
        bucket_reasoning[bucket] += 1
        if pattern.search(text):
            bucket_refs[bucket] += 1

    bucket_rates = []
    for i in range(n_buckets):
        rate = bucket_refs[i] / bucket_reasoning[i] if bucket_reasoning[i] > 0 else 0.0
        bucket_rates.append(rate)

    return {
        "n_steps": n_steps,
        "n_reasoning": sum(bucket_reasoning),
        "total_refs": sum(bucket_refs),
        "bucket_refs": bucket_refs,
        "bucket_reasoning": bucket_reasoning,
        "bucket_rates": bucket_rates,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze plan reference persistence in agent trajectories")
    parser.add_argument("--traj-dir", required=True, help="Directory with trajectory JSON files")
    parser.add_argument("--eval-csv", required=True, help="Path to detailed_judge_results.csv")
    parser.add_argument("--patterns", default=DEFAULT_PATTERNS, help="Regex patterns for plan references")
    parser.add_argument("--n-buckets", type=int, default=4, help="Number of trajectory segments (default: 4)")
    parser.add_argument("--min-steps", type=int, default=10, help="Skip trajectories shorter than this")
    parser.add_argument("--output-csv", default=None, help="Optional: save per-trajectory stats to CSV")
    args = parser.parse_args()

    pattern = re.compile(args.patterns, re.IGNORECASE)

    # Load eval labels
    eval_labels = {}  # query_id -> {"correct": bool, "completed": bool}
    with open(args.eval_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["query_id"])
            eval_labels[qid] = {
                "correct": row.get("judge_correct", "").strip().lower() == "true",
                "completed": row.get("is_completed", "").strip().lower() == "true",
            }

    # Analyze trajectories
    per_traj = []  # list of dicts
    for fpath in sorted(glob.glob(f"{args.traj_dir}/*.json")):
        with open(fpath) as f:
            data = json.load(f)

        qid = str(data.get("query_id", ""))
        result = data.get("result", [])

        if len(result) < args.min_steps:
            continue

        stats = analyze_trajectory(result, pattern, args.n_buckets)
        if stats is None:
            continue

        label = eval_labels.get(qid, {})
        stats["query_id"] = qid
        stats["correct"] = label.get("correct", None)
        stats["completed"] = label.get("completed", None)
        stats["file"] = Path(fpath).name
        per_traj.append(stats)

    if not per_traj:
        print("No trajectories found.", file=sys.stderr)
        sys.exit(1)

    # --- Aggregate reports ---

    # 1. Overall bucket rates
    bucket_labels = [f"seg_{i+1}/{args.n_buckets}" for i in range(args.n_buckets)]
    print(f"\n{'='*70}")
    print(f"Plan Reference Persistence Analysis")
    print(f"  Trajectories: {len(per_traj)}  |  Buckets: {args.n_buckets}  |  Min steps: {args.min_steps}")
    print(f"{'='*70}")

    print(f"\n--- Overall plan mention rate by trajectory segment ---")
    for b in range(args.n_buckets):
        total_refs = sum(t["bucket_refs"][b] for t in per_traj)
        total_reasoning = sum(t["bucket_reasoning"][b] for t in per_traj)
        rate = total_refs / total_reasoning * 100 if total_reasoning > 0 else 0
        print(f"  {bucket_labels[b]:>12s}: {total_refs:>5d}/{total_reasoning:>6d} reasoning steps ({rate:.1f}%)")

    # 2. By correctness
    print(f"\n--- Plan mention rate by correctness ---")
    for group_label, group_filter in [("CORRECT", True), ("INCORRECT", False)]:
        subset = [t for t in per_traj if t["correct"] == group_filter]
        if not subset:
            continue
        print(f"\n  {group_label} (n={len(subset)}):")
        for b in range(args.n_buckets):
            total_refs = sum(t["bucket_refs"][b] for t in subset)
            total_reasoning = sum(t["bucket_reasoning"][b] for t in subset)
            rate = total_refs / total_reasoning * 100 if total_reasoning > 0 else 0
            print(f"    {bucket_labels[b]:>12s}: {total_refs:>5d}/{total_reasoning:>6d} ({rate:.1f}%)")

        # Persistence ratio: last bucket rate / first bucket rate
        first_refs = sum(t["bucket_refs"][0] for t in subset)
        first_reas = sum(t["bucket_reasoning"][0] for t in subset)
        last_refs = sum(t["bucket_refs"][-1] for t in subset)
        last_reas = sum(t["bucket_reasoning"][-1] for t in subset)
        first_rate = first_refs / first_reas if first_reas > 0 else 0
        last_rate = last_refs / last_reas if last_reas > 0 else 0
        persistence = last_rate / first_rate * 100 if first_rate > 0 else 0
        print(f"    Persistence (last/first): {persistence:.0f}%")

    # 3. By completion
    print(f"\n--- Plan mention rate by completion ---")
    for group_label, group_filter in [("COMPLETED", True), ("INCOMPLETE", False)]:
        subset = [t for t in per_traj if t["completed"] == group_filter]
        if not subset:
            continue
        print(f"\n  {group_label} (n={len(subset)}):")
        for b in range(args.n_buckets):
            total_refs = sum(t["bucket_refs"][b] for t in subset)
            total_reasoning = sum(t["bucket_reasoning"][b] for t in subset)
            rate = total_refs / total_reasoning * 100 if total_reasoning > 0 else 0
            print(f"    {bucket_labels[b]:>12s}: {total_refs:>5d}/{total_reasoning:>6d} ({rate:.1f}%)")

    # 4. Zero-reference trajectories
    zero_ref = [t for t in per_traj if t["total_refs"] == 0]
    nonzero_ref = [t for t in per_traj if t["total_refs"] > 0]
    print(f"\n--- Trajectories with zero plan references ---")
    print(f"  Zero refs: {len(zero_ref)}/{len(per_traj)} ({len(zero_ref)/len(per_traj)*100:.1f}%)")
    if zero_ref:
        zero_correct = sum(1 for t in zero_ref if t["correct"])
        nonzero_correct = sum(1 for t in nonzero_ref if t["correct"])
        print(f"  Zero-ref accuracy:    {zero_correct}/{len(zero_ref)} ({zero_correct/len(zero_ref)*100:.1f}%)")
    if nonzero_ref:
        nonzero_correct = sum(1 for t in nonzero_ref if t["correct"])
        print(f"  Nonzero-ref accuracy: {nonzero_correct}/{len(nonzero_ref)} ({nonzero_correct/len(nonzero_ref)*100:.1f}%)")

    # 5. Optional CSV output
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = ["query_id", "file", "correct", "completed", "n_steps", "n_reasoning", "total_refs"]
            for b in range(args.n_buckets):
                fieldnames.extend([f"refs_{b+1}", f"reasoning_{b+1}", f"rate_{b+1}"])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in per_traj:
                row = {
                    "query_id": t["query_id"],
                    "file": t["file"],
                    "correct": t["correct"],
                    "completed": t["completed"],
                    "n_steps": t["n_steps"],
                    "n_reasoning": t["n_reasoning"],
                    "total_refs": t["total_refs"],
                }
                for b in range(args.n_buckets):
                    row[f"refs_{b+1}"] = t["bucket_refs"][b]
                    row[f"reasoning_{b+1}"] = t["bucket_reasoning"][b]
                    row[f"rate_{b+1}"] = f"{t['bucket_rates'][b]:.4f}"
                writer.writerow(row)
        print(f"\nPer-trajectory stats saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
