"""
Correlate plan quality scores (1-5) with actual agent accuracy (0/1).

Plan scores: judge_results/qwen3.5-122B-A10B/pointwise/*.jsonl
Actual scores: evals/Qwen3-Embedding-8B/*/detailed_judge_results.csv
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

BASE = Path("/scratch/hc3337/projects/BrowseComp-Plus")
EVALS = BASE / "evals" / "Qwen3-Embedding-8B"
JUDGE = BASE / "judge_results" / "qwen3.5-122B-A10B" / "pointwise"

# Plan JSONL → list of eval directories that use those plans.
# Only new_prompt runs are included.
# start_ext runs use Gemini-generated plans (gemini_new_prompt.jsonl).
MAPPING = {
    "gpt-oss-120b_new_prompt.jsonl": [
        "gpt-oss-120b_planning_new_prompt_seed0",
    ],
    "gpt-oss-120b_new_prompt_aftersteps5.jsonl": [
        "gpt-oss-120b_planning_new_prompt_after_steps_5_seed0",
    ],
    "tongyi_new_prompt.jsonl": [
        "tongyi_planning_new_prompt_seed0",
    ],
    "tongyi_new_prompt_aftersteps5.jsonl": [
        "tongyi_planning_new_prompt_after_steps_5_seed0",
    ],
    "gemini_new_prompt.jsonl": [
        "gpt-oss-120b_planning_new_prompt_start_ext_seed0",
        "tongyi_planning_new_prompt_start_ext_seed0",
    ],
}


def load_plan_scores(jsonl_path):
    """Returns dict: query_id (str) -> plan_score (int 1-5)."""
    scores = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec["query_id"])
            try:
                output = json.loads(rec["judge_output"])
                score = int(output["score"])
            except Exception:
                continue
            scores[qid] = score
    return scores


def load_actual_scores(eval_dir):
    """Returns dict: query_id (str) -> actual_correct (0 or 1), skipping parse errors."""
    csv_path = eval_dir / "detailed_judge_results.csv"
    if not csv_path.exists():
        return {}
    scores = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["query_id"])
            # skip parse errors or missing judge result
            if row.get("parse_error", "").strip().lower() == "true":
                continue
            jc = row.get("judge_correct", "").strip().lower()
            if jc == "true":
                scores[qid] = 1
            elif jc == "false":
                scores[qid] = 0
            # else skip empty/missing
    return scores


def main():
    # Collect all (plan_score, actual_score) pairs
    all_pairs = []  # list of (plan_score, actual_score, source_label)
    missing_eval_dirs = []

    for plan_file, eval_dirs in MAPPING.items():
        plan_path = JUDGE / plan_file
        if not plan_path.exists():
            print(f"[WARN] Plan file not found: {plan_path}")
            continue
        plan_scores = load_plan_scores(plan_path)

        for eval_dir_name in eval_dirs:
            eval_dir = EVALS / eval_dir_name
            if not eval_dir.exists():
                missing_eval_dirs.append(eval_dir_name)
                continue
            actual_scores = load_actual_scores(eval_dir)

            matched = 0
            for qid, plan_score in plan_scores.items():
                if qid in actual_scores:
                    all_pairs.append((plan_score, actual_scores[qid], eval_dir_name))
                    matched += 1
            print(f"  {plan_file} x {eval_dir_name}: {matched} matched pairs")

    if missing_eval_dirs:
        print(f"\n[WARN] Missing eval dirs: {missing_eval_dirs}")

    print(f"\nTotal matched (plan_score, actual_score) pairs: {len(all_pairs)}")

    # ── Analysis 1: Plan score distribution by actual correctness ──
    print("\n" + "=" * 60)
    print("Analysis 1: Plan score distribution by actual correctness")
    print("(What plan scores do correct vs incorrect answers tend to get?)")
    print("=" * 60)

    correct_plan_scores = defaultdict(int)    # plan_score -> count when actual=1
    incorrect_plan_scores = defaultdict(int)  # plan_score -> count when actual=0

    for plan_score, actual_score, _ in all_pairs:
        if actual_score == 1:
            correct_plan_scores[plan_score] += 1
        else:
            incorrect_plan_scores[plan_score] += 1

    total_correct = sum(correct_plan_scores.values())
    total_incorrect = sum(incorrect_plan_scores.values())

    print(f"\n{'Plan Score':<12} {'Correct (actual=1)':<22} {'Incorrect (actual=0)':<22}")
    print("-" * 56)
    for s in range(1, 6):
        c = correct_plan_scores.get(s, 0)
        i = incorrect_plan_scores.get(s, 0)
        cp = f"{c} ({100*c/total_correct:.1f}%)" if total_correct else "0"
        ip = f"{i} ({100*i/total_incorrect:.1f}%)" if total_incorrect else "0"
        print(f"{s:<12} {cp:<22} {ip:<22}")
    print(f"{'Total':<12} {total_correct:<22} {total_incorrect:<22}")

    # Mean plan score for each actual outcome
    correct_scores = [p for p, a, _ in all_pairs if a == 1]
    incorrect_scores = [p for p, a, _ in all_pairs if a == 0]
    if correct_scores:
        print(f"\nMean plan score | correct answers:   {sum(correct_scores)/len(correct_scores):.3f}")
    if incorrect_scores:
        print(f"Mean plan score | incorrect answers: {sum(incorrect_scores)/len(incorrect_scores):.3f}")

    # ── Analysis 2: Actual score distribution by plan score ──
    print("\n" + "=" * 60)
    print("Analysis 2: Actual accuracy (%) by plan score level")
    print("(For each plan score 1-5, what fraction of answers are correct?)")
    print("=" * 60)

    by_plan_score = defaultdict(list)  # plan_score -> list of actual scores
    for plan_score, actual_score, _ in all_pairs:
        by_plan_score[plan_score].append(actual_score)

    print(f"\n{'Plan Score':<12} {'Count':<8} {'# Correct':<12} {'Accuracy':<10}")
    print("-" * 42)
    for s in range(1, 6):
        scores_list = by_plan_score.get(s, [])
        n = len(scores_list)
        n_correct = sum(scores_list)
        acc = f"{100*n_correct/n:.1f}%" if n else "N/A"
        print(f"{s:<12} {n:<8} {n_correct:<12} {acc:<10}")

    # ── Point-biserial correlation ──
    print("\n" + "=" * 60)
    print("Point-biserial correlation: plan_score vs actual_correct")
    print("=" * 60)

    if all_pairs:
        import math
        plan_scores_all = [p for p, _, _ in all_pairs]
        actual_scores_all = [a for _, a, _ in all_pairs]
        n = len(all_pairs)

        mean_plan = sum(plan_scores_all) / n
        mean_actual = sum(actual_scores_all) / n

        cov = sum((p - mean_plan) * (a - mean_actual) for p, a, _ in all_pairs) / n
        std_plan = math.sqrt(sum((p - mean_plan) ** 2 for p, _, _ in all_pairs) / n)
        std_actual = math.sqrt(sum((a - mean_actual) ** 2 for _, a, _ in all_pairs) / n)

        if std_plan > 0 and std_actual > 0:
            pearson_r = cov / (std_plan * std_actual)
            print(f"\nPearson r (plan_score, actual_correct): {pearson_r:.4f}")
            print(f"n = {n}, mean_plan_score = {mean_plan:.3f}, mean_accuracy = {100*mean_actual:.1f}%")
        else:
            print("Cannot compute correlation (zero variance).")


if __name__ == "__main__":
    main()
