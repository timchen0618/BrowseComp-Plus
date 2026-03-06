"""Evaluate GAR pilot experiment results across conditions.

Primary metrics: final answer accuracy (string match) + efficiency.
Secondary diagnostics: retrieval overlap, generation statistics.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def extract_answer_from_trajectory(traj: dict) -> str:
    for entry in reversed(traj.get("result", [])):
        if entry.get("type") == "output_text":
            return entry.get("output", "").strip()
    return ""


def string_match(prediction: str, gold: str) -> bool:
    return gold.lower().strip() in prediction.lower().strip()


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def bootstrap_delta_ci(values_a, values_b, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap CI on difference (B - A) using paired resampling."""
    rng = np.random.RandomState(seed)
    assert len(values_a) == len(values_b)
    n = len(values_a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        mean_a = np.mean([values_a[i] for i in idx])
        mean_b = np.mean([values_b[i] for i in idx])
        deltas.append(mean_b - mean_a)
    lower = np.percentile(deltas, (1 - ci) / 2 * 100)
    upper = np.percentile(deltas, (1 + ci) / 2 * 100)
    return np.mean(deltas), lower, upper


def load_condition(condition_dir: Path, ground_truth: dict, pilot_ids: set):
    """Load trajectories for a condition, matched to pilot IDs."""
    results = {}
    for json_path in sorted(condition_dir.glob("run_*.json")):
        with open(json_path) as f:
            traj = json.load(f)
        qid = str(traj.get("query_id", ""))
        if qid not in pilot_ids:
            continue
        if qid in results:
            continue  # skip duplicates
        prediction = extract_answer_from_trajectory(traj)
        gold = ground_truth.get(qid, {}).get("answer", "")
        meta = traj.get("metadata", {})
        results[qid] = {
            "prediction": prediction,
            "gold": gold,
            "correct": string_match(prediction, gold) if gold else False,
            "status": traj.get("status", "unknown"),
            "time_taken": meta.get("time_taken", 0.0),
            "total_inference_time": meta.get("total_inference_time", 0.0),
            "total_tool_call_time": meta.get("total_tool_call_time", 0.0),
            "total_prompt_tokens": meta.get("total_prompt_tokens", 0),
            "total_completion_tokens": meta.get("total_completion_tokens", 0),
            "search_count": traj.get("tool_call_counts", {}).get("search", 0),
            # GAR-specific
            "gar_mode": meta.get("gar_mode", "off"),
            "total_generation_time_s": meta.get("total_generation_time_s", 0.0),
            "total_generation_tokens": meta.get("total_generation_tokens", 0),
            "generation_call_count": meta.get("generation_call_count", 0),
        }
    return results


def print_condition_stats(name: str, results: dict):
    n = len(results)
    if n == 0:
        print(f"\n{name}: NO RESULTS")
        return

    correct = [r["correct"] for r in results.values()]
    acc = np.mean(correct)
    ci_lo, ci_hi = bootstrap_ci([float(c) for c in correct])

    times = [r["time_taken"] for r in results.values() if r["time_taken"] > 0]
    searches = [r["search_count"] for r in results.values()]
    prompt_tokens = [r["total_prompt_tokens"] for r in results.values()]
    completion_tokens = [r["total_completion_tokens"] for r in results.values()]
    gen_times = [r["total_generation_time_s"] for r in results.values()]
    gen_tokens = [r["total_generation_tokens"] for r in results.values()]
    gen_calls = [r["generation_call_count"] for r in results.values()]

    print(f"\n{'='*60}")
    print(f"Condition: {name} (n={n})")
    print(f"{'='*60}")
    print(f"  Accuracy:        {acc:.1%} [{ci_lo:.1%}, {ci_hi:.1%}] (95% CI)")
    print(f"  Completed:       {sum(1 for r in results.values() if r['status'] == 'completed')}/{n}")
    print(f"  Avg searches:    {np.mean(searches):.1f} (median {np.median(searches):.0f})")
    print(f"  Avg wall-clock:  {np.mean(times):.0f}s (median {np.median(times):.0f}s)")
    print(f"  Avg prompt tok:  {np.mean(prompt_tokens):.0f}")
    print(f"  Avg compl tok:   {np.mean(completion_tokens):.0f}")

    if any(gc > 0 for gc in gen_calls):
        print(f"  --- Generation overhead ---")
        print(f"  Avg gen calls:   {np.mean(gen_calls):.1f}")
        print(f"  Avg gen tokens:  {np.mean(gen_tokens):.0f}")
        print(f"  Avg gen time:    {np.mean(gen_times):.1f}s")
        print(f"  Gen time / total:{np.mean(gen_times) / max(np.mean(times), 1):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAR pilot results")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/scratch/afw8937/efficient-search-agents/runs/gar_pilot",
        help="Base directory containing condition subdirectories",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="/scratch/afw8937/efficient-search-agents/data/browsecomp_plus_decrypted.jsonl",
    )
    parser.add_argument(
        "--pilot-jsonl",
        type=str,
        default=None,
        help="Path to pilot_100.jsonl (if not set, evaluates all questions in run dirs)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "gar_no_gate", "gar_gated"],
    )
    args = parser.parse_args()

    # Load ground truth
    gt = {}
    with open(args.ground_truth) as f:
        for line in f:
            entry = json.loads(line)
            gt[str(entry["query_id"])] = {
                "query": entry["query"],
                "answer": entry["answer"],
            }

    # Load pilot IDs (or use all)
    pilot_ids = None
    if args.pilot_jsonl:
        pilot_ids = set()
        with open(args.pilot_jsonl) as f:
            for line in f:
                entry = json.loads(line)
                pilot_ids.add(str(entry["query_id"]))
        print(f"Evaluating on {len(pilot_ids)} pilot questions")
    else:
        pilot_ids = set(gt.keys())
        print(f"Evaluating on all {len(pilot_ids)} questions")

    # Load and evaluate each condition
    all_results = {}
    for cond in args.conditions:
        cond_dir = Path(args.run_dir) / cond
        if not cond_dir.exists():
            print(f"Skipping {cond}: directory not found at {cond_dir}")
            continue
        results = load_condition(cond_dir, gt, pilot_ids)
        all_results[cond] = results
        print_condition_stats(cond, results)

    # Paired comparison: baseline vs gar_no_gate
    if "baseline" in all_results and "gar_no_gate" in all_results:
        base = all_results["baseline"]
        gar = all_results["gar_no_gate"]
        common = set(base.keys()) & set(gar.keys())
        if len(common) > 10:
            base_correct = [float(base[qid]["correct"]) for qid in sorted(common)]
            gar_correct = [float(gar[qid]["correct"]) for qid in sorted(common)]
            delta, ci_lo, ci_hi = bootstrap_delta_ci(base_correct, gar_correct)
            print(f"\n{'='*60}")
            print(f"Paired comparison: gar_no_gate - baseline (n={len(common)})")
            print(f"{'='*60}")
            print(f"  Delta accuracy:  {delta:+.1%} [{ci_lo:+.1%}, {ci_hi:+.1%}] (95% CI)")
            sig = "YES" if (ci_lo > 0 or ci_hi < 0) else "NO"
            print(f"  Significant:     {sig} (CI excludes 0)")

            # Per-question breakdown
            improved = sum(1 for qid in common if not base[qid]["correct"] and gar[qid]["correct"])
            degraded = sum(1 for qid in common if base[qid]["correct"] and not gar[qid]["correct"])
            print(f"  Improved:        {improved} questions")
            print(f"  Degraded:        {degraded} questions")


if __name__ == "__main__":
    main()
