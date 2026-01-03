#!/usr/bin/env python3
"""
Aggregate per-query scores from multiple evaluation_summary.json files.

For each directory passed via --dirs, this script reads the file
<dir>/evaluation_summary.json, collects the per-query metric values,
takes the max score per query_id across all provided dirs, and reports
the average of those per-query maxima.

Typical usage:
    python compute_pass_k.py --dirs run1 run2 run3 --metric correct
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union, Any

Number = Union[int, float]


def read_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def load_per_query_scores(summary_path: Path, metric: str) -> List[Tuple[str, float]]:
    """Load per-query metric values from a summary file."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_query = data.get("per_query_metrics")
    if not isinstance(per_query, list):
        raise ValueError(f"'per_query_metrics' missing or not a list in {summary_path}")

    results: List[Tuple[str, float]] = []
    for item in per_query:
        if not isinstance(item, dict):
            continue
        qid = item.get("query_id")
        if qid is None:
            continue
        raw_val = item.get(metric)
        if raw_val is None:
            continue
        # Coerce booleans to numeric for aggregation.
        if isinstance(raw_val, bool):
            val: Number = 1.0 if raw_val else 0.0
        elif isinstance(raw_val, (int, float)):
            val = float(raw_val)
        else:
            raise ValueError(f"Metric '{metric}' in {summary_path} is not numeric/bool")
        results.append((str(qid), float(val)))
    return results


def aggregate_scores(dirs: Iterable[Path], metric: str, ground_truth: str) -> None:
    """Aggregate per-query scores and print the average of per-query maxima."""
    if ground_truth is not None:
        ground_truth = read_jsonl(ground_truth)
        gt_query_ids = [inst['query_id'] for inst in ground_truth]
    else:
        gt_query_ids = None
        
    dirs = list(dirs)  # ensure we can iterate multiple times
    scores_by_qid: Dict[str, List[float]] = {}

    for directory in dirs:
        summary_path = directory / "evaluation_summary.json"
        if not summary_path.is_file():
            print(f"Skip {summary_path}: file not found")
            continue
        try:
            entries = load_per_query_scores(summary_path, metric)
        except Exception as exc:
            print(f"Skip {summary_path}: {exc}")
            continue
        
        if gt_query_ids is None:
            per_dir_avg_score = sum([score for qid, score in entries]) / len(entries)
        else:
            per_dir_avg_score = sum([score for qid, score in entries if qid in gt_query_ids]) / len(gt_query_ids)
        print(f"Average score for {directory}: {per_dir_avg_score:.4f}")
        for qid, score in entries:
            if (gt_query_ids is not None) and (qid not in gt_query_ids):
                continue
            scores_by_qid.setdefault(qid, []).append(score)

    if not scores_by_qid:
        print("No scores found to aggregate.")
        return

    max_by_qid: Dict[str, float] = {qid: max(vals) for qid, vals in scores_by_qid.items()}
    average_by_qid = {qid: sum(vals) / len(vals) for qid, vals in scores_by_qid.items()}
    num_answers_by_qid = {qid: len(vals) for qid, vals in scores_by_qid.items()}
    average_num_answers = sum(num_answers_by_qid.values()) / len(num_answers_by_qid)
    average_max_score = sum(max_by_qid.values()) / len(max_by_qid)
    average_average_score = sum(average_by_qid.values()) / len(average_by_qid)

    print(f"Aggregated metric '{metric}' from {len(scores_by_qid)} queries.")
    print(f"Directories processed: {len(dirs)}")
    print(f"Average of per-query maxima: {average_max_score:.4f}")
    print(f"Average of per-query averages: {average_average_score:.4f}")
    print(f"Average number of answers: {average_num_answers:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average of per-query max scores across evaluation summaries."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        type=Path,
        help="One or more directories containing evaluation_summary.json",
    )
    parser.add_argument(
        "--metric",
        default="correct",
        help="Metric key inside per_query_metrics to aggregate (default: correct).",
    )
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground truth file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_scores(args.dirs, metric=args.metric, ground_truth=args.ground_truth)


if __name__ == "__main__":
    main()

