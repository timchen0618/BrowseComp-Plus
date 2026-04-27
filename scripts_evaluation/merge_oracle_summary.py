#!/usr/bin/env python3
"""
Merge multiple evaluation_summary.json files into a single oracle summary.

For each query, takes the oracle (best-of-N) result across all input directories:
  - correct = True if ANY seed was correct
  - recall / num_search_calls taken from the correct seed (or best-recall seed if none correct)

Typical usage:
    python scripts_evaluation/merge_oracle_summary.py \
        --dirs evals/.../seed0 evals/.../seed1 evals/.../seed2 \
        --output /tmp/oracle_group_a.json

    # Restrict to a specific query set:
    python scripts_evaluation/merge_oracle_summary.py \
        --dirs evals/.../seed0 evals/.../seed1 \
        --output /tmp/oracle.json \
        --ground_truth data/browsecomp_plus_decrypted_test150.jsonl
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_oracle(dirs: list[Path], gt_qids: set[str] | None) -> list[dict]:
    # best[qid] = {"correct": bool, "recall": float, "num_search_calls": int}
    best: dict[str, dict] = {}

    for d in dirs:
        summary_path = d / "evaluation_summary.json"
        if not summary_path.is_file():
            print(f"Skip {summary_path}: not found")
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = data.get("per_query_metrics", [])
        for row in rows:
            qid = str(row["query_id"])
            if gt_qids is not None and qid not in gt_qids:
                continue
            correct = bool(row.get("correct", False))
            recall = float(row.get("recall", 0.0))
            calls = int(row.get("num_search_calls", 0))

            prev = best.get(qid)
            if prev is None:
                best[qid] = {"correct": correct, "recall": recall, "num_search_calls": calls}
            else:
                if correct and not prev["correct"]:
                    # upgrade to first correct seed
                    best[qid] = {"correct": True, "recall": recall, "num_search_calls": calls}
                elif not correct and not prev["correct"] and recall > prev["recall"]:
                    # both wrong, take better recall
                    best[qid] = {"correct": False, "recall": recall, "num_search_calls": calls}
                # if prev already correct, keep it

    return [{"query_id": qid, **vals} for qid, vals in sorted(best.items())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge eval summaries into a single oracle (best-of-N) summary."
    )
    parser.add_argument("--dirs", nargs="+", required=True, type=Path,
                        help="Directories containing evaluation_summary.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output path for merged evaluation_summary.json")
    parser.add_argument("--ground_truth", default=None,
                        help="Optional JSONL ground truth file to filter query IDs")
    args = parser.parse_args()

    gt_qids = None
    if args.ground_truth:
        rows = read_jsonl(args.ground_truth)
        gt_qids = {str(r["query_id"]) for r in rows}
        print(f"Filtering to {len(gt_qids)} query IDs from {args.ground_truth}")

    per_query = merge_oracle(args.dirs, gt_qids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"per_query_metrics": per_query}, f, indent=2)

    n_correct = sum(1 for r in per_query if r["correct"])
    print(f"Written {len(per_query)} queries to {args.output}")
    print(f"Oracle accuracy: {n_correct}/{len(per_query)} = {n_correct/len(per_query)*100:.2f}%")


if __name__ == "__main__":
    main()
