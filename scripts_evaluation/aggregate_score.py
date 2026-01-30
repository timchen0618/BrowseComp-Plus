#!/usr/bin/env python3
"""
Aggregate evaluation scores from retriever/run_name subdirectories.

Expected layout under the provided root directory:
<root>/<retriever>/<run_name>/evaluation_summary.json

For each summary, the script extracts:
- Accuracy (%)                -> written as accuracy_percent
- Recall (%)                  -> written as recall_percent
- token_limit_reached_percent -> written as token_limit_percent

The collected rows are written to a CSV (default: <root>/aggregated_scores.csv).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FieldNames = Tuple[
    str, str, str, str, str
]  # retriever, run_name, accuracy, recall, token limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy, recall, and token limit stats from evaluation summaries."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing <retriever>/<run_name>/evaluation_summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output CSV (default: <root>/aggregated_scores.csv).",
    )
    return parser.parse_args()


def find_summary_paths(root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (retriever, run_name, summary_path) tuples for every evaluation_summary.json
    found two directory levels below the root.
    """
    for retriever_dir in sorted(root.iterdir()):
        if not retriever_dir.is_dir():
            continue
        for run_dir in sorted(retriever_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "evaluation_summary.json"
            yield retriever_dir.name, run_dir.name, summary_path


def extract_metrics(summary_path: Path) -> Dict[str, float]:
    """Read the summary JSON and pull out the metrics we care about."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "accuracy_percent": data.get("Accuracy (%)"),
        "recall_percent": data.get("Recall (%)"),
        "token_limit_percent": data.get("token_limit_reached_percent"),
        "average_tool_calls": data.get("avg_tool_stats", {}).get("search", 0),
    }


def collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for retriever, run_name, summary_path in find_summary_paths(root):
        if not summary_path.is_file():
            print(f"Skip {summary_path} (file not found)", file=sys.stderr)
            continue
        try:
            metrics = extract_metrics(summary_path)
        except Exception as exc:
            print(f"Skip {summary_path} (failed to read): {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "retriever": retriever,
                "run_name": run_name,
                **metrics,
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: FieldNames = (
        "retriever",
        "run_name",
        "accuracy_percent",
        "recall_percent",
        "token_limit_percent",
        "average_tool_calls",
    )
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (root / "aggregated_scores.csv")
    rows = collect_rows(root)

    if not rows:
        print("No evaluation_summary.json files found; no CSV written.", file=sys.stderr)
        sys.exit(1)

    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
