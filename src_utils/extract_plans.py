#!/usr/bin/env python3
"""
Extract plans from agent output trace JSON files.

Scans a directory of trace files for "user" entries that contain plans
(prefixed with the planner injection message), strips the prefix, and
writes a JSONL file with {query_id, query_text, output}.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional


PLAN_PREFIX = (
    "Here is the planner's response; please follow the plan "
    "to answer the user's question: "
)


def load_query_map(tsv_path: Path) -> Dict[str, str]:
    """Load query_id -> query_text mapping from a TSV file."""
    mapping: Dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def extract_plan(trace: dict) -> Optional[str]:
    """Return the stripped plan text from a trace, or None if no plan exists."""
    for entry in trace.get("result", []):
        if entry.get("type") != "user":
            continue
        output = entry.get("output", "")
        if isinstance(output, str) and output.startswith(PLAN_PREFIX):
            return output[len(PLAN_PREFIX):]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plans from agent output trace JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing run_*.json trace files",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--query-file",
        default=Path("topics-qrels/queries.tsv"),
        type=Path,
        help="TSV file mapping query_id to query_text",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include traces with no plan as records with an empty output string",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    query_map = load_query_map(args.query_file)

    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    plans_found = 0
    skipped = 0

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: skipping {json_file.name}: {e}", file=sys.stderr)
                skipped += 1
                continue

            query_id = str(trace.get("query_id", ""))
            plan_text = extract_plan(trace)

            if plan_text is None:
                skipped += 1
                if not args.include_empty:
                    continue
                plan_text = ""
            else:
                plans_found += 1
                plan_text = "<plan>" + plan_text + "</plan>"

            record = {
                "query_id": query_id,
                "query_text": query_map.get(query_id, ""),
                "output": plan_text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(json_files)
    print(f"Processed {total} trace files")
    print(f"  Plans extracted: {plans_found}")
    print(f"  Skipped (no plan or error): {skipped}")
    print(f"  Output: {args.output_file}")


if __name__ == "__main__":
    main()

    # python src_utils/extract_plans.py --input-dir runs_first50/Qwen3-Embedding-8B/tongyi_planning_seed0/ --output-file tongyi_plans.jsonl --query-file topics-qrels/queries_first50.tsv