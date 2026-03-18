#!/usr/bin/env python3
"""Find query IDs that are in the query file but missing from a plan JSONL file."""

import argparse
import csv
import json


def load_query_ids_from_tsv(path: str) -> set[str]:
    """Load query IDs from a TSV (query_id in first column, tab-separated)."""
    ids = set()
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                ids.add(str(row[0]).strip())
    return ids


def load_query_ids_from_plan_jsonl(path: str) -> set[str]:
    """Load query IDs from a plan JSONL (each line has query_id key)."""
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("query_id")
                if qid is not None:
                    ids.add(str(qid))
            except json.JSONDecodeError:
                pass
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find query IDs in query file that are missing from plan file."
    )
    parser.add_argument(
        "--plan-file",
        "-p",
        required=True,
        help="Path to plan JSONL file (e.g. qwen3.5-35B-A3B_plans.jsonl)",
    )
    parser.add_argument(
        "--query-file",
        "-q",
        default="topics-qrels/queries.tsv",
        help="Path to query TSV file (default: topics-qrels/queries.tsv)",
    )
    args = parser.parse_args()

    query_ids = load_query_ids_from_tsv(args.query_file)
    plan_ids = load_query_ids_from_plan_jsonl(args.plan_file)

    missing = sorted(query_ids - plan_ids, key=lambda x: (len(x), x))
    extra = sorted(plan_ids - query_ids, key=lambda x: (len(x), x))

    print(f"Query file: {args.query_file} ({len(query_ids)} query IDs)")
    print(f"Plan file:  {args.plan_file} ({len(plan_ids)} plan entries)")
    print(f"\nMissing from plan (in query file but not in plan): {len(missing)}")
    if missing:
        print(missing)
    else:
        print("  (none)")

    if extra:
        print(f"\nExtra in plan (in plan but not in query file): {len(extra)}")
        print(extra)


if __name__ == "__main__":
    main()
