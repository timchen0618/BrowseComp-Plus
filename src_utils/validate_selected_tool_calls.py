#!/usr/bin/env python3
"""
Validate selected_tool_calls JSONL files.

Checks per file:
  - Total record count
  - Duplicate query_ids
  - Records with empty selected_indices
  - Records with malformed selected_indices (not a list of ints)
  - Records flagged with an error field
  - Coverage against an expected query TSV (optional)

Usage:
    # Validate a single file
    python src_utils/validate_selected_tool_calls.py \
        --input selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls.jsonl

    # Validate all files in a directory
    python src_utils/validate_selected_tool_calls.py \
        --input selected_tool_calls/all/gpt-oss-120b/seed0/

    # Also check coverage against expected query set
    python src_utils/validate_selected_tool_calls.py \
        --input selected_tool_calls/all/gpt-oss-120b/seed0/ \
        --query-tsv topics-qrels/bcp/queries.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_expected_qids(tsv_path: str) -> list[str]:
    qids = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                qids.append(line.split("\t")[0])
    return qids


def is_valid_selected_indices(val) -> bool:
    return isinstance(val, list) and len(val) > 0 and all(isinstance(x, int) for x in val)


def validate_file(path: Path, expected_qids: list[str] | None) -> bool:
    records = []
    parse_errors = 0

    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
                print(f"  [ERROR] line {i}: invalid JSON")

    total = len(records)
    print(f"\n{path}")
    print(f"  total records : {total}" + (f"  (+ {parse_errors} unparseable lines)" if parse_errors else ""))

    # Duplicate query_ids
    qids_seen: dict[str, int] = {}
    for r in records:
        qid = r.get("query_id", "<missing>")
        qids_seen[qid] = qids_seen.get(qid, 0) + 1
    dupes = {qid: n for qid, n in qids_seen.items() if n > 1}
    if dupes:
        print(f"  duplicates    : {len(dupes)} query_ids appear more than once")
        for qid, n in list(dupes.items())[:5]:
            print(f"    {qid}: {n}x")
        if len(dupes) > 5:
            print(f"    ... and {len(dupes) - 5} more")
    else:
        print(f"  duplicates    : 0")

    # selected_indices issues (exclude records with errors — empty indices is expected there)
    empty = [r.get("query_id") for r in records
             if not r.get("error")
             and isinstance(r.get("selected_indices"), list) and len(r["selected_indices"]) == 0]
    malformed = [r.get("query_id") for r in records
                 if not r.get("error")
                 and not isinstance(r.get("selected_indices"), list)]
    print(f"  empty sel_idx : {len(empty)}")
    for qid in empty:
        print(f"    {qid}")
    print(f"  malformed     : {len(malformed)}")
    for qid in malformed:
        print(f"    {qid}")

    # correct_num_selected (exclude error records for the same reason)
    incorrect = [r.get("query_id") for r in records
                 if not r.get("error") and r.get("correct_num_selected") is False]
    pct = 100 * len(incorrect) / total if total else 0
    print(f"  incorrect_num : {len(incorrect)}/{total} ({pct:.1f}%)")
    for qid in incorrect:
        print(f"    {qid}")

    # Error records
    errors = [r.get("query_id") for r in records if r.get("error")]
    if errors:
        error_types: dict[str, int] = {}
        for r in records:
            if r.get("error"):
                et = str(r["error"])
                error_types[et] = error_types.get(et, 0) + 1
        print(f"  error records : {len(errors)}")
        for et, n in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {et}: {n}")
        for r in records:
            if r.get("error"):
                print(f"      {r.get('query_id')}")
    else:
        print(f"  error records : 0")

    # Coverage against expected query set
    if expected_qids is not None:
        present = set(qids_seen.keys())
        expected_set = set(expected_qids)
        missing = expected_set - present
        extra = present - expected_set
        print(f"  coverage      : {len(present & expected_set)}/{len(expected_qids)} expected query IDs present")
        if missing:
            print(f"  missing qids  : {len(missing)}")
            for qid in sorted(missing)[:5]:
                print(f"    {qid}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if extra:
            print(f"  extra qids    : {len(extra)} not in expected set")

    ok = parse_errors == 0 and not dupes and not empty and not malformed and not errors
    if expected_qids is not None:
        ok = ok and not missing and not extra
    print(f"  status        : {'OK' if ok else 'ISSUES FOUND'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Validate selected_tool_calls JSONL files.")
    ap.add_argument("--input", required=True,
                    help="Path to a JSONL file or directory of JSONL files.")
    ap.add_argument("--query-tsv", default=None,
                    help="Optional TSV of expected query IDs to check coverage (e.g. topics-qrels/bcp/queries.tsv).")
    args = ap.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {input_path}")
            return
    else:
        files = [input_path]

    expected_qids = load_expected_qids(args.query_tsv) if args.query_tsv else None

    all_ok = True
    for f in files:
        ok = validate_file(f, expected_qids)
        all_ok = all_ok and ok

    print(f"\n{'All files OK' if all_ok else 'Some files have issues'} ({len(files)} file(s) checked)")


if __name__ == "__main__":
    main()
