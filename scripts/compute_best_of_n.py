#!/usr/bin/env python3
"""Compute pass@N (best-of-N) accuracy across multiple eval JSONs over the qid intersection.

For best-of-4 random tool calls: a question is "best-of-N correct" if ANY of the
N seeds' evaluation_summary.json marks that qid correct. We compute this on the
qid intersection (qids that appear in ALL N evals) so the math is clean —
seeds with partial-N coverage (e.g., Qwen3.5 seed42 N=130) only contribute their
qids to the comparison.

Output:
  - per-seed accuracy on the intersection (for honest comparison)
  - pass@N accuracy on the intersection
  - count of qids in the intersection vs union
  - lift over best single seed

Usage:
    python scripts/compute_best_of_n.py \\
        --inputs evals/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/random_tools_seed{42,43,44,45}/evaluation_summary.json \\
        --label "GLM-4.7-Flash random tool calls"

Or to dump per-qid breakdown:
    python scripts/compute_best_of_n.py --inputs ... --output-csv overlap_glm.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_eval(path: Path) -> dict[str, dict[str, Any]]:
    """Return qid -> per-query metric dict."""
    if not path.is_file():
        sys.exit(f"Not a file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in obj.get("per_query_metrics", []):
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue
        out[qid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="Two or more evaluation_summary.json files")
    ap.add_argument("--label", type=str, default=None,
                    help="Human-readable label (used in output)")
    ap.add_argument("--output-csv", type=Path, default=None,
                    help="If set, write per-qid breakdown to this path")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        sys.exit("--inputs must list at least 2 evals")

    seed_data: dict[str, dict[str, dict[str, Any]]] = {}
    used_names: set[str] = set()
    for path in args.inputs:
        # Friendly seed name: parent dir (e.g. random_tools_seed43). If collision
        # (e.g. comparing same seed across models), include the grandparent too.
        name = path.parent.name
        if name in used_names:
            name = f"{path.parent.parent.name}/{name}"
        used_names.add(name)
        seed_data[name] = load_eval(path)

    seed_names = list(seed_data.keys())
    qid_sets = [set(d.keys()) for d in seed_data.values()]
    intersection = set.intersection(*qid_sets)
    union = set.union(*qid_sets)

    label = args.label or f"{len(seed_names)}-seed best-of-N"
    print(f"=== {label} ===")
    print(f"inputs: {len(seed_names)} eval files")
    for name, qids in zip(seed_names, qid_sets):
        print(f"  {name}: N={len(qids)}")
    print(f"qids in all seeds (intersection): {len(intersection)}")
    print(f"qids in any seed (union): {len(union)}")
    print()

    # Per-seed accuracy on the intersection
    print("Per-seed accuracy on intersection:")
    per_seed_correct: dict[str, int] = {}
    for name in seed_names:
        n_correct = sum(1 for qid in intersection if seed_data[name][qid].get("correct"))
        per_seed_correct[name] = n_correct
        print(f"  {name}: {n_correct}/{len(intersection)} = {100 * n_correct / max(len(intersection), 1):.2f}%")

    # pass@N: any seed correct counts
    n_pass_at_n = 0
    rows_for_csv: list[dict[str, Any]] = []
    for qid in sorted(intersection, key=lambda x: (len(x), x)):
        per_seed_flags = {name: bool(seed_data[name][qid].get("correct")) for name in seed_names}
        any_correct = any(per_seed_flags.values())
        if any_correct:
            n_pass_at_n += 1
        if args.output_csv:
            row = {"query_id": qid}
            for name in seed_names:
                row[name + "_correct"] = int(per_seed_flags[name])
                row[name + "_recall"] = seed_data[name][qid].get("recall")
                row[name + "_search_calls"] = seed_data[name][qid].get("num_search_calls")
            row["any_correct"] = int(any_correct)
            row["all_correct"] = int(all(per_seed_flags.values()))
            row["n_correct"] = sum(per_seed_flags.values())
            rows_for_csv.append(row)

    pass_at_n_pct = 100 * n_pass_at_n / max(len(intersection), 1)
    best_single = max(per_seed_correct.values()) if per_seed_correct else 0
    best_single_pct = 100 * best_single / max(len(intersection), 1)
    lift = pass_at_n_pct - best_single_pct

    print()
    print(f"pass@{len(seed_names)}: {n_pass_at_n}/{len(intersection)} = {pass_at_n_pct:.2f}%")
    print(f"best single seed:        {best_single}/{len(intersection)} = {best_single_pct:.2f}%")
    print(f"lift over best single:   {lift:+.2f}pp")

    # How many qids are uniquely solved by each seed (not solved by any of the others)
    print()
    print(f"Per-seed unique solves (correct in this seed AND wrong in all others):")
    for name in seed_names:
        others = [n for n in seed_names if n != name]
        unique = sum(
            1 for qid in intersection
            if seed_data[name][qid].get("correct")
            and not any(seed_data[o][qid].get("correct") for o in others)
        )
        print(f"  {name}: {unique}")

    if args.output_csv and rows_for_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows_for_csv[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_for_csv)
        print(f"\nWrote per-qid breakdown to {args.output_csv}")


if __name__ == "__main__":
    main()
