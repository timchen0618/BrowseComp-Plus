#!/usr/bin/env python3
"""Measure pairwise overlap of random_tool_calls JSONLs across multiple seeds.

For each query_id present in all input files, compute:
  - candidate_count: # candidate tool calls (same across seeds)
  - per-seed selected_indices set
  - mean pairwise Jaccard overlap across the seed sets
  - union size: # distinct tool calls used across all seeds (proxy for diversity)

Output:
  - per-query CSV (one row per query, one column per pairwise Jaccard)
  - summary stats: # queries with low/high overlap, # queries where all seeds picked the same calls

Usage:
    python scripts/validate_random_tool_calls_overlap.py \\
        --inputs selected_tool_calls/glm_random_tools_calls_seed42.jsonl \\
                 selected_tool_calls/glm_random_tools_calls_seed43.jsonl \\
                 selected_tool_calls/glm_random_tools_calls_seed44.jsonl \\
                 selected_tool_calls/glm_random_tools_calls_seed45.jsonl \\
        --output overlap_glm.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """qid -> row dict (skips error rows)."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{line_no} invalid JSON: {e}", file=sys.stderr)
                continue
            if row.get("error"):
                continue
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue
            out[qid] = row
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="Two or more random_tool_calls JSONL files (different seeds)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Per-query CSV output path (optional)")
    ap.add_argument("--high-overlap-threshold", type=float, default=0.6,
                    help="Mean pairwise Jaccard above which a query is 'high overlap' (default 0.6)")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        sys.exit("--inputs must list at least 2 files")

    seed_files: dict[str, dict[str, dict[str, Any]]] = {}
    for path in args.inputs:
        if not path.is_file():
            sys.exit(f"Not a file: {path}")
        seed_files[path.stem] = load_jsonl(path)

    common_qids = set.intersection(*(set(d.keys()) for d in seed_files.values()))
    seed_names = list(seed_files.keys())
    print(f"Inputs: {len(args.inputs)} files, {len(common_qids)} qids in common", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for qid in sorted(common_qids, key=lambda x: (len(x), x)):
        sel_sets: dict[str, set] = {}
        cand_count = None
        for sname, d in seed_files.items():
            r = d[qid]
            sel_sets[sname] = set(r.get("selected_indices", []))
            if cand_count is None:
                cand_count = r.get("candidate_count")
        pairwise = [jaccard(sel_sets[a], sel_sets[b]) for a, b in combinations(seed_names, 2)]
        mean_jac = sum(pairwise) / len(pairwise) if pairwise else 0.0
        union = set().union(*sel_sets.values())
        all_identical = all(s == sel_sets[seed_names[0]] for s in sel_sets.values())
        rows.append({
            "query_id": qid,
            "candidate_count": cand_count,
            "union_size": len(union),
            "diversity_ratio": (len(union) / cand_count) if cand_count else 0.0,
            "mean_pairwise_jaccard": round(mean_jac, 3),
            "all_identical": all_identical,
        })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}", file=sys.stderr)

    n = len(rows)
    high = sum(1 for r in rows if r["mean_pairwise_jaccard"] >= args.high_overlap_threshold)
    identical = sum(1 for r in rows if r["all_identical"])
    deg = sum(1 for r in rows if (r["candidate_count"] or 0) <= 5)
    overall_mean_jac = sum(r["mean_pairwise_jaccard"] for r in rows) / n if n else 0.0
    overall_diversity = sum(r["diversity_ratio"] for r in rows) / n if n else 0.0

    print()
    print(f"=== Overlap summary across {len(seed_names)} seeds ===")
    print(f"queries: {n}")
    print(f"mean pairwise Jaccard (all queries): {overall_mean_jac:.3f}")
    print(f"mean diversity ratio (union/candidates): {overall_diversity:.3f}")
    print(f"queries where all seeds picked identical sets: {identical} ({100*identical/n:.1f}%)")
    print(f"queries with mean Jaccard >= {args.high_overlap_threshold}: {high} ({100*high/n:.1f}%)")
    print(f"queries with <= 5 candidates (degenerate): {deg} ({100*deg/n:.1f}%)")
    print()

    # Print 5 highest-overlap queries that are NOT degenerate (interesting cases)
    interesting = [r for r in rows if (r["candidate_count"] or 0) > 5]
    interesting.sort(key=lambda r: -r["mean_pairwise_jaccard"])
    print("Highest-overlap non-degenerate queries (cand > 5):")
    for r in interesting[:5]:
        print(f"  qid={r['query_id']}  cand={r['candidate_count']}  jac={r['mean_pairwise_jaccard']:.2f}  union={r['union_size']}")


if __name__ == "__main__":
    main()
