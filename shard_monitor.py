#!/usr/bin/env python3
"""
Shard Monitor — autonomous shard completeness checker and resubmission generator.

Reads shard TSV files to determine expected query IDs, scans output directories
for completed results, reports gaps, and generates SLURM resubmission commands.

Usage:
    # Report completeness for all runs under a given path
    python shard_monitor.py bcp Qwen3-Embedding-8B full gpt-oss-120b

    # Verify after resubmission (same args + --verify)
    python shard_monitor.py bcp Qwen3-Embedding-8B full gpt-oss-120b --verify

    # Generate submit_missing.py entries (copy-paste ready)
    python shard_monitor.py bcp Qwen3-Embedding-8B full gpt-oss-120b --gen-submit

    # Only show incomplete runs
    python shard_monitor.py bcp Qwen3-Embedding-8B full gpt-oss-120b --incomplete-only
"""

import argparse
import json
import os
import sys
from collections import defaultdict


# ── Shard directory discovery ────────────────────────────────────────────────

SHARD_DIR_PATTERNS = {
    # dataset -> (shard_dir_relative_to topics-qrels/{dataset}/, shard_file_glob)
    # Discovered automatically from topics-qrels/{dataset}/{dataset}_*_shards/
}


def find_shard_dir(dataset: str) -> str | None:
    """Find the shard directory for a dataset under topics-qrels/."""
    base = os.path.join("topics-qrels", dataset)
    if not os.path.isdir(base):
        return None
    for entry in os.listdir(base):
        if entry.endswith("_shards") and os.path.isdir(os.path.join(base, entry)):
            return os.path.join(base, entry)
    return None


def load_shard_query_ids(shard_dir: str) -> dict[str, set[str]]:
    """Load query IDs from each shard TSV file.

    Returns: {shard_name: {query_id, ...}, ...}
      e.g. {"q_0": {"769", "770", ...}, "q_1": {"231", "233", ...}}
    """
    shards = {}
    for fname in sorted(os.listdir(shard_dir)):
        if not fname.endswith(".tsv"):
            continue
        shard_name = fname.replace(".tsv", "")
        ids = set()
        with open(os.path.join(shard_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    qid = line.split("\t")[0]
                    ids.add(qid)
        shards[shard_name] = ids
    return shards


def get_all_expected_ids(shards: dict[str, set[str]]) -> set[str]:
    """Union of all query IDs across all shards."""
    all_ids = set()
    for ids in shards.values():
        all_ids |= ids
    return all_ids


def load_split_query_ids(dataset: str, split: str) -> set[str] | None:
    """Load query IDs for a specific split (first50, first100, etc.).

    Returns None if split is 'full' or no split file exists (use all shard IDs).
    """
    if split == "full":
        return None
    # Try queries_{split}.tsv (e.g. queries_first50.tsv)
    candidates = [
        os.path.join("topics-qrels", dataset, f"queries_{split}.tsv"),
        os.path.join("topics-qrels", dataset, f"queries_{split.replace('_', '')}.tsv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            ids = set()
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.add(line.split("\t")[0])
            return ids
    return None


def filter_shards_by_split(
    shards: dict[str, set[str]], split_ids: set[str]
) -> dict[str, set[str]]:
    """Filter shard query IDs to only include those in the split."""
    filtered = {}
    for shard_name, ids in shards.items():
        intersection = ids & split_ids
        if intersection:
            filtered[shard_name] = intersection
    return filtered


# ── Output scanning ─────────────────────────────────────────────────────────

def scan_completed_ids(run_dir: str) -> set[str]:
    """Scan a run directory for completed query IDs from output JSON files.

    Reads in chunks to find query_id, which may appear far into the file
    when a large metadata object precedes it.
    """
    completed = set()
    if not os.path.isdir(run_dir):
        return completed
    import re
    qid_pattern = re.compile(r'"query_id"\s*:\s*"?(\w+)"?')
    for fname in os.listdir(run_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(run_dir, fname)
        try:
            with open(fpath) as f:
                # Read in growing chunks until query_id is found.
                # Start at 1KB, double up to 64KB, then read the rest.
                buf = ""
                for chunk_size in (1024, 1024, 2048, 4096, 8192, 16384, 32768):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    buf += chunk
                    m = qid_pattern.search(buf)
                    if m:
                        completed.add(m.group(1))
                        break
                else:
                    # Fell through — read remaining content
                    buf += f.read()
                    m = qid_pattern.search(buf)
                    if m:
                        completed.add(m.group(1))
        except OSError:
            continue
    return completed


# ── Reporting ────────────────────────────────────────────────────────────────

def compute_shard_gaps(
    shards: dict[str, set[str]],
    completed: set[str],
) -> dict[str, set[str]]:
    """For each shard, compute which query IDs are missing."""
    gaps = {}
    for shard_name, expected_ids in shards.items():
        missing = expected_ids - completed
        if missing:
            gaps[shard_name] = missing
    return gaps


def print_run_report(
    run_name: str,
    total_expected: int,
    completed_count: int,
    shard_gaps: dict[str, set[str]],
    verbose: bool = False,
):
    """Print a completeness report for a single run."""
    missing_count = total_expected - completed_count
    pct = (completed_count / total_expected * 100) if total_expected > 0 else 0
    status = "COMPLETE" if missing_count == 0 else "INCOMPLETE"
    marker = " " if missing_count == 0 else "!"

    print(f"  {marker} {run_name}: {completed_count}/{total_expected} ({pct:.1f}%) [{status}]")

    if missing_count > 0 and verbose:
        incomplete_shards = sorted(shard_gaps.keys(), key=lambda s: int(s.split("_")[1]))
        print(f"    Incomplete shards: {', '.join(incomplete_shards)}")
        for shard in incomplete_shards:
            missing_ids = sorted(shard_gaps[shard], key=lambda x: int(x) if x.isdigit() else x)
            print(f"      {shard}: {len(shard_gaps[shard])} missing — {', '.join(missing_ids[:20])}"
                  + ("..." if len(missing_ids) > 20 else ""))


def generate_submit_entries(
    run_name: str,
    shard_gaps: dict[str, set[str]],
) -> list[int]:
    """Return list of incomplete shard indices for SLURM array jobs."""
    indices = []
    for shard_name in shard_gaps:
        # Extract shard index from name like "q_3"
        parts = shard_name.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            indices.append(int(parts[-1]))
    return sorted(indices)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monitor shard completeness and generate resubmission commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help="Dataset name (e.g. bcp, frames, musique)")
    parser.add_argument("retriever", help="Retriever name (e.g. Qwen3-Embedding-8B)")
    parser.add_argument("split", help="Split name (e.g. full, first50, first100)")
    parser.add_argument("agent_model", help="Agent model dir (e.g. gpt-oss-120b, tongyi)")
    parser.add_argument("--verify", action="store_true",
                        help="Verification mode: stricter reporting, exit 1 if any gaps remain")
    parser.add_argument("--gen-submit", action="store_true",
                        help="Generate MISSING dict entries for submit_missing.py")
    parser.add_argument("--incomplete-only", action="store_true",
                        help="Only show runs with missing queries")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-shard missing query IDs")
    parser.add_argument("--runs-dir", default="runs",
                        help="Base runs directory (default: runs)")

    args = parser.parse_args()

    # 1. Find and load shard files
    shard_dir = find_shard_dir(args.dataset)
    if shard_dir is None:
        print(f"ERROR: No shard directory found for dataset '{args.dataset}' "
              f"under topics-qrels/{args.dataset}/")
        sys.exit(1)

    shards = load_shard_query_ids(shard_dir)

    # Filter by split if not 'full'
    split_ids = load_split_query_ids(args.dataset, args.split)
    if split_ids is not None:
        shards = filter_shards_by_split(shards, split_ids)
        all_expected = split_ids
    else:
        all_expected = get_all_expected_ids(shards)

    print(f"Dataset: {args.dataset} | Retriever: {args.retriever} | "
          f"Split: {args.split} | Agent: {args.agent_model}")
    print(f"Shard dir: {shard_dir} ({len(shards)} shards, {len(all_expected)} expected queries)")
    print()

    # 2. Scan run directories
    base_path = os.path.join(args.runs_dir, args.dataset, args.retriever,
                             args.split, args.agent_model)
    if not os.path.isdir(base_path):
        print(f"ERROR: Run directory does not exist: {base_path}")
        print(f"Available under {os.path.dirname(base_path)}:")
        parent = os.path.dirname(base_path)
        if os.path.isdir(parent):
            for d in sorted(os.listdir(parent)):
                if os.path.isdir(os.path.join(parent, d)):
                    print(f"  {d}")
        sys.exit(1)

    run_names = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

    # Handle flat structure: JSON files directly in agent_model dir (no run_name subdirs)
    flat_mode = False
    if not run_names:
        json_files = [f for f in os.listdir(base_path) if f.endswith(".json")]
        if json_files:
            flat_mode = True
            run_names = [args.agent_model]  # treat the dir itself as one run
        else:
            print(f"No run directories or JSON files found under {base_path}")
            sys.exit(1)

    print(f"Found {len(run_names)} run{'s' if len(run_names) != 1 else ''} under {base_path}"
          + (" (flat layout)" if flat_mode else ""))
    print("-" * 70)

    # 3. Analyze each run
    total_complete = 0
    total_incomplete = 0
    submit_entries = {}

    for run_name in run_names:
        run_dir = base_path if flat_mode else os.path.join(base_path, run_name)
        completed = scan_completed_ids(run_dir)
        shard_gaps = compute_shard_gaps(shards, completed)
        missing_count = len(all_expected) - len(completed & all_expected)

        if args.incomplete_only and missing_count == 0:
            total_complete += 1
            continue

        print_run_report(
            run_name,
            len(all_expected),
            len(completed & all_expected),
            shard_gaps,
            verbose=args.verbose or args.verify,
        )

        if missing_count == 0:
            total_complete += 1
        else:
            total_incomplete += 1
            indices = generate_submit_entries(run_name, shard_gaps)
            if indices:
                submit_entries[run_name] = indices

    # 4. Summary
    print("-" * 70)
    shown = total_incomplete if args.incomplete_only else total_complete + total_incomplete
    print(f"Summary: {total_complete} complete, {total_incomplete} incomplete "
          f"(out of {total_complete + total_incomplete} runs)")

    # 5. Generate submit_missing.py entries
    if args.gen_submit and submit_entries:
        print()
        print("# ── Copy into submit_missing.py MISSING dict ──")
        print("MISSING = {")
        for run_name, indices in sorted(submit_entries.items()):
            if set(indices) == set(range(len(shards))):
                print(f'    "{run_name}": list(range({len(shards)})),')
            else:
                print(f'    "{run_name}": {indices},')
        print("}")

    # 6. Verify mode exit code
    if args.verify:
        if total_incomplete > 0:
            print(f"\nVERIFICATION FAILED: {total_incomplete} runs still have gaps.")
            sys.exit(1)
        else:
            print("\nVERIFICATION PASSED: All runs complete.")
            sys.exit(0)


if __name__ == "__main__":
    main()
