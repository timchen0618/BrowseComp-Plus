#!/usr/bin/env python3
"""
Parse output from find_missing_ids.py and generate a MISSING dict
suitable for pasting into submit_missing.py.

Usage:
    # Recursive mode (auto-detects run names from "--- Checking" headers)
    python src_utils/find_missing_ids.py --recursive \
      --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b \
      --reference_file "topics-qrels/bcp/bcp_10_shards/*" \
      | python scripts/update_submit_missing.py

    # Single-directory mode (provide --run-name)
    python src_utils/find_missing_ids.py \
      --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/planning_v4_seed0 \
      --reference_file "topics-qrels/bcp/bcp_10_shards/*" \
      | python scripts/update_submit_missing.py --run-name planning_v4_seed0

    # From a saved file
    python scripts/update_submit_missing.py --input missing_output.txt --run-name planning_v4_seed0
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_missing_output(lines: list[str], run_name: str | None = None) -> dict[str, list[int]]:
    """Parse find_missing_ids.py stdout into {run_name: [shard_indices]}.

    Handles two formats:

    1. --recursive mode (multiple runs, each with a header):
        --- Checking runs/.../some_run_seed0 (75 query ids) ---
        ***Missing 5 query ids in topics-qrels/bcp/bcp_10_shards/q_3.tsv:***
         ['Q123', 'Q456', ...]

    2. Single-directory mode (no headers, just missing lines):
        Found 828 output run query ids
        ***Missing 1 query ids in topics-qrels/bcp/bcp_10_shards/q_3.tsv:***
         ['262']

       For this mode, pass run_name explicitly or use --run-name on the CLI.
    """
    result: dict[str, list[int]] = defaultdict(list)
    current_run = run_name

    checking_re = re.compile(r"^--- Checking (.+?) \(\d+ query ids?\) ---$")
    missing_re = re.compile(r"^\*\*\*Missing \d+ query ids? in .+/q_(\d+)\.tsv:\*\*\*$")
    complete_re = re.compile(r"^All query files complete")

    for line in lines:
        line = line.rstrip()

        m = checking_re.match(line)
        if m:
            path = m.group(1)
            current_run = path.rstrip("/").rsplit("/", 1)[-1]
            continue

        if complete_re.match(line):
            if run_name is None:
                current_run = None
            continue

        m = missing_re.match(line)
        if m and current_run is not None:
            shard_idx = int(m.group(1))
            if shard_idx not in result[current_run]:
                result[current_run].append(shard_idx)
            continue

    for shards in result.values():
        shards.sort()

    return dict(result)


def format_missing_dict(missing: dict[str, list[int]], var_name: str = "MISSING") -> str:
    """Format the dict as a Python source snippet."""
    if not missing:
        return f"{var_name} = {{}}\n"

    lines = [f"{var_name} = {{"]
    max_key_len = max(len(k) for k in missing)
    for run_name, shards in sorted(missing.items()):
        padding = " " * (max_key_len - len(run_name))
        if shards == list(range(10)):
            val = "list(range(10))"
        else:
            val = repr(shards)
        lines.append(f'    "{run_name}": {padding}{val},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Parse find_missing_ids.py output and generate a MISSING dict for submit_missing.py."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to saved find_missing_ids.py output. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name for single-directory (non-recursive) output. "
             "If omitted, inferred from the last component of --input-dir when present in the output.",
    )
    parser.add_argument(
        "--var-name", type=str, default="MISSING",
        help="Variable name for the generated dict (default: MISSING).",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            lines = f.readlines()
    else:
        if sys.stdin.isatty():
            print("Reading from stdin (pipe find_missing_ids.py output or use --input)...",
                  file=sys.stderr)
        lines = sys.stdin.readlines()

    run_name = args.run_name
    if run_name is None:
        # Auto-detect from "Found N output run query ids from ..." or
        # check if any "--- Checking" headers exist (recursive mode).
        has_checking_headers = any(
            line.lstrip().startswith("--- Checking ") for line in lines
        )
        if not has_checking_headers:
            # Single-directory mode — try to find input_dir from the command
            # that produced this output. Fall back to a placeholder.
            run_name = "UNKNOWN_RUN"
            print("Warning: no --run-name given and no recursive headers found. "
                  "Using placeholder run name. Re-run with --run-name <name>.",
                  file=sys.stderr)

    missing = parse_missing_output(lines, run_name=run_name)

    if not missing:
        print("No missing shards found — all runs are complete.", file=sys.stderr)
        return

    total_shards = sum(len(s) for s in missing.values())
    print(f"# {len(missing)} runs with {total_shards} missing shards total\n",
          file=sys.stderr)

    print(format_missing_dict(missing, var_name=args.var_name))


if __name__ == "__main__":
    main()
