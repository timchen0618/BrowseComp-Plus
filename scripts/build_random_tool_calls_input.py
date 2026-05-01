#!/usr/bin/env python3
"""Build input JSONL for random_select_tool_calls.py from a trajectory directory.

random_select_tool_calls.py needs an --input-jsonl file with one row per query
that has at minimum {query_id, source_file}. When the Gemini-selected JSONL
already exists you can reuse that as the input (it has source_file mappings),
but for benchmarks/splits where Gemini selection has not been run yet (e.g.,
FRAMES) we synthesize the input by listing the trajectory directory.

Each input row has:
    {"query_id": <str>, "source_file": <basename.json>}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory-dir", type=Path, required=True,
                    help="Directory containing trajectory *.json files")
    ap.add_argument("--output", type=Path, required=True,
                    help="Path to write input JSONL")
    args = ap.parse_args()

    if not args.trajectory_dir.is_dir():
        sys.exit(f"Not a directory: {args.trajectory_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for fn in sorted(args.trajectory_dir.iterdir()):
            if fn.suffix != ".json":
                continue
            try:
                obj = json.loads(fn.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"[warn] skipping {fn.name}: {e}", file=sys.stderr)
                continue
            qid = str(obj.get("query_id", "")).strip()
            if not qid:
                print(f"[warn] skipping {fn.name}: missing query_id", file=sys.stderr)
                continue
            fout.write(json.dumps({"query_id": qid, "source_file": fn.name}) + "\n")
            n += 1
    print(f"Wrote {args.output} ({n} rows)")


if __name__ == "__main__":
    main()
