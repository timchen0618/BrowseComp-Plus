#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            yield obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter selected_tool_calls JSONL by dropping specified error records.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--drop-error",
        action="append",
        default=[],
        help='Repeatable. Drop rows whose `error` equals this string (e.g. "no_candidate_tool_calls").',
    )
    ap.add_argument(
        "--write-dropped",
        type=Path,
        default=None,
        help="Optional JSONL path to write dropped rows (full row).",
    )
    args = ap.parse_args()

    drop_set = set(args.drop_error or [])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_dropped is not None:
        args.write_dropped.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0

    dropped_f = args.write_dropped.open("w", encoding="utf-8") if args.write_dropped is not None else None
    try:
        with args.output.open("w", encoding="utf-8") as out:
            for row in _iter_jsonl(args.input):
                err = row.get("error", None)
                if isinstance(err, str) and err in drop_set:
                    dropped += 1
                    if dropped_f is not None:
                        dropped_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    finally:
        if dropped_f is not None:
            dropped_f.close()

    print(f"{args.input}: kept={kept} dropped={dropped} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

