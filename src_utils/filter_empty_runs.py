#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ScanResult:
    empty_paths: list[Path]
    nonempty_paths: list[Path]


def _is_empty_run_payload(payload: dict[str, Any]) -> bool:
    """
    "Empty run" in this repo: trajectory JSON that contains only the initial
    system/user prompt and nothing else (no assistant reasoning, tool calls, etc).
    """
    result = payload.get("result", None)
    if not isinstance(result, list):
        return False
    if len(result) != 1:
        return False
    entry = result[0]
    if not isinstance(entry, dict):
        return False
    # Canonical empty: exactly one "user" entry which is the system prompt + question.
    if entry.get("type") != "user":
        return False

    # Strong additional signal: no tool calls were made.
    tcc = payload.get("tool_call_counts", None)
    if isinstance(tcc, dict) and len(tcc) != 0:
        return False

    return True


def _iter_json_files(input_dir: Path) -> Iterable[Path]:
    # Most runs are named run_*.json; we accept all *.json to be safe.
    for p in sorted(input_dir.glob("*.json")):
        if p.is_file():
            yield p


def scan_empty_runs(input_dir: Path) -> ScanResult:
    empty_paths: list[Path] = []
    nonempty_paths: list[Path] = []

    for path in _iter_json_files(input_dir):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            # If it's unreadable/corrupt JSON, treat it as non-empty so we don't drop data silently.
            nonempty_paths.append(path)
            continue

        if isinstance(payload, dict) and _is_empty_run_payload(payload):
            empty_paths.append(path)
        else:
            nonempty_paths.append(path)

    return ScanResult(empty_paths=empty_paths, nonempty_paths=nonempty_paths)


def _write_list(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


def _copy_paths(paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, out_dir / p.name)


def _move_paths(paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.move(str(p), str(out_dir / p.name))


def _delete_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter 'empty' trajectory JSON files (only the initial prompt) from run directories."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more run directories (e.g., .../seed4 .../seed5).",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "copy-nonempty", "move-empty", "delete-empty"],
        default="list",
        help=(
            "list: only report counts and optionally write lists. "
            "copy-nonempty: copy non-empty JSONs into --output-root/<seed_name>/. "
            "move-empty: move empty JSONs into --output-root/<seed_name>/. "
            "delete-empty: permanently delete empty JSONs (no output-root needed)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for copy/move outputs (required for copy-nonempty and move-empty).",
    )
    parser.add_argument(
        "--write-lists",
        action="store_true",
        help="Write empty/non-empty path lists under --output-root (or next to each input dir if output-root omitted).",
    )

    args = parser.parse_args()

    if args.mode in {"copy-nonempty", "move-empty"} and args.output_root is None:
        parser.error("--output-root is required for mode copy-nonempty or move-empty")

    total_empty = 0
    total_nonempty = 0

    for input_dir in args.input_dirs:
        scan = scan_empty_runs(input_dir)
        total_empty += len(scan.empty_paths)
        total_nonempty += len(scan.nonempty_paths)

        seed_name = input_dir.name
        print(f"{input_dir}: empty={len(scan.empty_paths)} nonempty={len(scan.nonempty_paths)}")

        if args.write_lists:
            if args.output_root is not None:
                lists_dir = args.output_root / "_lists"
            else:
                lists_dir = input_dir
            _write_list(scan.empty_paths, lists_dir / f"{seed_name}.empty.txt")
            _write_list(scan.nonempty_paths, lists_dir / f"{seed_name}.nonempty.txt")

        if args.mode == "copy-nonempty":
            assert args.output_root is not None
            _copy_paths(scan.nonempty_paths, args.output_root / seed_name)
        elif args.mode == "move-empty":
            assert args.output_root is not None
            _move_paths(scan.empty_paths, args.output_root / seed_name)
        elif args.mode == "delete-empty":
            _delete_paths(scan.empty_paths)

    print(f"TOTAL: empty={total_empty} nonempty={total_nonempty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

