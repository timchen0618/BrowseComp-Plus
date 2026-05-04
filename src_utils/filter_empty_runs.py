#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_TRAJ_TAGS = ("trajectory", "trajectory_summary")


@dataclass(frozen=True)
class ScanResult:
    empty_paths: list[Path]
    nonempty_paths: list[Path]
    empty_traj_content_paths: list[Path] = field(default_factory=list)


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


def _has_empty_trajectory_content(payload: dict[str, Any]) -> bool:
    """
    Returns True when the first user prompt contains a <trajectory> or
    <trajectory_summary> tag pair but the *second* occurrence of that pair
    (the actual injected content, not the template placeholder) is empty.

    The first pair is always the empty template reference in the instructions;
    the second pair should hold the real trajectory/summary text.
    """
    result = payload.get("result", None)
    if not isinstance(result, list) or not result:
        return False

    first_user_output: str | None = None
    for entry in result:
        if isinstance(entry, dict) and entry.get("type") == "user":
            out = entry.get("output", "")
            if isinstance(out, str):
                first_user_output = out
            break

    if first_user_output is None:
        return False

    for tag in _TRAJ_TAGS:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(first_user_output)
        if len(matches) < 2:
            continue
        # matches[0] is the empty template placeholder; matches[1] is the content.
        if not matches[1].strip():
            return True

    return False


def _get_leaf_dirs_with_json(root: Path) -> list[Path]:
    """Return dirs that directly contain *.json files; stop descending once found."""
    import os
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.endswith(".json") for f in filenames):
            results.append(Path(dirpath))
            dirnames.clear()
    return sorted(results)


def _iter_json_files(input_dir: Path) -> Iterable[Path]:
    # Most runs are named run_*.json; we accept all *.json to be safe.
    for p in sorted(input_dir.glob("*.json")):
        if p.is_file():
            yield p


def scan_empty_runs(input_dir: Path) -> ScanResult:
    empty_paths: list[Path] = []
    nonempty_paths: list[Path] = []
    empty_traj_content_paths: list[Path] = []

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
            if isinstance(payload, dict) and _has_empty_trajectory_content(payload):
                empty_traj_content_paths.append(path)
            nonempty_paths.append(path)

    return ScanResult(
        empty_paths=empty_paths,
        nonempty_paths=nonempty_paths,
        empty_traj_content_paths=empty_traj_content_paths,
    )


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
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Treat each --input-dirs entry as a root to walk recursively; scans every leaf dir containing JSON files.",
    )

    args = parser.parse_args()

    if args.mode in {"copy-nonempty", "move-empty"} and args.output_root is None:
        parser.error("--output-root is required for mode copy-nonempty or move-empty")

    # Expand input dirs to leaf dirs when -r is set.
    input_dirs: list[Path] = []
    for d in args.input_dirs:
        if args.recursive:
            leaves = _get_leaf_dirs_with_json(d)
            if not leaves:
                print(f"WARNING: no JSON-containing subdirs found under {d}")
            input_dirs.extend(leaves)
        else:
            input_dirs.append(d)

    if args.recursive:
        print(f"Found {len(input_dirs)} leaf directories with JSON files")

    total_empty = 0
    total_nonempty = 0
    total_empty_traj = 0

    for input_dir in input_dirs:
        scan = scan_empty_runs(input_dir)
        total_empty += len(scan.empty_paths)
        total_nonempty += len(scan.nonempty_paths)
        total_empty_traj += len(scan.empty_traj_content_paths)

        seed_name = input_dir.name
        print(
            f"{input_dir}: empty={len(scan.empty_paths)}"
            f" nonempty={len(scan.nonempty_paths)}"
            f" empty_traj_content={len(scan.empty_traj_content_paths)}"
        )
        for p in scan.empty_traj_content_paths:
            print(f"  [empty traj content] {p}")

        if args.write_lists:
            if args.output_root is not None:
                lists_dir = args.output_root / "_lists"
            else:
                lists_dir = input_dir
            _write_list(scan.empty_paths, lists_dir / f"{seed_name}.empty.txt")
            _write_list(scan.nonempty_paths, lists_dir / f"{seed_name}.nonempty.txt")
            _write_list(
                scan.empty_traj_content_paths,
                lists_dir / f"{seed_name}.empty_traj_content.txt",
            )

        if args.mode == "copy-nonempty":
            assert args.output_root is not None
            _copy_paths(scan.nonempty_paths, args.output_root / seed_name)
        elif args.mode == "move-empty":
            assert args.output_root is not None
            _move_paths(scan.empty_paths, args.output_root / seed_name)
        elif args.mode == "delete-empty":
            _delete_paths(scan.empty_paths)

    print(f"TOTAL: empty={total_empty} nonempty={total_nonempty} empty_traj_content={total_empty_traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

