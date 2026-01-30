#!/usr/bin/env python3
"""
Remove trajectory JSON files whose top-level ``status`` equals a target value.

Mirrors the loading approach used in scripts_evaluation/deduplicate_trajectories.py:
each file is a single JSON object (not JSONL). If that object has status
`"exceed available llm calls"` (default), the file is deleted.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_STATUS = "exceed available llm calls"


def collect_trajectory_files(directory: Path) -> List[Path]:
    """Return all *.json trajectory files in the given directory."""
    return [p for p in directory.glob("*.json") if p.is_file()]


def file_has_status(path: Path, status: str) -> bool:
    """Return True if the file's top-level status equals the target."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == status and len(data.get("tool_call_counts", {})) == 0
    except Exception as exc:
        print(f"Skip {path}: {exc}")
        return False


def remove_files(files: Iterable[Path], dry_run: bool = False) -> None:
    """Remove files, honoring dry-run."""
    if not files:
        print("No files to remove.")
        return

    for path in files:
        if dry_run:
            print(f"[DRY RUN] Would delete {path}")
            continue
        try:
            path.unlink()
            print(f"Deleted {path}")
        except Exception as exc:
            print(f"Failed to delete {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete trajectory JSON files whose status matches a target value."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing trajectory JSON files (not recursive).",
    )
    parser.add_argument(
        "--status",
        default=DEFAULT_STATUS,
        help=f"Status value to match (default: '{DEFAULT_STATUS}').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would be deleted without removing them.",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    files = collect_trajectory_files(directory)
    if not files:
        print(f"No JSON files found in {directory}.")
        return

    to_delete = [p for p in files if file_has_status(p, args.status)]

    if not to_delete:
        print(f"No files with status '{args.status}' found in {directory}.")
        return

    print(f"Found {len(to_delete)} file(s) with status '{args.status}'.")
    remove_files(to_delete, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

