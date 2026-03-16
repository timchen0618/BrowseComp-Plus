#!/usr/bin/env python3
"""
Remove trajectory JSON files whose top-level ``status`` equals a target value.

Mirrors the loading approach used in scripts_evaluation/deduplicate_trajectories.py:
each file is a single JSON object (not JSONL). If that object has status
`"exceed available llm calls"` (default), the file is deleted.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List


# DEFAULT_STATUS = "exceed available llm calls"
DEFAULT_STATUS = "incomplete"


def get_leaf_dirs_with_json(root_path: Path) -> List[Path]:
    """Find directories that directly contain JSON files. Stops descending when such dirs are found."""
    results = []
    for dirpath, dirnames, filenames in os.walk(str(root_path)):
        if any(f.endswith(".json") for f in filenames):
            results.append(Path(dirpath))
            dirnames.clear()  # Don't recurse into subdirs
    return sorted(results)


def collect_trajectory_files(directory: Path) -> List[Path]:
    """Return all *.json trajectory files in the given directory."""
    return [p for p in directory.glob("*.json") if p.is_file()]


def file_has_status(path: Path, status: str) -> bool:
    """Return True if the file's top-level status equals the target."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # if data.get("status") == status and len(data.get("tool_call_counts", {})) == 0:
        #     print(data)
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
        help="Directory containing trajectory JSON files; with --recursive, root to walk.",
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
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process every subfolder; stops at dirs containing JSON files.",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    if args.recursive:
        leaf_dirs = get_leaf_dirs_with_json(directory)
        if not leaf_dirs:
            print(f"No leaf directories with JSON files found under {directory}.")
            return
        print(f"Found {len(leaf_dirs)} leaf directory(ies) with JSON files.")
        total_deleted = 0
        for leaf_dir in leaf_dirs:
            print(f"\n--- Processing {leaf_dir} ---")
            files = collect_trajectory_files(leaf_dir)
            if not files:
                continue
            to_delete = [p for p in files if file_has_status(p, args.status)]
            if not to_delete:
                continue
            print(f"\n--- {leaf_dir} ({len(to_delete)} file(s) with status '{args.status}') ---")
            remove_files(to_delete, dry_run=args.dry_run)
            total_deleted += len(to_delete)
        print(f"\nTotal: {total_deleted} file(s) {'would be ' if args.dry_run else ''}deleted.")
        return

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

