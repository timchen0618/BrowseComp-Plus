#!/usr/bin/env python3
"""
Deduplicate trajectory JSON files by qid, keeping the earliest file
(earliest filename interpreted as creation timestamp) and deleting later ones.

Usage:
    python deduplicat_trajectories.py /path/to/dir
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def collect_qids(directory: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Return mapping: qid -> list of (filename, path)."""
    qid_map: Dict[str, List[Tuple[str, Path]]] = {}
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            qid = data.get("query_id")
            if qid is None:
                continue  # skip files without qid
            qid_map.setdefault(qid, []).append((path.name, path))
        except Exception as exc:  # skip unreadable/invalid files
            print(f"Skip {path}: {exc}")
    return qid_map


def dedupe(directory: Path, dry_run: bool = False) -> None:
    """Delete later-created (by filename) duplicates per qid."""
    qid_map = collect_qids(directory)

    to_delete: List[Path] = []
    for qid, entries in qid_map.items():
        if len(entries) <= 1:
            continue
        # sort by filename ascending: earliest name first
        entries.sort(key=lambda t: t[0])
        keep = entries[0][1]
        drop_paths = [p for _, p in entries[1:]]
        to_delete.extend(drop_paths)
        print(f"qid {qid}: keep {keep.name}, delete {[p.name for p in drop_paths]}")

    if not to_delete:
        print("No duplicates found.")
        return

    if dry_run:
        print("Dry run: no files deleted.")
        return

    for path in to_delete:
        try:
            path.unlink()
        except Exception as exc:
            print(f"Failed to delete {path}: {exc}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate trajectory JSON files by qid.")
    parser.add_argument("directory", type=Path, help="Directory containing JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not delete")
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    dedupe(directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

