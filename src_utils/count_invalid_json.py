#!/usr/bin/env python3
"""
Count invalid tool calls from tool_call_counts_all in trajectories.

Only "search" is treated as valid; all other tool types are invalid.
Categories: invalid_json, visit (keys starting with visit/Visit), others.
Reports average invalid count, average percentage, and breakdown by category.

Example:
    python src_utils/count_invalid_json.py runs_first100/ -r
    python src_utils/count_invalid_json.py runs_first100/Qwen3-Embedding-8B/mirothinker_rewrite/
"""

import json
import sys
from pathlib import Path
from typing import Iterator, Tuple
from collections import defaultdict


def find_trajectory_files(folder: Path, recursive: bool) -> Tuple[list[Path], list[Path]]:
    """Find .json and .jsonl files in the folder."""
    if recursive:
        json_files = sorted(p for p in folder.rglob("*.json") if p.is_file())
        jsonl_files = sorted(p for p in folder.rglob("*.jsonl") if p.is_file())
    else:
        json_files = sorted(p for p in folder.glob("*.json") if p.is_file())
        jsonl_files = sorted(p for p in folder.glob("*.jsonl") if p.is_file())
    return json_files, jsonl_files


def iter_trajectories_with_path(
    json_files: list[Path], jsonl_files: list[Path]
) -> Iterator[Tuple[Path, dict]]:
    """Yield (file_path, trajectory) from JSON and JSONL files."""
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield (path, json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    for path in jsonl_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield (path, json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue


def _key_to_category(key: str) -> str:
    """Map tool_call_counts_all keys to category: invalid_json, visit, or others."""
    if key == "invalid_json":
        return "invalid_json"
    if key.startswith("visit") or key.startswith("Visit"):
        return "visit"
    return "others"


def extract_invalid_stats(traj: dict) -> Tuple[int, int, dict[str, int]]:
    """
    Extract (invalid_count, total, per_category_counts) from tool_call_counts_all.
    Only "search" is valid; all other tool types are invalid.
    Categories: invalid_json, visit (keys starting with visit/Visit), others.
    Returns (0, 0, {}) if field is missing.
    """
    counts = traj.get("tool_call_counts_all")
    if not isinstance(counts, dict):
        return (0, 0, {})
    total = sum(int(v) for v in counts.values() if isinstance(v, (int, float)))
    by_category: dict[str, int] = {}
    for k, v in counts.items():
        if k == "search":
            continue
        if isinstance(v, (int, float)):
            cat = _key_to_category(k)
            by_category[cat] = by_category.get(cat, 0) + int(v)
    invalid = total - int(counts.get("search", 0))
    return (invalid, total, by_category)


def stats_from_pairs(pairs: list[Tuple[int, int]]) -> Tuple[int, float, float]:
    """Compute (n, avg_invalid, avg_pct) from a list of (invalid, total) pairs."""
    n = len(pairs)
    if n == 0:
        return (0, 0.0, 0.0)
    sum_invalid = sum(p[0] for p in pairs)
    pct_sum = 0.0
    pct_count = 0
    for invalid, total in pairs:
        if total > 0:
            pct_sum += (invalid / total) * 100
            pct_count += 1
    avg_invalid = sum_invalid / n
    avg_pct = (pct_sum / pct_count) if pct_count > 0 else 0.0
    return (n, avg_invalid, avg_pct)


def compute_stats(
    folder: Path,
    recursive: bool,
) -> None:
    """
    Compute invalid JSON stats per folder and overall, printing results.
    """
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory {folder} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files, jsonl_files = find_trajectory_files(folder, recursive)
    if not json_files and not jsonl_files:
        print(f"No trajectory files found in {folder}", file=sys.stderr)
        return

    # Group (invalid, total) and per-category counts by containing folder
    by_folder: dict[Path, list[Tuple[int, int]]] = defaultdict(list)
    by_folder_categories: dict[Path, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    folder_resolved = folder.resolve()

    for path, traj in iter_trajectories_with_path(json_files, jsonl_files):
        invalid, total, categories = extract_invalid_stats(traj)
        file_folder = path.parent.resolve()
        by_folder[file_folder].append((invalid, total))
        for k, v in categories.items():
            by_folder_categories[file_folder][k] += v

    # Build rows: (folder, n, avg_invalid, avg_pct, ij_avg, ij_pct, v_avg, v_pct, o_avg, o_pct)
    def row(folder_name: Path | str, pairs: list[Tuple[int, int]], cats: dict[str, int]) -> tuple:
        n, avg_invalid, avg_pct = stats_from_pairs(pairs)
        sum_total = sum(p[1] for p in pairs)
        def cat_avg(k: str) -> float:
            return cats.get(k, 0) / n if n > 0 else 0.0
        def cat_pct(k: str) -> float:
            return (cats.get(k, 0) / sum_total * 100) if sum_total > 0 else 0.0
        return (
            str(folder_name),
            n,
            avg_invalid,
            avg_pct,
            cat_avg("invalid_json"),
            cat_pct("invalid_json"),
            cat_avg("visit"),
            cat_pct("visit"),
            cat_avg("others"),
            cat_pct("others"),
        )

    rows: list[tuple] = []
    for file_folder in sorted(by_folder.keys()):
        pairs = by_folder[file_folder]
        cats = dict(by_folder_categories[file_folder])
        try:
            rel = file_folder.relative_to(folder_resolved)
        except ValueError:
            rel = file_folder
        rows.append(row(rel, pairs, cats))

    all_pairs: list[Tuple[int, int]] = []
    all_cats: dict[str, int] = defaultdict(int)
    for p in by_folder.values():
        all_pairs.extend(p)
    for cats in by_folder_categories.values():
        for k, v in cats.items():
            all_cats[k] += v
    rows.append(row(f"Overall ({folder})", all_pairs, dict(all_cats)))

    # Print table
    cols = (
        "Folder",
        "Traj",
        "Avg inv",
        "Avg inv %",
        "inv_json avg",
        "inv_json %",
        "visit avg",
        "visit %",
        "others avg",
        "others %",
    )
    w = (36, 6, 8, 10, 12, 11, 10, 9, 11, 10)
    sep = "+" + "+".join("-" * (wi + 2) for wi in w) + "+"
    hdr = "|" + "|".join(f" {str(c)[:wi]:^{wi}} " for c, wi in zip(cols, w)) + "|"

    def fmt_row(r: tuple) -> str:
        fs, n, ai, ap, ij_a, ij_p, v_a, v_p, o_a, o_p = r
        fs = (fs[:34] + "..") if len(fs) > 36 else fs
        return (
            f"| {fs:<36} | {n:>6} | {ai:>8.2f} | {ap:>9.2f}% | "
            f"{ij_a:>12.2f} | {ij_p:>10.2f}% | {v_a:>10.2f} | {v_p:>8.2f}% | {o_a:>11.2f} | {o_p:>9.2f}% |"
        )

    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Count invalid tool calls from tool_call_counts_all (only 'search' is valid)",
        epilog="Example: python src_utils/count_invalid_json.py runs_first100/ -r",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing trajectory (.json, .jsonl) files",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    args = parser.parse_args()

    compute_stats(args.folder, recursive=args.recursive)


if __name__ == "__main__":
    main()
