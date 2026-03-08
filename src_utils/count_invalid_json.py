#!/usr/bin/env python3
"""
Count invalid tool calls from tool_call_counts_all in trajectories.

Only "search" is treated as valid; all other tool types are invalid.
Categories: invalid_json, visit (keys starting with visit/Visit), others.
Reports average invalid count, average percentage, and breakdown by category.

Example:
    python src_utils/count_invalid_json.py runs_first100/ -r
    python src_utils/count_invalid_json.py runs_first100/ -r -o invalid_stats.csv
"""

import csv
import json
import sys
from pathlib import Path
from typing import Iterator, Tuple
from collections import defaultdict


def find_trajectory_files(folder: Path, recursive: bool) -> Tuple[list[Path], list[Path]]:
    """Find .json and .jsonl files in the folder. Single walk when recursive to avoid 2x tree traversal."""
    if recursive:
        json_files: list[Path] = []
        jsonl_files: list[Path] = []
        for p in folder.rglob("*"):
            if p.is_file():
                if p.suffix == ".jsonl":
                    jsonl_files.append(p)
                elif p.suffix == ".json":
                    json_files.append(p)
        json_files.sort()
        jsonl_files.sort()
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


def stats_from_records(
    records: list[Tuple[int, int, int, int, int]]
) -> Tuple[int, float, float, float, float, float]:
    """
    Compute stats from per-trajectory records (invalid, total, ij, visit, others).
    All percentages use mean of per-trajectory ratios so each trajectory contributes equally.
    Returns (n, avg_invalid, avg_pct, ij_pct, visit_pct, others_pct).
    """
    n = len(records)
    if n == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sum_invalid = sum(r[0] for r in records)
    pct_vals = []
    ij_pct_vals = []
    v_pct_vals = []
    o_pct_vals = []
    for invalid, total, ij, v, o in records:
        if total > 0:
            pct_vals.append((invalid / total) * 100)
            ij_pct_vals.append((ij / total) * 100)
            v_pct_vals.append((v / total) * 100)
            o_pct_vals.append((o / total) * 100)
    avg_invalid = sum_invalid / n
    avg_pct = sum(pct_vals) / len(pct_vals) if pct_vals else 0.0
    ij_pct = sum(ij_pct_vals) / len(ij_pct_vals) if ij_pct_vals else 0.0
    v_pct = sum(v_pct_vals) / len(v_pct_vals) if v_pct_vals else 0.0
    o_pct = sum(o_pct_vals) / len(o_pct_vals) if o_pct_vals else 0.0
    return (n, avg_invalid, avg_pct, ij_pct, v_pct, o_pct)


def compute_stats(
    folder: Path,
    recursive: bool,
    output_csv: Path | None = None,
) -> None:
    """
    Compute invalid JSON stats per folder and overall, printing results.
    If output_csv is set, also write the data to that CSV file.
    """
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory {folder} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files, jsonl_files = find_trajectory_files(folder, recursive)
    if not json_files and not jsonl_files:
        print(f"No trajectory files found in {folder}", file=sys.stderr)
        return

    # Group per-trajectory records by containing folder
    # Record: (invalid, total, invalid_json, visit, others)
    by_folder: dict[Path, list[Tuple[int, int, int, int, int]]] = defaultdict(list)
    folder_resolved = folder.resolve()

    for path, traj in iter_trajectories_with_path(json_files, jsonl_files):
        invalid, total, categories = extract_invalid_stats(traj)
        ij = categories.get("invalid_json", 0)
        v = categories.get("visit", 0)
        o = categories.get("others", 0)
        file_folder = path.parent.resolve()
        by_folder[file_folder].append((invalid, total, ij, v, o))

    # Build rows: (folder, n, avg_invalid, avg_pct, ij_avg, ij_pct, v_avg, v_pct, o_avg, o_pct)
    def row(folder_name: Path | str, records: list[Tuple[int, int, int, int, int]]) -> tuple:
        n, avg_invalid, avg_pct, ij_pct, v_pct, o_pct = stats_from_records(records)
        ij_avg = sum(r[2] for r in records) / n if n > 0 else 0.0
        v_avg = sum(r[3] for r in records) / n if n > 0 else 0.0
        o_avg = sum(r[4] for r in records) / n if n > 0 else 0.0
        return (
            str(folder_name),
            n,
            avg_invalid,
            avg_pct,
            ij_avg,
            ij_pct,
            v_avg,
            v_pct,
            o_avg,
            o_pct,
        )

    rows: list[tuple] = []
    for file_folder in sorted(by_folder.keys()):
        records = by_folder[file_folder]
        try:
            rel = file_folder.relative_to(folder_resolved)
        except ValueError:
            rel = file_folder
        rows.append(row(rel, records))

    all_records: list[Tuple[int, int, int, int, int]] = []
    for recs in by_folder.values():
        all_records.extend(recs)
    rows.append(row(f"Overall ({folder})", all_records))

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

    if output_csv:
        csv_cols = [
            "Folder", "Traj", "Avg inv", "Avg inv %",
            "inv_json avg", "inv_json %", "visit avg", "visit %",
            "others avg", "others %",
        ]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_cols)
            for r in rows:
                fs, n, ai, ap, ij_a, ij_p, v_a, v_p, o_a, o_p = r
                writer.writerow([
                    fs, n,
                    round(ai, 2), round(ap, 2),
                    round(ij_a, 2), round(ij_p, 2),
                    round(v_a, 2), round(v_p, 2),
                    round(o_a, 2), round(o_p, 2),
                ])
        print(f"\nWrote {output_csv}", file=sys.stderr)


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
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write results to CSV file",
    )
    args = parser.parse_args()

    compute_stats(args.folder, recursive=args.recursive, output_csv=args.output)


if __name__ == "__main__":
    main()
