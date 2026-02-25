#!/usr/bin/env python3
"""
Compute the average time_taken for trajectory JSON files in a directory.

Each file is expected to be a single JSON object (same convention as
scripts_evaluation/deduplicate_trajectories.py) and to store a numeric
value at data["meta_data"]["time_taken"].
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple


def collect_files(directory: Path) -> List[Path]:
    """Return all *.json trajectory files in the directory (non-recursive)."""
    return [p for p in directory.glob("*.json") if p.is_file()]


def extract_time_taken(path: Path) -> Optional[Tuple[float, float, float]]:
    """Load a trajectory file and return meta_data.time_taken if present and numeric."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata") or {}
        value = meta.get("time_taken")
        total_inference_time = meta.get("total_inference_time")
        total_tool_call_time = meta.get("total_tool_call_time")
        
        if isinstance(value, (int, float)):
            return float(value), float(total_inference_time), float(total_tool_call_time)
        return None, None, None
    except Exception as exc:
        print(f"Skip {path}: {exc}")
        return None, None, None
    

def extract_tool_call_counts(path: Path) -> int:
    """Load a trajectory file and return meta_data.time_taken if present and numeric."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return min(int(data.get("tool_call_counts", {}).get("search", 0)), 20)
    except Exception as exc:
        print(f"Skip {path}: {exc}")
        return 0.0


def compute_average(times: List[float]) -> Tuple[int, float]:
    """Return count and average of the provided times."""
    if not times:
        return 0, 0.0
    total = sum(times)
    return len(times), total / len(times)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average meta_data.time_taken for trajectory JSON files in a folder."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing trajectory JSON files (not recursive).",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    files = collect_files(directory)
    if not files:
        print(f"No JSON files found in {directory}.")
        return

    times: List[float] = []
    total_inference_times: List[float] = []
    total_tool_call_times: List[float] = []
    missing = 0
    for path in files:
        value, total_inference_time, total_tool_call_time = extract_time_taken(path)
        if value is None or total_inference_time is None or total_tool_call_time is None:
            missing += 1
            continue
        times.append(value)
        total_inference_times.append(total_inference_time)
        total_tool_call_times.append(total_tool_call_time)

    count, avg = compute_average(times)
    total_inference_count, total_inference_avg = compute_average(total_inference_times)
    total_tool_call_count, total_tool_call_avg = compute_average(total_tool_call_times)
    print(f"Scanned {len(files)} file(s) in {directory}.")
    print(f"Found time_taken in {count} file(s); missing/invalid: {missing}.")
    if count == 0:
        print("Average time_taken: N/A")
    else:
        print(f"Average time_taken: {avg:.4f}")
        print(f"Average total_inference_time: {total_inference_avg:.4f}")
        print(f"Average total_tool_call_time: {total_tool_call_avg:.4f}")
        
        
        
    tool_call_counts: List[int] = []
    missing = 0
    for path in files
        tool_call_count = extract_tool_call_counts(path)
        if tool_call_count is None:
            missing += 1
            continue
        tool_call_counts.append(tool_call_count)
    count, avg = compute_average(tool_call_counts)
    print(f"Scanned {len(files)} file(s) in {directory}.")
    print(f"Found tool_call_counts in {count} file(s); missing/invalid: {missing}.")
    if count == 0:
        print("Average tool_call_counts: N/A")
    else:
        print(f"Average tool_call_counts: {avg:.4f}")


if __name__ == "__main__":
    main()

