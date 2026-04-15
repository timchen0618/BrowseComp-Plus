#!/usr/bin/env python3
"""
Augment selected_tool_calls.jsonl with:
  - candidates: candidate tool-call indices reconstructed from trajectories
  - correct_num_selected: boolean computed from selected_indices vs k_effective and candidates

This updates the file in-place, but writes a .bak backup first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_correct_num_selected(
    row: Dict[str, Any],
    *,
    candidates: List[int],
) -> bool:
    sel = row.get("selected_indices")
    if not isinstance(sel, list) or not all(_is_int(x) for x in sel):
        return False
    cand_set = set(candidates)
    if any(i not in cand_set for i in sel):
        return False
    k_eff = row.get("k_effective")
    if _is_int(k_eff):
        return len(set(sel)) == int(k_eff)
    # If k_effective missing, we can only validate subset-of-candidates.
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected-jsonl",
        default="selected_tool_calls/selected_tool_calls.jsonl",
        help="Path to selected_tool_calls.jsonl to modify in-place.",
    )
    ap.add_argument(
        "--seed0-dir",
        default="runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed0",
        help="Directory containing seed0 trajectories (JSON).",
    )
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix to write before replacing the file.",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from select_useful_tool_calls import DEFAULT_TOOL_NAMES, find_candidate_tool_indices  # noqa: WPS433

    allowed_tool_names: Set[str] = set(DEFAULT_TOOL_NAMES)
    selected_path = Path(args.selected_jsonl)
    seed0_dir = Path(args.seed0_dir)
    backup_path = selected_path.with_name(selected_path.name + args.backup_suffix)
    tmp_path = selected_path.with_name(selected_path.name + ".tmp")

    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not seed0_dir.exists():
        raise FileNotFoundError(seed0_dir)

    # Backup (only if not already present)
    if not backup_path.exists():
        shutil.copy2(selected_path, backup_path)

    n_total = 0
    n_augmented = 0
    n_missing_traj = 0
    n_traj_load_err = 0
    n_parse_err = 0

    with selected_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except Exception:
                n_parse_err += 1
                continue

            if not isinstance(row, dict):
                continue

            source_file = str(row.get("source_file", ""))
            if not source_file:
                # Preserve row as-is (can't map to trajectory)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            traj_path = seed0_dir / source_file
            if not traj_path.exists():
                n_missing_traj += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            try:
                traj = _load_json(traj_path)
            except Exception:
                n_traj_load_err += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            candidates = find_candidate_tool_indices(traj, allowed_tool_names)
            row["candidates"] = candidates
            row["correct_num_selected"] = _compute_correct_num_selected(row, candidates=candidates)
            n_augmented += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp_path, selected_path)

    print(
        json.dumps(
            {
                "selected_jsonl": str(selected_path),
                "backup_path": str(backup_path),
                "rows_total_seen": n_total,
                "rows_augmented": n_augmented,
                "rows_missing_trajectory": n_missing_traj,
                "rows_trajectory_load_error": n_traj_load_err,
                "rows_json_parse_error": n_parse_err,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

