#!/usr/bin/env python3
"""
Compute summary stats for selected_tool_calls JSONL files.

Metrics:
  (1) % valid instances
  (2) % items that generate correct_num_selected (field present) + optional true rates
  (3) average number of indices generated (len(selected_indices))

Validity definitions:
  - relaxed (default): JSON parses; selected_indices exists and is a list[int];
                       if candidates exists, selected_indices must be subset of candidates.
  - strict: relaxed + requires query_id, k_effective, k_requested, source_file;
            if candidates absent, enforces 0 <= idx < k_requested for all selected indices.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass
class Summary:
    path: str
    total_rows: int
    parse_ok_rows: int
    valid_rows: int
    has_correct_num_selected_rows: int
    correct_num_selected_true_rows: int
    indices_count_sum: int
    indices_count_n: int
    invalid_reasons: Dict[str, int]

    def pct(self, num: int, den: int) -> float:
        return 0.0 if den == 0 else 100.0 * num / den

    def avg_indices(self) -> float:
        return float("nan") if self.indices_count_n == 0 else self.indices_count_sum / self.indices_count_n

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_rows": self.total_rows,
            "parse_ok_pct": self.pct(self.parse_ok_rows, self.total_rows),
            "valid_pct": self.pct(self.valid_rows, self.total_rows),
            "has_correct_num_selected_pct": self.pct(
                self.has_correct_num_selected_rows, self.total_rows
            ),
            "correct_num_selected_true_pct_of_all": self.pct(
                self.correct_num_selected_true_rows, self.total_rows
            ),
            "correct_num_selected_true_pct_of_present": self.pct(
                self.correct_num_selected_true_rows, self.has_correct_num_selected_rows
            ),
            "avg_num_indices_generated": self.avg_indices(),
            "top_invalid_reasons": sorted(
                self.invalid_reasons.items(), key=lambda kv: kv[1], reverse=True
            )[:10],
        }


def _validate_relaxed(obj: Dict[str, Any], invalid_reasons: Dict[str, int]) -> Optional[List[int]]:
    if "selected_indices" not in obj:
        invalid_reasons["missing_selected_indices"] = invalid_reasons.get("missing_selected_indices", 0) + 1
        return None

    sel = obj["selected_indices"]
    if not isinstance(sel, list):
        invalid_reasons["selected_indices_not_list"] = invalid_reasons.get("selected_indices_not_list", 0) + 1
        return None
    if not all(_is_int(x) for x in sel):
        invalid_reasons["selected_indices_non_int"] = invalid_reasons.get("selected_indices_non_int", 0) + 1
        return None

    if "candidates" in obj:
        c = obj["candidates"]
        if not isinstance(c, list) or not all(_is_int(x) for x in c):
            invalid_reasons["candidates_bad"] = invalid_reasons.get("candidates_bad", 0) + 1
            return None
        cset = set(c)
        if any(i not in cset for i in sel):
            invalid_reasons["selected_not_subset_of_candidates"] = invalid_reasons.get(
                "selected_not_subset_of_candidates", 0
            ) + 1
            return None

    return sel


def _validate_strict(obj: Dict[str, Any], invalid_reasons: Dict[str, int]) -> Optional[List[int]]:
    for k in ["query_id", "k_effective", "k_requested", "source_file", "selected_indices"]:
        if k not in obj:
            invalid_reasons[f"missing_{k}"] = invalid_reasons.get(f"missing_{k}", 0) + 1
            return None

    if not _is_int(obj["k_effective"]) or not _is_int(obj["k_requested"]):
        invalid_reasons["k_not_int"] = invalid_reasons.get("k_not_int", 0) + 1
        return None

    sel = _validate_relaxed(obj, invalid_reasons)
    if sel is None:
        return None

    # If candidates absent, enforce range using k_requested as a proxy for available items.
    if "candidates" not in obj:
        kreq = obj["k_requested"]
        if any(i < 0 or i >= kreq for i in sel):
            invalid_reasons["selected_out_of_range_k_requested"] = invalid_reasons.get(
                "selected_out_of_range_k_requested", 0
            ) + 1
            return None

    return sel


def summarize_jsonl(path: Path, validity: str) -> Summary:
    invalid_reasons: Dict[str, int] = {}
    total = 0
    parse_ok = 0
    valid = 0
    has_cns = 0
    cns_true = 0
    idx_sum = 0
    idx_n = 0

    if validity not in {"relaxed", "strict"}:
        raise ValueError(f"Unknown validity mode: {validity}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                invalid_reasons["empty_line"] = invalid_reasons.get("empty_line", 0) + 1
                continue

            try:
                obj = json.loads(line)
                parse_ok += 1
            except Exception:
                invalid_reasons["json_parse_error"] = invalid_reasons.get("json_parse_error", 0) + 1
                continue

            if isinstance(obj, dict) and "correct_num_selected" in obj:
                has_cns += 1
                if obj["correct_num_selected"] is True:
                    cns_true += 1

            if not isinstance(obj, dict):
                invalid_reasons["not_object"] = invalid_reasons.get("not_object", 0) + 1
                continue

            if validity == "relaxed":
                sel = _validate_relaxed(obj, invalid_reasons)
            else:
                sel = _validate_strict(obj, invalid_reasons)

            if sel is None:
                continue

            valid += 1
            idx_sum += len(sel)
            idx_n += 1

    return Summary(
        path=str(path),
        total_rows=total,
        parse_ok_rows=parse_ok,
        valid_rows=valid,
        has_correct_num_selected_rows=has_cns,
        correct_num_selected_true_rows=cns_true,
        indices_count_sum=idx_sum,
        indices_count_n=idx_n,
        invalid_reasons=invalid_reasons,
    )


def dump_invalid_jsonl(
    path: Path,
    validity: str,
    *,
    max_items: int,
    preview_chars: int,
) -> None:
    """
    Print invalid instances for quick inspection.

    Output is line-oriented:
      - file:line number (1-indexed)
      - reason
      - short JSON preview (truncated)
    """
    if validity not in {"relaxed", "strict"}:
        raise ValueError(f"Unknown validity mode: {validity}")

    printed = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if printed >= max_items:
                break

            s = raw.strip()
            if not s:
                print(f"{path}:{line_no}\treason=empty_line\tpreview=")
                printed += 1
                continue

            try:
                obj = json.loads(s)
            except Exception:
                preview = s[:preview_chars] + ("..." if len(s) > preview_chars else "")
                print(f"{path}:{line_no}\treason=json_parse_error\tpreview={preview}")
                printed += 1
                continue

            if not isinstance(obj, dict):
                preview = json.dumps(obj, ensure_ascii=False)
                preview = preview[:preview_chars] + ("..." if len(preview) > preview_chars else "")
                print(f"{path}:{line_no}\treason=not_object\tpreview={preview}")
                printed += 1
                continue

            local_reasons: Dict[str, int] = {}
            if validity == "relaxed":
                sel = _validate_relaxed(obj, local_reasons)
            else:
                sel = _validate_strict(obj, local_reasons)

            if sel is not None:
                continue  # valid

            reason = next(iter(local_reasons.keys()), "unknown")
            preview = json.dumps(obj, ensure_ascii=False)
            preview = preview[:preview_chars] + ("..." if len(preview) > preview_chars else "")
            print(f"{path}:{line_no}\treason={reason}\tpreview={preview}")
            printed += 1


def _print_human(s: Summary) -> None:
    d = s.as_dict()
    print(f"\n== {d['path']} ==")
    print(f"total_rows: {d['total_rows']}")
    print(f"parse_ok_pct: {d['parse_ok_pct']:.2f}%")
    print(f"valid_pct: {d['valid_pct']:.2f}%")
    print(f"has_correct_num_selected_pct: {d['has_correct_num_selected_pct']:.2f}%")
    print(f"correct_num_selected_true_pct_of_all: {d['correct_num_selected_true_pct_of_all']:.2f}%")
    print(f"correct_num_selected_true_pct_of_present: {d['correct_num_selected_true_pct_of_present']:.2f}%")
    print(f"avg_num_indices_generated: {d['avg_num_indices_generated']:.4f}")
    if d["top_invalid_reasons"]:
        print("top_invalid_reasons:", d["top_invalid_reasons"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help="One or more JSONL files to summarize.",
    )
    ap.add_argument(
        "--validity",
        choices=["relaxed", "strict"],
        default="relaxed",
        help="Which validity definition to use.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON (list of summary dicts).",
    )
    ap.add_argument(
        "--dump-invalid",
        action="store_true",
        help="Print invalid instances (reason + JSON preview).",
    )
    ap.add_argument(
        "--max-invalid",
        type=int,
        default=20,
        help="Max invalid instances to print per file when --dump-invalid is set.",
    )
    ap.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Max characters to show for each invalid JSON preview.",
    )
    args = ap.parse_args()

    summaries = [summarize_jsonl(Path(p), validity=args.validity) for p in args.paths]

    if args.json:
        print(json.dumps([s.as_dict() for s in summaries], indent=2))
    else:
        for s in summaries:
            _print_human(s)
            if args.dump_invalid:
                dump_invalid_jsonl(
                    Path(s.path),
                    validity=args.validity,
                    max_items=args.max_invalid,
                    preview_chars=args.preview_chars,
                )


if __name__ == "__main__":
    main()

