#!/usr/bin/env python3
"""
Check whether selected indices are a subset of candidate tool-call indices.

This script validates rows from:
  selected_tool_calls/selected_tool_calls.jsonl

Against candidates reconstructed from trajectories in:
  runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed0

Candidate set definition mirrors select_useful_tool_calls.py (no original_messages):
  candidates = find_candidate_tool_indices(trajectory, allowed_tool_names)
  where allowed_tool_names defaults to DEFAULT_TOOL_NAMES
    ("local_knowledge_base_retrieval", "search", "get_document")

Outputs:
  - summary counts and % candidate-valid
  - optional invalid row dumps with candidate previews via build_catalog_lines
  - optional JSONL with per-row status for downstream analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure repo root is importable when running from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from select_useful_tool_calls import DEFAULT_TOOL_NAMES, build_catalog_lines, find_candidate_tool_indices


@dataclass
class RowStatus:
    line_no: int
    query_id: str
    source_file: str
    selected_indices: List[int]
    num_candidates: int
    bad_indices: List[int]
    status: str  # ok | bad_indices | load_error | parse_error | schema_error
    error: Optional[str] = None


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Sequence[Tuple[int, str]]:
    # (line_no, line_text)
    out: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            out.append((i, line))
    return out


def check_file(
    *,
    selected_jsonl: Path,
    seed0_dir: Path,
    allowed_tool_names: Set[str],
    dump_invalid: bool,
    max_invalid: int,
    preview_candidates: int,
    preview_chars: int,
    write_jsonl: Optional[Path],
) -> List[RowStatus]:
    statuses: List[RowStatus] = []
    invalid_printed = 0

    for line_no, raw in iter_jsonl(selected_jsonl):
        s = raw.strip()
        if not s:
            continue

        try:
            row = json.loads(s)
        except Exception as e:
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id="",
                    source_file="",
                    selected_indices=[],
                    num_candidates=0,
                    bad_indices=[],
                    status="parse_error",
                    error=str(e),
                )
            )
            continue

        if not isinstance(row, dict):
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id="",
                    source_file="",
                    selected_indices=[],
                    num_candidates=0,
                    bad_indices=[],
                    status="schema_error",
                    error="row_not_object",
                )
            )
            continue

        qid = str(row.get("query_id", ""))
        source_file = str(row.get("source_file", ""))
        selected = row.get("selected_indices")
        if not qid or not source_file or not isinstance(selected, list) or not all(_is_int(x) for x in selected):
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id=qid,
                    source_file=source_file,
                    selected_indices=[] if not isinstance(selected, list) else [x for x in selected if _is_int(x)],
                    num_candidates=0,
                    bad_indices=[],
                    status="schema_error",
                    error="missing_or_bad_fields(query_id/source_file/selected_indices)",
                )
            )
            continue

        traj_path = seed0_dir / source_file
        if not traj_path.exists():
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id=qid,
                    source_file=source_file,
                    selected_indices=selected,
                    num_candidates=0,
                    bad_indices=selected,
                    status="load_error",
                    error=f"missing_trajectory_file:{traj_path}",
                )
            )
            continue

        try:
            traj = load_json(traj_path)
        except Exception as e:
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id=qid,
                    source_file=source_file,
                    selected_indices=selected,
                    num_candidates=0,
                    bad_indices=selected,
                    status="load_error",
                    error=f"trajectory_json_load_failed:{e}",
                )
            )
            continue

        try:
            candidates = find_candidate_tool_indices(traj, allowed_tool_names)
        except Exception as e:
            statuses.append(
                RowStatus(
                    line_no=line_no,
                    query_id=qid,
                    source_file=source_file,
                    selected_indices=selected,
                    num_candidates=0,
                    bad_indices=selected,
                    status="schema_error",
                    error=f"candidate_build_failed:{e}",
                )
            )
            continue

        cand_set = set(candidates)
        bad = [i for i in selected if i not in cand_set]
        st = RowStatus(
            line_no=line_no,
            query_id=qid,
            source_file=source_file,
            selected_indices=selected,
            num_candidates=len(candidates),
            bad_indices=bad,
            status="ok" if not bad else "bad_indices",
        )
        statuses.append(st)

        if dump_invalid and bad and invalid_printed < max_invalid:
            invalid_printed += 1
            print("\n---")
            print(f"line_no={line_no} query_id={qid} source_file={source_file}")
            print(f"selected_indices={selected}")
            print(f"bad_indices={bad}")
            head = candidates[:preview_candidates]
            print(f"num_candidates={len(candidates)} candidates_head={head}")
            try:
                for l in build_catalog_lines(traj, head, preview_chars):
                    print(l)
            except Exception as e:
                print(f"(build_catalog_lines failed: {e})")

    if write_jsonl is not None:
        write_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with write_jsonl.open("w", encoding="utf-8") as f:
            for st in statuses:
                f.write(
                    json.dumps(
                        {
                            "line_no": st.line_no,
                            "query_id": st.query_id,
                            "source_file": st.source_file,
                            "selected_indices": st.selected_indices,
                            "num_candidates": st.num_candidates,
                            "bad_indices": st.bad_indices,
                            "status": st.status,
                            "error": st.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return statuses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected-jsonl",
        default="selected_tool_calls/selected_tool_calls.jsonl",
        help="Path to selected_tool_calls JSONL.",
    )
    ap.add_argument(
        "--seed0-dir",
        default="runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed0",
        help="Directory containing seed0 trajectory JSON files.",
    )
    ap.add_argument(
        "--allowed-tool-names",
        default=",".join(DEFAULT_TOOL_NAMES),
        help="Comma-separated allowed tool names (default: DEFAULT_TOOL_NAMES).",
    )
    ap.add_argument("--dump-invalid", action="store_true", help="Print invalid rows with previews.")
    ap.add_argument("--max-invalid", type=int, default=20, help="Max invalid rows to print.")
    ap.add_argument(
        "--preview-candidates",
        type=int,
        default=10,
        help="How many candidate indices to preview per invalid row.",
    )
    ap.add_argument(
        "--preview-chars",
        type=int,
        default=250,
        help="Preview chars per candidate line.",
    )
    ap.add_argument(
        "--write-jsonl",
        default="",
        help="Optional: write per-row status JSONL to this path.",
    )
    args = ap.parse_args()

    selected_jsonl = Path(args.selected_jsonl)
    seed0_dir = Path(args.seed0_dir)
    allowed_tool_names = {s.strip() for s in args.allowed_tool_names.split(",") if s.strip()}

    write_jsonl = Path(args.write_jsonl) if args.write_jsonl else None

    statuses = check_file(
        selected_jsonl=selected_jsonl,
        seed0_dir=seed0_dir,
        allowed_tool_names=allowed_tool_names,
        dump_invalid=args.dump_invalid,
        max_invalid=args.max_invalid,
        preview_candidates=args.preview_candidates,
        preview_chars=args.preview_chars,
        write_jsonl=write_jsonl,
    )

    total = len(statuses)
    ok = sum(1 for s in statuses if s.status == "ok")
    bad = sum(1 for s in statuses if s.status == "bad_indices")
    load_err = sum(1 for s in statuses if s.status == "load_error")
    parse_err = sum(1 for s in statuses if s.status == "parse_error")
    schema_err = sum(1 for s in statuses if s.status == "schema_error")

    denom = total if total else 1
    print("\n== Summary ==")
    print(f"rows_total={total}")
    print(f"candidate_valid_ok={ok} ({100.0*ok/denom:.2f}%)")
    print(f"bad_indices={bad} ({100.0*bad/denom:.2f}%)")
    print(f"load_error={load_err} ({100.0*load_err/denom:.2f}%)")
    print(f"parse_error={parse_err} ({100.0*parse_err/denom:.2f}%)")
    print(f"schema_error={schema_err} ({100.0*schema_err/denom:.2f}%)")


if __name__ == "__main__":
    main()

