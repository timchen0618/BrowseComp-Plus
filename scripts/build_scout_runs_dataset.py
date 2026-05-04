#!/usr/bin/env python3
"""
Build a HuggingFace dataset for "scout runs" visualization.

Joins two run_*.json directories on query_id:
  --scout-dir   : short "scout" run (budget-limited, fewer steps)
  --new-dir     : full run conditioned on the scout trajectory

Columns:
  query_id, scout_trajectory, new_trajectory,
  scout_status, new_status, scout_tool_calls, new_tool_calls

trajectory format (consumed by parseTrajectory in ScoutRunsApp.tsx):
  Blocks separated by "\\n\\n---\\n\\n"
  [Reasoning]\\n<text>
  [Tool Call: name]\\n<args>\\n\\n[Tool Result]\\n<result>
  [Final Answer]\\n<text>

Requires: pip install 'datasets>=2.14'
Environment for push: HF_TOKEN
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any


COLUMNS: tuple[str, ...] = (
    "query_id",
    "scout_trajectory",
    "new_trajectory",
    "scout_status",
    "new_status",
    "scout_tool_calls",
    "new_tool_calls",
)


def _sort_key_query_id(qid: int | str) -> tuple[int, str]:
    s = str(qid).strip()
    if s.isdigit():
        return (0, f"{int(s):010d}")
    return (1, s)


def _coerce_query_id(qid: str) -> int | str:
    s = str(qid).strip()
    if s.isdigit():
        return int(s)
    return s


def _parse_tool_call_block(text: str) -> str:
    """Convert a raw '[Tool call] ...' string into '[Tool Call: name]\\n...' format."""
    rest = text[len("[Tool call]"):].lstrip()
    newline_idx = rest.find("\n")
    if newline_idx >= 0:
        tool_name = rest[:newline_idx].strip()
        body = rest[newline_idx + 1:].strip()
    else:
        tool_name = rest.strip()
        body = ""

    if body.lower().startswith("arguments:\n"):
        body = body[len("arguments:\n"):]

    result_part = ""
    for marker in ("\n\n[Tool result]:\n", "\n[Tool result]:\n"):
        if marker in body:
            args_part, result_part = body.split(marker, 1)
            body = args_part
            break

    block = f"[Tool Call: {tool_name}]\n{body.strip()}"
    if result_part:
        block += f"\n\n[Tool Result]\n{result_part.strip()}"
    return block


def _split_reasoning_text(text: str) -> list[tuple[str, str]]:
    """Split a mixed reasoning+tool-call string into typed sub-parts."""
    parts = re.split(r"(?m)(?=^\[Tool call\])", text, flags=re.MULTILINE)
    out: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("[Tool call]"):
            out.append(("tool_call", part))
        else:
            out.append(("reasoning", part))
    return out


def format_trajectory(result: list[dict[str, Any]]) -> str:
    """Convert a run JSON result list into the dashboard block format.

    Handles two raw formats:
      Format A: tool calls embedded as '[Tool call] ...' text inside reasoning items
      Format B: explicit type=tool_call items with tool_name / arguments / output fields

    Output blocks joined by '\\n\\n---\\n\\n':
      [Reasoning]\\n<text>
      [Tool Call: name]\\n<args>\\n\\n[Tool Result]\\n<result>
      [Final Answer]\\n<text>
    """
    blocks: list[str] = []
    pending_reasoning: list[str] = []

    def flush_reasoning() -> None:
        if pending_reasoning:
            combined = "\n\n".join(t for t in pending_reasoning if t.strip())
            if combined.strip():
                blocks.append(f"[Reasoning]\n{combined.strip()}")
            pending_reasoning.clear()

    for item in result:
        item_type = item.get("type", "")
        output = item.get("output", "")

        if item_type == "user":
            continue

        text: str
        if isinstance(output, list):
            text = "\n".join(str(s) for s in output)
        else:
            text = str(output)
        text = text.strip()
        if not text:
            continue

        if item_type == "reasoning":
            for sub_type, sub_text in _split_reasoning_text(text):
                if sub_type == "tool_call":
                    flush_reasoning()
                    blocks.append(_parse_tool_call_block(sub_text))
                else:
                    pending_reasoning.append(sub_text)

        elif item_type == "tool_call":
            flush_reasoning()
            tool_name = str(item.get("tool_name") or "unknown")
            args = str(item.get("arguments") or "").strip()
            block = f"[Tool Call: {tool_name}]\n{args}"
            if text:
                block += f"\n\n[Tool Result]\n{text}"
            blocks.append(block)

        elif item_type == "output_text":
            flush_reasoning()
            blocks.append(f"[Final Answer]\n{text}")

        else:
            flush_reasoning()
            blocks.append(text)

    flush_reasoning()
    return "\n\n---\n\n".join(blocks)


def _tool_call_counts_str(run: dict[str, Any]) -> str:
    raw = run.get("tool_call_counts") or {}
    if not isinstance(raw, dict):
        return "{}"
    counts: dict[str, int] = {}
    for k, v in raw.items():
        if v is None:
            continue
        try:
            counts[str(k)] = int(v)
        except (TypeError, ValueError):
            pass
    return json.dumps(counts, ensure_ascii=False, sort_keys=True)


def _total_tool_calls(counts_str: str) -> int:
    try:
        d = json.loads(counts_str)
        return sum(d.values())
    except Exception:
        return 0


def load_run_dir(run_dir: Path) -> dict[str, dict[str, Any]]:
    runs = sorted(run_dir.glob("run_*.json"))
    if not runs:
        raise FileNotFoundError(f"No run_*.json under {run_dir}")
    by_id: dict[str, dict[str, Any]] = {}
    for p in runs:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        qid = str(d.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"{p}: missing query_id")
        by_id[qid] = d
    return by_id


def build_rows(
    *,
    scout_by_id: dict[str, dict[str, Any]],
    new_by_id: dict[str, dict[str, Any]],
    strict: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    all_ids = set(scout_by_id) | set(new_by_id)
    for qid in sorted(all_ids, key=_sort_key_query_id):
        scout = scout_by_id.get(qid)
        new = new_by_id.get(qid)

        if scout is None:
            msg = f"query_id={qid!r} has no scout run"
            if strict:
                raise KeyError(msg)
            print(f"warning: skip {msg}", file=sys.stderr)
            continue
        if new is None:
            msg = f"query_id={qid!r} has no new trajectory run"
            if strict:
                raise KeyError(msg)
            print(f"warning: skip {msg}", file=sys.stderr)
            continue

        scout_tc = _tool_call_counts_str(scout)
        new_tc = _tool_call_counts_str(new)

        rows.append({
            "query_id": _coerce_query_id(qid),
            "scout_trajectory": format_trajectory(scout.get("result", [])),
            "new_trajectory": format_trajectory(new.get("result", [])),
            "scout_status": str(scout.get("status", "")),
            "new_status": str(new.get("status", "")),
            "scout_tool_calls": scout_tc,
            "new_tool_calls": new_tc,
        })

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Join scout run dir + new trajectory dir into an HF scout-runs dataset."
    )
    ap.add_argument("--scout-dir", type=Path, required=True,
                    help="Directory of scout run_*.json files (budget-limited).")
    ap.add_argument("--new-dir", type=Path, required=True,
                    help="Directory of new trajectory run_*.json files.")
    ap.add_argument("--no-strict", action="store_true",
                    help="Allow missing query_ids on either side (skip instead of error).")
    ap.add_argument("--output", type=Path,
                    help="Output path: .parquet, .jsonl, or .json.")
    ap.add_argument("--push-to-hub", metavar="REPO_ID",
                    help="Push to HuggingFace (requires HF_TOKEN and datasets).")
    ap.add_argument("--split", default="train",
                    help="Split name for push_to_hub (default: train).")
    ap.add_argument("--commit-message",
                    default="Update dataset from build_scout_runs_dataset.py")
    args = ap.parse_args()

    if not args.output and not args.push_to_hub:
        print("Error: --output or --push-to-hub required", file=sys.stderr)
        sys.exit(2)

    scout_by_id = load_run_dir(args.scout_dir.resolve())
    new_by_id = load_run_dir(args.new_dir.resolve())

    rows = build_rows(
        scout_by_id=scout_by_id,
        new_by_id=new_by_id,
        strict=not args.no_strict,
    )
    rows.sort(key=lambda r: _sort_key_query_id(str(r["query_id"])))

    if args.output:
        suf = args.output.suffix.lower()
        if suf == ".jsonl":
            with args.output.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suf == ".json":
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        elif suf == ".parquet":
            try:
                from datasets import Dataset
            except ImportError as e:
                print("Error: .parquet requires: pip install datasets pyarrow", file=sys.stderr)
                raise SystemExit(1) from e
            Dataset.from_list(rows).to_parquet(str(args.output))
        else:
            print(f"Error: unsupported extension {suf!r}", file=sys.stderr)
            sys.exit(2)

    if args.push_to_hub:
        try:
            from datasets import Dataset
        except ImportError as e:
            print("Error: --push-to-hub requires: pip install datasets", file=sys.stderr)
            raise SystemExit(1) from e
        ds = Dataset.from_list(rows)
        ds.push_to_hub(args.push_to_hub, split=args.split, commit_message=args.commit_message)

    n = len(rows)
    scout_incomplete = sum(1 for r in rows if r["scout_status"] == "incomplete")
    out_msg = f"Wrote {n} rows ({scout_incomplete} scout incomplete), columns: {COLUMNS!r}."
    if args.output:
        out_msg += f" -> {args.output}"
    if args.push_to_hub:
        out_msg += f" Pushed to {args.push_to_hub} (split {args.split!r})."
    print(out_msg)


if __name__ == "__main__":
    main()
