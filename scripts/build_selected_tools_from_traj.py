#!/usr/bin/env python3
"""
Build a Selected Tools HF dataset by extracting excerpts directly from the
<trajectory_summary> tag embedded in each run file's user message.

Now supports --eval-dir to join with evaluation results (question, correct_answer, correct).

No JSONL file required — works for any traj_summary_orig_ext_selected_tools_* dir.

Output columns (compatible with selected_tools.py backend):
  query_id, excerpt, new_trajectory, status, tool_call_counts,
  rationale, selected_indices, k_requested, k_effective, direct_answer,
  question, correct_answer, correct

Requires: pip install 'datasets>=2.14'
Python env: /scratch/hc3337/envs/raca-py312/bin/python
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any


# ── Trajectory formatter (same as build_scout_runs_dataset.py) ────────────────

def _parse_tool_call_block(text: str) -> str:
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
    parts = re.split(r"(?m)(?=^\[Tool call\])", text, flags=re.MULTILINE)
    out: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out.append(("tool_call" if part.startswith("[Tool call]") else "reasoning", part))
    return out


def format_new_trajectory(result: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    pending: list[str] = []

    def flush():
        if pending:
            combined = "\n\n".join(t for t in pending if t.strip())
            if combined.strip():
                blocks.append(f"[Reasoning]\n{combined.strip()}")
            pending.clear()

    for item in result:
        item_type = item.get("type", "")
        output = item.get("output", "")
        if item_type == "user":
            continue
        text = "\n".join(str(s) for s in output) if isinstance(output, list) else str(output)
        text = text.strip()
        if not text:
            continue
        if item_type == "reasoning":
            for sub_type, sub_text in _split_reasoning_text(text):
                if sub_type == "tool_call":
                    flush()
                    blocks.append(_parse_tool_call_block(sub_text))
                else:
                    pending.append(sub_text)
        elif item_type == "tool_call":
            flush()
            tool_name = str(item.get("tool_name") or "unknown")
            args = str(item.get("arguments") or "").strip()
            block = f"[Tool Call: {tool_name}]\n{args}"
            if text:
                block += f"\n\n[Tool Result]\n{text}"
            blocks.append(block)
        elif item_type == "output_text":
            flush()
            blocks.append(f"[Final Answer]\n{text}")
        else:
            flush()
            blocks.append(text)
    flush()
    return "\n\n---\n\n".join(blocks)


def extract_trajectory_summary(user_text: str) -> str:
    """Extract raw content between <trajectory_summary> tags.
    Takes the LAST match — the first is an empty demonstration in the prompt preamble.
    """
    matches = re.findall(r"<trajectory_summary>(.*?)</trajectory_summary>", user_text, re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


# ── Eval data loading ─────────────────────────────────────────────────────────

def load_eval_data(eval_dir: Path) -> dict[str | int, dict]:
    """Load eval files and return query_id -> {question, correct_answer, correct}."""
    eval_map: dict[str | int, dict] = {}
    for p in eval_dir.glob("*_eval.json"):
        try:
            d = json.load(p.open("r", encoding="utf-8"))
            qid_raw = str(d.get("query_id", "")).strip()
            qid: str | int = int(qid_raw) if qid_raw.isdigit() else qid_raw
            judge_result = d.get("judge_result") or {}
            correct_val = judge_result.get("correct")
            eval_map[qid] = {
                "question": str(d.get("question") or ""),
                "correct_answer": str(d.get("correct_answer") or ""),
                "correct": bool(correct_val) if correct_val is not None else None,
            }
        except Exception as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
    print(f"Loaded {len(eval_map)} eval entries from {eval_dir}", file=sys.stderr)
    return eval_map


# ── Run file loading ──────────────────────────────────────────────────────────

def _coerce_query_id(qid: str) -> int | str:
    s = str(qid).strip()
    return int(s) if s.isdigit() else s


def _sort_key(qid: int | str) -> tuple[int, str]:
    s = str(qid).strip()
    return (0, f"{int(s):010d}") if s.isdigit() else (1, s)


def build_rows(run_dir: Path, strict: bool,
               eval_map: dict | None = None) -> list[dict[str, Any]]:
    rows = []
    files = sorted(run_dir.glob("run_*.json"))
    if not files:
        raise FileNotFoundError(f"No run_*.json under {run_dir}")

    for p in files:
        d = json.load(p.open("r", encoding="utf-8"))
        qid = _coerce_query_id(str(d.get("query_id", "")).strip())

        result = d.get("result", [])

        # extract user message
        user_text = ""
        for item in result:
            if item.get("type") == "user":
                out = item.get("output", "")
                user_text = "\n".join(str(s) for s in out) if isinstance(out, list) else str(out)
                break

        excerpt = extract_trajectory_summary(user_text)
        if not excerpt:
            msg = f"{p.name}: no <trajectory_summary> content found"
            if strict:
                raise ValueError(msg)
            print(f"warning: skip {msg}", file=sys.stderr)
            continue

        tc_raw = d.get("tool_call_counts") or {}
        tc = {str(k): int(v) for k, v in tc_raw.items() if v is not None}

        # Join with eval data if available
        eval_entry = (eval_map or {}).get(qid, {})
        question = eval_entry.get("question", "")
        correct_answer = eval_entry.get("correct_answer", "")
        correct = eval_entry.get("correct", None)

        rows.append({
            "query_id": qid,
            "excerpt": excerpt,
            "new_trajectory": format_new_trajectory(result),
            "status": str(d.get("status", "")),
            "tool_call_counts": json.dumps(tc, sort_keys=True),
            # JSONL-derived fields — not available, set compatible defaults
            "rationale": "",
            "selected_indices": "[]",
            "k_requested": 0,
            "k_effective": 0,
            "direct_answer": False,
            # Eval fields
            "question": question,
            "correct_answer": correct_answer,
            "correct": correct,
        })

    rows.sort(key=lambda r: _sort_key(str(r["query_id"])))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Build Selected Tools HF dataset from <trajectory_summary> in run files."
    )
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory of run_*.json files.")
    ap.add_argument("--eval-dir", type=Path, default=None,
                    help="Directory of *_eval.json files with question/correct_answer/judge_result.")
    ap.add_argument("--no-strict", action="store_true",
                    help="Skip files missing <trajectory_summary> instead of erroring.")
    ap.add_argument("--output", type=Path,
                    help="Output path (.parquet, .jsonl, .json).")
    ap.add_argument("--push-to-hub", metavar="REPO_ID",
                    help="Push to HuggingFace Hub.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--commit-message",
                    default="Update dataset from build_selected_tools_from_traj.py")
    args = ap.parse_args()

    if not args.output and not args.push_to_hub:
        print("Error: --output or --push-to-hub required", file=sys.stderr)
        sys.exit(2)

    eval_map = None
    if args.eval_dir:
        eval_map = load_eval_data(args.eval_dir.resolve())

    rows = build_rows(args.run_dir.resolve(), strict=not args.no_strict, eval_map=eval_map)

    if args.output:
        suf = args.output.suffix.lower()
        if suf == ".jsonl":
            with args.output.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suf == ".json":
            with args.output.open("w") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        elif suf == ".parquet":
            from datasets import Dataset
            Dataset.from_list(rows).to_parquet(str(args.output))
        else:
            print(f"Error: unsupported extension {suf!r}", file=sys.stderr)
            sys.exit(2)

    if args.push_to_hub:
        from datasets import Dataset
        ds = Dataset.from_list(rows)
        ds.push_to_hub(args.push_to_hub, split=args.split,
                       commit_message=args.commit_message)

    correct_count = sum(1 for r in rows if r.get("correct") is True)
    total_with_eval = sum(1 for r in rows if r.get("correct") is not None)
    print(f"Wrote {len(rows)} rows. Columns: {list(rows[0].keys()) if rows else []}")
    if total_with_eval:
        print(f"Eval coverage: {total_with_eval}/{len(rows)} rows, {correct_count} correct ({100*correct_count//total_with_eval}%)")
    if args.push_to_hub:
        print(f"Pushed to {args.push_to_hub} (split {args.split!r}).")


if __name__ == "__main__":
    main()
