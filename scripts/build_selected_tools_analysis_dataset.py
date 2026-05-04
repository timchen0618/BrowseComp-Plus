#!/usr/bin/env python3
"""
Build a Hugging Face–style table for "selected tools analysis" (tool JSONL + new run JSON).

Columns match timchen0618/browsecomp-plus-selected-tools-analysis-v1:
  query_id, rationale, selected_indices, k_requested, k_effective, excerpt,
  new_trajectory, direct_answer, tool_call_counts, status
Types aligned with the Hub: numeric query_id as int, tool_call_counts as a JSON string.

new_trajectory format (consumed by parseNewTrajectory in SelectedToolsApp.tsx):
  Blocks separated by "\n\n---\n\n"
  [Reasoning]\n<text>
  [Tool Call: name]\n<args>\n\n[Tool Result]\n<result>
  [Final Answer]\n<text>

Requires: pip install 'datasets>=2.14'  (or: pip install -e ".[hf]")

Environment for push: HF_TOKEN
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any


COLUMNS: tuple[str, ...] = (
    "query_id",
    "rationale",
    "selected_indices",
    "k_requested",
    "k_effective",
    "excerpt",
    "new_trajectory",
    "direct_answer",
    "tool_call_counts",
    "status",
)


def _sort_key_query_id(qid: int | str) -> tuple[int, str]:
    s = str(qid).strip()
    if s.isdigit():
        return (0, f"{int(s):010d}")
    return (1, s)


def _coerce_query_id(qid: str) -> int | str:
    """Match Hub: numeric query_id as int, else str."""
    s = str(qid).strip()
    if s.isdigit():
        return int(s)
    return s


def _load_jsonl_paths(
    paths: list[Path],
    *,
    last_wins: bool,
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"JSONL not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{line_no}: invalid JSON: {e}") from e
                qid = str(row.get("query_id", "")).strip()
                if not qid:
                    raise ValueError(f"{p}:{line_no}: missing query_id")
                if qid in by_id and not last_wins:
                    raise ValueError(
                        f"Duplicate query_id {qid!r} in {p}; use --last-wins to allow"
                    )
                by_id[qid] = row
    return by_id


def _expand_globs(repo_root: Path, patterns: list[str]) -> list[Path]:
    if not patterns:
        return []
    out: list[Path] = []
    for pat in patterns:
        if Path(pat).is_absolute():
            base = pat
        else:
            base = (repo_root / pat).as_posix()
        recursive = "**" in pat
        for f in sorted(glob.glob(base, recursive=recursive)):
            p = Path(f)
            if p.is_file() and p.suffix == ".jsonl":
                out.append(p)
    # dedupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _tool_total_calls(counts: dict[str, Any] | None) -> int:
    if not counts:
        return 0
    return int(sum(int(v) for v in counts.values() if v is not None))


def _parse_tool_call_block(text: str) -> str:
    """Convert a raw '[Tool call] ...' string into the dashboard's '[Tool Call: name]\\n...' format."""
    rest = text[len("[Tool call]"):].lstrip()
    newline_idx = rest.find("\n")
    if newline_idx >= 0:
        tool_name = rest[:newline_idx].strip()
        body = rest[newline_idx + 1:].strip()
    else:
        tool_name = rest.strip()
        body = ""

    # Strip optional "arguments:\n" header
    if body.lower().startswith("arguments:\n"):
        body = body[len("arguments:\n"):]

    # Split off "[Tool result]:" section
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
    """Split a mixed reasoning+tool-call string into typed sub-parts.

    A reasoning item's output may contain both plain reasoning text and one or more
    embedded '[Tool call] ...' sections. This splits them at [Tool call] boundaries so
    each section can be formatted independently.
    Returns a list of ('reasoning', text) or ('tool_call', text) tuples.
    """
    import re
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


def _format_new_trajectory(result: list[dict[str, Any]]) -> str:
    """Convert a run JSON result list into the block-formatted string the dashboard expects.

    Input items have keys: type, tool_name, arguments, output
      type "user"        → skipped
      type "reasoning"   → output is list[str]; the text is split at [Tool call] boundaries
                           so embedded tool calls are extracted as Tool Call + Tool Result
                           blocks and surrounding text becomes Reasoning blocks
      type "output_text" → output is str; becomes the Final Answer block

    Output format (blocks joined by "\\n\\n---\\n\\n"):
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
            # Format A: tool calls embedded as "[Tool call] ..." markers in the text
            for sub_type, sub_text in _split_reasoning_text(text):
                if sub_type == "tool_call":
                    flush_reasoning()
                    blocks.append(_parse_tool_call_block(sub_text))
                else:
                    pending_reasoning.append(sub_text)

        elif item_type == "tool_call":
            # Format B: explicit tool_call item with tool_name, arguments, output (= result)
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
            # Unknown type: flush pending reasoning and emit as-is
            flush_reasoning()
            blocks.append(text)

    flush_reasoning()
    return "\n\n---\n\n".join(blocks)


def _row_from_join(
    sel: dict[str, Any],
    run: dict[str, Any],
    *,
    k_default: int,
) -> dict[str, Any]:
    result = run.get("result", [])
    if not isinstance(result, list):
        raise TypeError("run JSON must have list result")

    selected_indices = sel.get("selected_indices", [])
    if not isinstance(selected_indices, list):
        selected_indices = list(selected_indices)  # type: ignore[arg-type]
    k_eff = sel.get("k_effective")
    if not isinstance(k_eff, int) or k_eff < 0:
        k_eff = len(selected_indices)
    k_req = sel.get("k_requested")
    if not isinstance(k_req, int) or k_req < 0:
        k_req = k_default

    raw_counts = run.get("tool_call_counts") or {}
    if not isinstance(raw_counts, dict):
        raise TypeError("tool_call_counts must be a dict in run JSON")
    counts: dict[str, int] = {}
    for k, v in raw_counts.items():
        if v is None:
            continue
        try:
            counts[str(k)] = int(v)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Non-int value in tool_call_counts[{k!r}]: {v!r}") from e

    new_traj = _format_new_trajectory(result)
    direct = _tool_total_calls(counts) == 0
    status = run.get("status", "")
    if not isinstance(status, str):
        status = str(status)

    qid = str(sel.get("query_id", run.get("query_id", ""))).strip()
    if not qid:
        raise ValueError("missing query_id in selection and run")
    return {
        "query_id": _coerce_query_id(qid),
        "rationale": str(sel.get("rationale", "")),
        "selected_indices": [int(x) for x in selected_indices],
        "k_requested": int(k_req),
        "k_effective": int(k_eff),
        "excerpt": str(sel.get("excerpt", "")),
        "new_trajectory": new_traj,
        "direct_answer": direct,
        # Hub stores a JSON string (e.g. '{"search":17}' or '{}')
        "tool_call_counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
        "status": status,
    }


def _check_trajectory_file(
    run: dict[str, Any],
    loaded_names: set[str],
) -> None:
    meta = run.get("metadata") or {}
    if not isinstance(meta, dict):
        return
    tf = meta.get("trajectory_summary_file")
    if not tf:
        return
    name = Path(str(tf)).name
    if name not in loaded_names:
        raise ValueError(
            f"Run query_id={run.get('query_id')!r} metadata.trajectory_summary_file "
            f"basename {name!r} not among loaded JSONL basenames: {sorted(loaded_names)}"
        )


def build_rows(
    *,
    run_dir: Path,
    selection_by_id: dict[str, dict[str, Any]],
    jsonl_basenames: set[str],
    k_default: int,
    check_metadata: bool,
    strict: bool,
) -> list[dict[str, Any]]:
    runs = sorted(run_dir.glob("run_*.json"))
    if not runs:
        raise FileNotFoundError(f"No run_*.json under {run_dir}")

    rows: list[dict[str, Any]] = []
    run_qids: set[str] = set()
    for path in runs:
        with path.open("r", encoding="utf-8") as f:
            run = json.load(f)
        qid = str(run.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"{path}: missing query_id")
        if check_metadata:
            _check_trajectory_file(run, jsonl_basenames)
        if qid not in selection_by_id:
            if strict:
                raise KeyError(
                    f"Run {path.name} query_id={qid!r} has no row in selected JSONL"
                )
            print(
                f"warning: skip {path.name} — no selection row for query_id={qid!r}",
                file=sys.stderr,
            )
            continue
        run_qids.add(qid)
        rows.append(
            _row_from_join(selection_by_id[qid], run, k_default=k_default)
        )
    if not strict:
        orphan = set(selection_by_id.keys()) - run_qids
        if orphan:
            print(
                f"warning: {len(orphan)} JSONL query_ids have no run_*.json: "
                f"{sorted(orphan, key=_sort_key_query_id)[:10]!r}...",
                file=sys.stderr,
            )
    return rows


def _strict_bidirectional(
    selection_by_id: dict[str, dict[str, Any]],
    run_dir: Path,
) -> None:
    run_ids: set[str] = set()
    for path in run_dir.glob("run_*.json"):
        with path.open("r", encoding="utf-8") as f:
            run = json.load(f)
        qid = str(run.get("query_id", "")).strip()
        if qid:
            run_ids.add(qid)
    sel_ids = set(selection_by_id.keys())
    miss_sel = run_ids - sel_ids
    miss_run = sel_ids - run_ids
    if miss_sel:
        raise SystemExit(
            f"--strict: {len(miss_sel)} run query_ids missing in JSONL: "
            f"{sorted(miss_sel, key=_sort_key_query_id)[:20]!r}..."
        )
    if miss_run:
        raise SystemExit(
            f"--strict: {len(miss_run)} JSONL query_ids missing runs: "
            f"{sorted(miss_run, key=_sort_key_query_id)[:20]!r}..."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Join selected-tool JSONL with run_*.json into an HF analysis dataset table."
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing run_*.json (one trajectory per file).",
    )
    ap.add_argument(
        "--selected-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Path to a selected-tool JSONL (repeatable).",
    )
    ap.add_argument(
        "--selected-jsonl-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob under --repo-root (repeatable), e.g. selected_tool_calls/test150/.../selected_tool_calls_test150_*.jsonl",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root for relative JSONL glob paths (default: cwd).",
    )
    ap.add_argument(
        "--k-requested-default",
        type=int,
        default=5,
        help="k_requested when missing from a JSONL row (default: 5).",
    )
    ap.add_argument(
        "--last-wins",
        action="store_true",
        help="If a query_id appears in multiple JSONL files, use the last file's row.",
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not require JSONL and run query_id sets to match exactly (still skip runs with no selection).",
    )
    ap.add_argument(
        "--check-metadata-trajectory",
        action="store_true",
        help="Require each run's metadata.trajectory_summary_file basename to match a loaded JSONL filename.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        help="Output path: .parquet, .jsonl, or .json. Required unless --push-to-hub is set.",
    )
    ap.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Push to Hugging Face (requires HF_TOKEN and datasets).",
    )
    ap.add_argument(
        "--split",
        default="train",
        help="Split name for push_to_hub (default: train).",
    )
    ap.add_argument(
        "--commit-message",
        default="Update dataset from build_selected_tools_analysis_dataset.py",
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()

    file_paths: list[Path] = [p.resolve() for p in args.selected_jsonl]
    file_paths += _expand_globs(repo_root, list(args.selected_jsonl_glob))
    if not file_paths:
        print(
            "Error: provide --selected-jsonl and/or --selected-jsonl-glob",
            file=sys.stderr,
        )
        sys.exit(2)

    selection_by_id = _load_jsonl_paths(file_paths, last_wins=args.last_wins)
    jsonl_names = {p.name for p in file_paths}

    if not args.no_strict:
        _strict_bidirectional(selection_by_id, run_dir)

    try:
        rows = build_rows(
            run_dir=run_dir,
            selection_by_id=selection_by_id,
            jsonl_basenames=jsonl_names,
            k_default=args.k_requested_default,
            check_metadata=args.check_metadata_trajectory,
            strict=not args.no_strict,
        )
    except (OSError, ValueError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: _sort_key_query_id(str(r["query_id"])))

    if args.push_to_hub and not args.output:
        out_path = None
    elif not args.output:
        print("Error: --output is required (unless only --push-to-hub)", file=sys.stderr)
        sys.exit(2)
    else:
        out_path = args.output

    if out_path is not None:
        suf = out_path.suffix.lower()
        if suf == ".jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif suf == ".json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        elif suf == ".parquet":
            try:
                from datasets import Dataset
            except ImportError as e:
                print(
                    "Error: .parquet output requires: pip install datasets pyarrow",
                    file=sys.stderr,
                )
                raise SystemExit(1) from e
            ds = Dataset.from_list(rows, features=None)
            ds.to_parquet(str(out_path))
        else:
            print(
                f"Error: unsupported --output extension {suf!r} (use .parquet, .jsonl, .json)",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.push_to_hub:
        try:
            from datasets import Dataset
        except ImportError as e:
            print("Error: --push-to-hub requires: pip install datasets", file=sys.stderr)
            raise SystemExit(1) from e
        ds = Dataset.from_list(rows, features=None)
        ds.push_to_hub(args.push_to_hub, split=args.split, commit_message=args.commit_message)

    n = len(rows)
    n_direct = sum(1 for r in rows if r["direct_answer"])
    out_msg = f"Wrote {n} rows ({n_direct} direct_answer), columns: {COLUMNS!r}."
    if out_path is not None:
        out_msg += f" -> {out_path}"
    if args.push_to_hub:
        out_msg += f" Pushed to {args.push_to_hub} (split {args.split!r})."
    print(out_msg)


if __name__ == "__main__":
    main()
