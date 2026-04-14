#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_json_object(text: str) -> str | None:
    """
    Try to recover a JSON object from a model response that may be wrapped in
    Markdown fences (```json ... ```), or may have extra leading/trailing text.
    """
    if not text:
        return None

    s = text.strip()
    if s.startswith("```"):
        # Drop the first fence line (``` or ```json).
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :].strip()
        # Drop trailing fence if present.
        if s.endswith("```"):
            s = s[: -3].strip()

    lbrace = s.find("{")
    rbrace = s.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
        return None

    return s[lbrace : rbrace + 1]


def _recover_selected_indices(record: dict[str, Any]) -> list[int] | None:
    """
    Some records may have `error: json_parse_failed` and the real JSON embedded
    in `raw_response`. Recover selected_indices from there when possible.
    """
    raw = record.get("raw_response", None)
    if not isinstance(raw, str):
        return None

    obj_text = _extract_json_object(raw)
    if obj_text is not None:
        try:
            parsed = json.loads(obj_text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            selected = parsed.get("selected_indices", None)
            if isinstance(selected, list) and all(isinstance(x, int) for x in selected):
                return selected

    # Fallback for truncated fenced JSON (missing closing brace/fence).
    m = re.search(r"\"selected_indices\"\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    if not m:
        return None
    nums = re.findall(r"\b\d+\b", m.group(1))
    if not nums:
        return None
    return [int(x) for x in nums]


def _has_any_tool_calls(record: dict[str, Any]) -> bool:
    """
    For selected_tool_calls JSONL, treat a record as "empty" if it contains
    no selected tool call indices at all.
    """
    selected = record.get("selected_indices", None)
    if not isinstance(selected, list):
        recovered = _recover_selected_indices(record)
        if recovered is None:
            return False
        selected = recovered
        record["selected_indices"] = recovered

    return len(selected) > 0


def filter_file(input_path: Path, output_path: Path, dropped_path: Path | None) -> tuple[int, int]:
    kept = 0
    dropped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dropped_path is not None:
        dropped_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        dropped_f = dropped_path.open("w", encoding="utf-8") if dropped_path is not None else None
        try:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    # Preserve unknown/unparseable lines (safer than silently dropping).
                    fout.write(line + "\n")
                    kept += 1
                    continue

                if isinstance(record, dict) and _has_any_tool_calls(record):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped += 1
                    if dropped_f is not None:
                        qid = record.get("query_id", None) if isinstance(record, dict) else None
                        src = record.get("source_file", None) if isinstance(record, dict) else None
                        dropped_f.write(json.dumps({"query_id": qid, "source_file": src}, ensure_ascii=False) + "\n")
        finally:
            if dropped_f is not None:
                dropped_f.close()

    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter out JSONL records with no tool calls selected.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path (filtered).")
    parser.add_argument(
        "--dropped",
        type=Path,
        default=None,
        help="Optional JSONL path to write dropped records (query_id + source_file).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file (writes to a temp file then replaces).",
    )
    args = parser.parse_args()

    if args.inplace:
        tmp_out = args.input.with_suffix(args.input.suffix + ".tmp")
        kept, dropped = filter_file(args.input, tmp_out, args.dropped)
        tmp_out.replace(args.input)
        print(f"{args.input}: kept={kept} dropped={dropped} (in-place)")
    else:
        kept, dropped = filter_file(args.input, args.output, args.dropped)
        print(f"{args.input}: kept={kept} dropped={dropped} -> {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

