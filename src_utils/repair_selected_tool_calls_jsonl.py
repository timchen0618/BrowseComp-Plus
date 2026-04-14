#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None

    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :].strip()
        if s.endswith("```"):
            s = s[: -3].strip()

    lbrace = s.find("{")
    rbrace = s.rfind("}")
    if lbrace == -1 or rbrace == -1 or rbrace <= lbrace:
        return None
    return s[lbrace : rbrace + 1]


def _recover_selected_indices_from_raw(record: dict[str, Any]) -> list[int] | None:
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

    # Fallback: raw_response is sometimes truncated (missing closing brace/fence),
    # but still contains a complete selected_indices list we can extract.
    m = re.search(r"\"selected_indices\"\s*:\s*\[(.*?)\]", raw, flags=re.DOTALL)
    if not m:
        return None
    nums = re.findall(r"\b\d+\b", m.group(1))
    if not nums:
        return None
    return [int(x) for x in nums]


def repair_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """
    Returns (total_records, repaired_records, parse_failed_lines_preserved).
    """
    total = 0
    repaired = 0
    preserved_unparseable = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except Exception:
                # Keep the original line verbatim if it's not JSON (should be rare).
                fout.write(raw_line + "\n")
                preserved_unparseable += 1
                continue

            if not isinstance(record, dict):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                continue

            total += 1

            selected = record.get("selected_indices", None)
            needs_repair = not (isinstance(selected, list) and all(isinstance(x, int) for x in selected))
            if needs_repair:
                recovered = _recover_selected_indices_from_raw(record)
                if recovered is not None:
                    record["selected_indices"] = recovered
                    repaired += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return total, repaired, preserved_unparseable


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair selected_tool_calls JSONL by recovering selected_indices.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path (repaired).")
    args = parser.parse_args()

    total, repaired, preserved = repair_file(args.input, args.output)
    print(f"{args.input}: total={total} repaired={repaired} preserved_unparseable={preserved} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

