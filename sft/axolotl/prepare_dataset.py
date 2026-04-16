#!/usr/bin/env python3
"""
Convert a selected-tool-calls JSONL into the messages-style JSONL that
Axolotl's `type: chat_template` dataset expects.

Input line shape (e.g. selected_tool_calls/*.jsonl):
    {"source_file": "run_XXX.json",
     "excerpt": "<JSON item 1>\\n\\n<JSON item 2>\\n\\n...",
     ...}

where each JSON item in `excerpt` is an OpenAI Responses-API item of type
`reasoning`, `function_call`, or `function_call_output`.

The system + user prompt is pulled from the source trajectory at
`<trajectory_folder>/<source_file>` via its `original_messages[0]` entry
(a single merged `user` message containing both the system prompt and
the "User: <question>" line).

Output line shape (one JSON object per line):
    {"messages": [
        {"role": "user",      "content": "<system+question merged>"},
        {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
        {"role": "user",      "content": "<tool_response>...</tool_response>"},
        ...
    ]}

Loss masking happens in Axolotl (`roles_to_train: ["assistant"]`); this
script only reshapes data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Source-trajectory loading
# ---------------------------------------------------------------------------

class _SourceTrajectoryCache:
    """Tiny cache so repeated source_files aren't re-read from disk."""

    def __init__(self, trajectory_folder: Path) -> None:
        self.trajectory_folder = trajectory_folder
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def load(self, source_file: str) -> Optional[Dict[str, Any]]:
        if source_file in self._cache:
            return self._cache[source_file]
        path = self.trajectory_folder / source_file
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            self._cache[source_file] = None
            return None
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse source trajectory {path}: {e}")
            self._cache[source_file] = None
            return None
        self._cache[source_file] = data
        return data


def _system_user_from_source(traj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    The first item of original_messages is a `{role: "user"}` entry whose
    content already concatenates the system prompt and the "User: <question>"
    line. We keep it as a single merged user message.
    """
    om = traj.get("original_messages")
    if not isinstance(om, list) or not om:
        return None
    first = om[0]
    if not isinstance(first, dict):
        return None
    role = str(first.get("role", "")).lower()
    content = first.get("content", "")
    if role != "user" or not isinstance(content, str) or not content.strip():
        return None
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Excerpt -> messages
# ---------------------------------------------------------------------------

def _parse_excerpt_items(excerpt: str) -> List[Dict[str, Any]]:
    """Split an excerpt string into a list of Responses-API item dicts."""
    items: List[Dict[str, Any]] = []
    for chunk in excerpt.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "type" in obj:
            items.append(obj)
    return items


def _reasoning_text(item: Dict[str, Any]) -> str:
    """Extract text from a Responses-API reasoning item."""
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for c in content:
        if isinstance(c, dict):
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t)
    return "\n".join(parts).strip()


def _fmt_tool_call(item: Dict[str, Any]) -> str:
    """Render a function_call item as an inline <tool_call>...</tool_call>."""
    name = item.get("name", "")
    raw_args = item.get("arguments", "")
    if isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = raw_args
    else:
        parsed_args = raw_args
    payload = {"name": name, "arguments": parsed_args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def _fmt_tool_response(item: Dict[str, Any]) -> str:
    """Render a function_call_output item as a <tool_response> user turn."""
    out = item.get("output", "")
    if not isinstance(out, str):
        out = json.dumps(out, ensure_ascii=False)
    return "<tool_response>\n" + out + "\n</tool_response>"


def _excerpt_to_messages(excerpt: str) -> List[Dict[str, str]]:
    """
    Walk the Responses-API items into Qwen-style inline chat messages.

    Rules:
      - `reasoning`           -> append text to the current assistant buffer
      - `function_call`       -> append <tool_call>...</tool_call>, flush
                                 buffer as one assistant message
      - `function_call_output`-> flush any pending assistant buffer, then
                                 emit a user <tool_response>...</tool_response>
    """
    items = _parse_excerpt_items(excerpt)
    if not items:
        return []

    messages: List[Dict[str, str]] = []
    buf: List[str] = []

    def flush_assistant() -> None:
        if not buf:
            return
        text = "\n\n".join(s for s in buf if s).strip()
        buf.clear()
        if text:
            messages.append({"role": "assistant", "content": text})

    for it in items:
        kind = it.get("type")
        if kind == "reasoning":
            text = _reasoning_text(it)
            if text:
                buf.append(text)
        elif kind == "function_call":
            buf.append(_fmt_tool_call(it))
            flush_assistant()
        elif kind == "function_call_output":
            flush_assistant()
            messages.append({"role": "user", "content": _fmt_tool_response(it)})
        # Unknown item types are ignored intentionally.

    flush_assistant()
    return messages


def _coerce_excerpt(
    example: Dict[str, Any],
    source_cache: _SourceTrajectoryCache,
) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """
    Build messages for a (source_file, excerpt) record.

    Returns (messages | None, reason). `reason` is one of:
      "ok", "schema", "missing_source", "bad_excerpt".
    """
    if "source_file" not in example or "excerpt" not in example:
        return None, "schema"

    traj = source_cache.load(str(example["source_file"]))
    if traj is None:
        return None, "missing_source"

    prompt = _system_user_from_source(traj)
    if prompt is None:
        return None, "missing_source"

    excerpt_msgs = _excerpt_to_messages(str(example["excerpt"]))
    if not excerpt_msgs:
        return None, "bad_excerpt"

    has_tool_call = any(
        m["role"] == "assistant" and "<tool_call>" in m["content"]
        for m in excerpt_msgs
    )
    if not has_tool_call:
        return None, "bad_excerpt"

    return [prompt] + excerpt_msgs, "ok"


# ---------------------------------------------------------------------------
# I/O helpers & driver
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{line_num} could not parse: {e}")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a selected-tool-calls JSONL with {source_file, excerpt, ...} rows.",
    )
    parser.add_argument(
        "--trajectory-folder",
        type=Path,
        required=True,
        help=(
            "Folder containing the source trajectory JSON files referenced "
            "by each record's `source_file` field."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sft/axolotl/data"),
        help="Directory to write train.jsonl (and val.jsonl if --val-size>0).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of examples held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the train/val split.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"--input not found: {args.input}")
    if not args.trajectory_folder.is_dir():
        parser.error(
            f"--trajectory-folder not found or not a dir: {args.trajectory_folder}"
        )

    source_cache = _SourceTrajectoryCache(args.trajectory_folder)

    raw_total = 0
    kept: List[Dict[str, Any]] = []
    dropped_schema = 0
    dropped_missing_source = 0
    dropped_bad_excerpt = 0

    for record in _iter_jsonl(args.input):
        raw_total += 1
        messages, reason = _coerce_excerpt(record, source_cache)
        if messages is None:
            if reason == "schema":
                dropped_schema += 1
            elif reason == "missing_source":
                dropped_missing_source += 1
            else:
                dropped_bad_excerpt += 1
            continue
        kept.append({"messages": messages})

    if not kept:
        raise SystemExit(
            f"No usable examples found in {args.input}. "
            f"dropped_schema={dropped_schema} "
            f"dropped_missing_source={dropped_missing_source} "
            f"dropped_bad_excerpt={dropped_bad_excerpt}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    n_val = int(round(len(kept) * args.val_size)) if args.val_size > 0 else 0
    val, train = kept[:n_val], kept[n_val:]

    train_path = args.output_dir / "train.jsonl"
    _write_jsonl(train_path, train)
    print(f"wrote {len(train):>6} -> {train_path}")

    if n_val > 0:
        val_path = args.output_dir / "val.jsonl"
        _write_jsonl(val_path, val)
        print(f"wrote {len(val):>6} -> {val_path}")

    print(
        "summary: "
        f"read={raw_total} kept={len(kept)} "
        f"dropped_schema={dropped_schema} "
        f"dropped_missing_source={dropped_missing_source} "
        f"dropped_bad_excerpt={dropped_bad_excerpt}"
    )


if __name__ == "__main__":
    main()

    # python sft/axolotl/prepare_dataset.py \
    # --input selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
    # --trajectory-folder runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    # --output-dir sft/axolotl/data \
    # --val-size 0.1 --seed 42