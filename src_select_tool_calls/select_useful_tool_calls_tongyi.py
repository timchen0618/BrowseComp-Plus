#!/usr/bin/env python3
"""Select k useful tool calls from Tongyi agent trajectories using Gemini.

Tongyi original_messages format:
  role=system    — system prompt (index 0)
  role=user      — user query (index 1)
  role=assistant — <think>...</think> + <tool_call>{"name":..., "arguments":...}</tool_call>
  role=user      — <tool_response>...</tool_response>  (follows each tool call)
  ...
  role=assistant (final) — <think>...</think> + <answer>...</answer>

Default tool filter: search only. Tongyi also emits visit and google_scholar calls, but
visit calls consistently fail with "Tool visit not found" — exclude them by default.
Override with --tool-names to include other tools.

Resume: trajectories whose query_id already has a successful row in --output are skipped.
Use --no-skip-completed to reprocess all; use --skip-seen to skip any already-attempted row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from tool_call_utils import (  # noqa: E402
    add_common_args,
    resolve_common_args,
    run_one_om,
    run_pipeline,
)

DEFAULT_TONGYI_TOOL_NAMES = ("search",)


# ---------------------------------------------------------------------------
# Tongyi OM helpers
# ---------------------------------------------------------------------------

def _extract_think_block(content: str) -> str:
    m = re.search(r"<think>([\s\S]*?)</think>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_tool_call_block(content: str) -> Optional[dict]:
    """Extract and parse the JSON payload inside <tool_call>...</tool_call>."""
    m = re.search(r"<tool_call>([\s\S]*?)</tool_call>", content, re.DOTALL)
    if not m:
        return None
    text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tongyi sometimes emits unquoted name values: {"name": search, ...}
        fixed = re.sub(r'"name"\s*:\s*([a-zA-Z_]\w*)', r'"name": "\1"', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _get_tongyi_tool_response(messages: List[dict], assistant_idx: int) -> str:
    """Return the tool response text from the user message immediately following assistant_idx."""
    next_idx = assistant_idx + 1
    if next_idx >= len(messages):
        return ""
    next_msg = messages[next_idx]
    if next_msg.get("role") != "user":
        return ""
    content = next_msg.get("content", "")
    m = re.search(r"<tool_response>([\s\S]*?)</tool_response>", content, re.DOTALL)
    return m.group(1).strip() if m else content


def find_candidate_tool_indices_tongyi(
    trajectory: dict,
    allowed_tool_names: Set[str],
) -> List[int]:
    """Return indices of assistant messages that contain a matching <tool_call>."""
    out: List[int] = []
    for i, m in enumerate(trajectory.get("original_messages", [])):
        if m.get("role") != "assistant":
            continue
        tc = _parse_tool_call_block(m.get("content", ""))
        if tc is None:
            continue
        if tc.get("name", "") in allowed_tool_names:
            out.append(i)
    return out


def _preview_tool_step_tongyi(msg: dict, tool_response: str, preview_chars: int) -> str:
    tc = _parse_tool_call_block(msg.get("content", ""))
    if tc is None:
        return "args_hint=(unparseable) | output_preview=''"
    args = tc.get("arguments", {})
    if isinstance(args, dict):
        hint = args.get("query") or args.get("user_query") or json.dumps(args)[:200]
    elif isinstance(args, str):
        hint = args[:200]
    else:
        hint = str(args)[:200]
    body = tool_response
    if len(body) > preview_chars:
        body = body[:preview_chars] + "..."
    return f"args_hint={hint!r} | output_preview={body!r}"


def build_catalog_lines_tongyi(
    trajectory: dict,
    indices: Sequence[int],
    preview_chars: int,
) -> List[str]:
    messages = trajectory.get("original_messages", [])
    lines: List[str] = []
    for idx in indices:
        msg = messages[idx]
        tc = _parse_tool_call_block(msg.get("content", ""))
        name = tc.get("name", "?") if tc else "?"
        tool_resp = _get_tongyi_tool_response(messages, idx)
        prev = _preview_tool_step_tongyi(msg, tool_resp, preview_chars)
        lines.append(f"  index={idx}  tool={name}  {prev}")
    return lines


def verbatim_excerpt_for_tool_tongyi(trajectory: dict, tool_idx: int) -> str:
    """<think> reasoning + verbatim <tool_call> block + tool response."""
    messages = trajectory.get("original_messages", [])
    msg = messages[tool_idx]
    content = msg.get("content", "")

    parts: List[str] = []

    think = _extract_think_block(content)
    if think:
        parts.append(f"[Reasoning]:\n{think}")

    m = re.search(r"<tool_call>[\s\S]*?</tool_call>", content, re.DOTALL)
    if m:
        parts.append(f"[Tool call]:\n{m.group(0)}")

    tool_resp = _get_tongyi_tool_response(messages, tool_idx)
    parts.append(f"[Tool result]:\n{tool_resp}")

    return "\n\n".join(parts)


def build_full_excerpt_tongyi(
    trajectory: dict,
    selected_indices: Sequence[int],
    separator: str = "\n\n---\n\n",
) -> str:
    chunks = [verbatim_excerpt_for_tool_tongyi(trajectory, i) for i in selected_indices]
    return separator.join(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select k useful tool calls from Tongyi trajectories via Gemini; "
                    "excerpts from original_messages."
    )
    add_common_args(ap)
    args = ap.parse_args()

    # Default tool names for Tongyi if not specified
    if args.tool_names is None:
        args.tool_names = list(DEFAULT_TONGYI_TOOL_NAMES)

    paths, allowed, query_by_id, model, gen_params = resolve_common_args(args)

    def job(p):
        return run_one_om(
            p,
            model,
            gen_params,
            k=args.k,
            allowed_tool_names=allowed,
            preview_chars=args.preview_chars,
            context_max_chars=args.context_max_chars,
            context_reasoning_max=args.context_reasoning_max_chars,
            context_tool_max=args.context_tool_max_chars,
            dry_run=args.dry_run,
            query_by_id=query_by_id,
            find_candidates_fn=find_candidate_tool_indices_tongyi,
            build_catalog_fn=build_catalog_lines_tongyi,
            build_excerpt_fn=build_full_excerpt_tongyi,
        )

    run_pipeline(paths, args.num_threads, job, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
