#!/usr/bin/env python3
"""Select k useful tool calls from GLM agent trajectories using Gemini.

GLM original_messages format:
  role=user    — initial message (system prompt + question combined)
  role=assistant — <think>...</think> block + tool_calls: [{id, type, function:{name, arguments}}]
  role=tool    — tool_call_id + content (JSON results array)
  ...
  role=assistant (final, no tool_calls) — final answer

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
    DEFAULT_TOOL_NAMES,
    add_common_args,
    resolve_common_args,
    run_one_om,
    run_pipeline,
)

DEFAULT_GLM_TOOL_NAMES = ("local_knowledge_base_retrieval",)


# ---------------------------------------------------------------------------
# GLM OM helpers
# ---------------------------------------------------------------------------

def _extract_think_block(content: str) -> str:
    m = re.search(r"<think>([\s\S]*?)</think>", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _get_glm_tool_response(messages: List[dict], assistant_idx: int) -> str:
    """Return the content of the next role=tool message after assistant_idx."""
    for i in range(assistant_idx + 1, len(messages)):
        m = messages[i]
        if m.get("role") == "tool":
            return m.get("content", "")
        if m.get("role") == "assistant":
            break
    return ""


def find_candidate_tool_indices_glm(
    trajectory: dict,
    allowed_tool_names: Set[str],
) -> List[int]:
    """Return indices of assistant messages in original_messages that have matching tool_calls."""
    out: List[int] = []
    for i, m in enumerate(trajectory.get("original_messages", [])):
        if m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            if name in allowed_tool_names:
                out.append(i)
                break
    return out


def _preview_tool_step_glm(msg: dict, tool_response: str, preview_chars: int) -> str:
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return "args_hint=(none) | output_preview=''"
    func = tool_calls[0].get("function", {})
    args_raw = func.get("arguments", "{}")
    try:
        ad = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        hint = ad.get("user_query") or ad.get("query") or ad.get("docid") or json.dumps(ad)[:200]
    except (json.JSONDecodeError, TypeError):
        hint = str(args_raw)[:200]
    body = tool_response if isinstance(tool_response, str) else json.dumps(tool_response)
    if len(body) > preview_chars:
        body = body[:preview_chars] + "..."
    return f"args_hint={hint!r} | output_preview={body!r}"


def build_catalog_lines_glm(
    trajectory: dict,
    indices: Sequence[int],
    preview_chars: int,
) -> List[str]:
    messages = trajectory.get("original_messages", [])
    lines: List[str] = []
    for idx in indices:
        msg = messages[idx]
        tc = (msg.get("tool_calls") or [{}])[0]
        name = tc.get("function", {}).get("name", "?")
        tool_resp = _get_glm_tool_response(messages, idx)
        prev = _preview_tool_step_glm(msg, tool_resp, preview_chars)
        lines.append(f"  index={idx}  tool={name}  {prev}")
    return lines


def verbatim_excerpt_for_tool_glm(trajectory: dict, tool_idx: int) -> str:
    """Reasoning (<think> block) + tool call args + tool response."""
    messages = trajectory.get("original_messages", [])
    msg = messages[tool_idx]
    if msg.get("role") != "assistant" or not msg.get("tool_calls"):
        raise ValueError(f"om[{tool_idx}] is not an assistant message with tool_calls")

    parts: List[str] = []

    think = _extract_think_block(msg.get("content", ""))
    if think:
        parts.append(f"[Reasoning]:\n{think}")

    tc = msg["tool_calls"][0]
    func = tc.get("function", {})
    name = func.get("name", "?")
    args = func.get("arguments", "")
    parts.append(f"[Tool call] {name}\narguments:\n{args}")

    tool_resp = _get_glm_tool_response(messages, tool_idx)
    parts.append(f"[Tool result]:\n{tool_resp}")

    return "\n\n".join(parts)


def build_full_excerpt_glm(
    trajectory: dict,
    selected_indices: Sequence[int],
    separator: str = "\n\n---\n\n",
) -> str:
    chunks = [verbatim_excerpt_for_tool_glm(trajectory, i) for i in selected_indices]
    return separator.join(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select k useful tool calls from GLM trajectories via Gemini; "
                    "excerpts from original_messages."
    )
    add_common_args(ap)
    # Override default tool-names help to mention GLM default
    ap.set_defaults(tool_names=None)
    args = ap.parse_args()

    # Default tool names for GLM if not specified
    if args.tool_names is None:
        args.tool_names = list(DEFAULT_GLM_TOOL_NAMES)

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
            find_candidates_fn=find_candidate_tool_indices_glm,
            build_catalog_fn=build_catalog_lines_glm,
            build_excerpt_fn=build_full_excerpt_glm,
        )

    run_pipeline(paths, args.num_threads, job, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
