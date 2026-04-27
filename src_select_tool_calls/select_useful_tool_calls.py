#!/usr/bin/env python3
"""
Select k useful tool calls from agent trajectories using Gemini (Portkey) for
selection and Python for verbatim excerpts (reasoning + exact tool args/outputs).

See portkey.py for Gemini inference setup (PORTKEY_API_KEY, optional PORTKEY_BASE_URL).

Resume: when using --output, trajectories whose query_id already has a successful row in that
JSONL are skipped by default. Use --no-skip-completed to reprocess all; use --skip-seen to skip
any query_id already present (including failed rows).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set  # noqa: F401 (re-exported for random_select_tool_calls)

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from tool_call_utils import (  # noqa: E402
    DEFAULT_BCP_QUERIES_TSV,
    DEFAULT_TOOL_NAMES,
    SYSTEM_PROMPT_TEMPLATE,
    Gemini25Pro,
    GenParams,
    tqdm,
    _reasoning_text,
    _preview_tool_step,
    find_candidate_tool_indices,
    previous_tool_index,
    build_catalog_lines,
    verbatim_excerpt_for_tool,
    build_full_excerpt,
    format_trajectory_for_prompt,
    parse_json_response,
    validate_and_sort_indices,
    load_trajectory_files,
    load_query_ids_from_output_jsonl,
    read_query_id_from_trajectory,
    filter_paths_by_query_ids,
    load_query_id_to_text,
    question_from_bcp_map,
    parse_tool_names_arg,
    run_one_om,
    run_pipeline,
)

def format_original_messages_for_prompt(
    trajectory: dict,
    *,
    max_chars: int = 0,
) -> str:
    """Serialize a trajectory's original_messages into a plain string.

    Dumps the entire original_messages list as a JSON string, preserving the
    original structure without any reformatting.
    """
    msgs = trajectory.get("original_messages", [])
    result = json.dumps(msgs, ensure_ascii=False) if msgs else "(no original messages)"
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (messages truncated)"
    return result




def _truncate_om_reasoning_content_chunk(chunk: Any, max_chars: int) -> Any:
    """Shorten a single item in a reasoning message content list (strings or content-part dicts)."""
    if max_chars <= 0:
        return chunk
    if isinstance(chunk, str):
        if len(chunk) > max_chars:
            return chunk[:max_chars] + "..."
        return chunk
    if isinstance(chunk, dict) and "text" in chunk and isinstance(chunk.get("text"), str):
        t = chunk["text"]
        if len(t) > max_chars:
            out = copy.deepcopy(chunk)
            out["text"] = t[:max_chars] + "..."
            return out
        return chunk
    # Fallback: stringify non-string parts (e.g. unexpected shapes) for safe truncation
    s = (
        json.dumps(chunk, ensure_ascii=False)
        if isinstance(chunk, (dict, list))
        else str(chunk)
    )
    if len(s) > max_chars:
        return s[:max_chars] + "..."
    return chunk


def format_trajectory_for_prompt_orig(
    trajectory: dict,
    *,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
    query_max_chars: int = 1000,
) -> str:
    """Same structure as search_agent/oss_client._format_trajectory_for_prompt (truncation optional)."""
    parts: List[str] = []
    for step in trajectory.get("original_messages", []):
        # clone the whole thing to keep the original messages intact
        step_clone = copy.deepcopy(step)
        role = step_clone.get("role", "")
        stype = step_clone.get("type", "")
        if role == "user":
            try:
                parts.append(json.dumps(step_clone, ensure_ascii=False))
            except Exception as e:
                print(f"Error dumping user step: {e}")
        else:
            if stype == "reasoning":
                if "content" in step_clone and isinstance(step_clone["content"], list):
                    if reasoning_max_chars > 0:
                        for i in range(len(step_clone["content"])):
                            step_clone["content"][i] = _truncate_om_reasoning_content_chunk(
                                step_clone["content"][i], reasoning_max_chars
                            )
                try:
                    parts.append(json.dumps(step_clone, ensure_ascii=False))
                except Exception as e:
                    print(f"Error dumping reasoning step: {e}")
            elif stype == "function_call":
                try:
                    parts.append(json.dumps(step_clone, ensure_ascii=False))
                except Exception as e:
                    print(f"Error dumping function call step: {e}")
            elif stype == "function_call_output":
                if "output" in step_clone and isinstance(step_clone["output"], str):
                    if tool_output_max_chars > 0 and len(step["output"]) > tool_output_max_chars:
                        step_clone["output"] = step_clone["output"][:tool_output_max_chars] + "..."
                try:
                    parts.append(json.dumps(step_clone, ensure_ascii=False))
                except Exception as e:
                    print(f"Error dumping function call output step: {e}")
            elif stype == "message":
                try:
                    parts.append(json.dumps(step_clone, ensure_ascii=False))
                except Exception as e:
                    print(f"Error dumping message step: {e}")
            else:
                print(f"Unknown step type: {stype}")
                print(step)
                try:
                    parts.append(json.dumps(step_clone, ensure_ascii=False))
                except Exception as e:
                    print(f"Error dumping unknown step: {e}")
                

    result = "\n\n".join(parts) if parts else "(no trajectory steps)"
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (trajectory truncated)"
    return result


# ---------------------------------------------------------------------------
# original_messages helpers for gpt-oss-120b (OpenAI function_call format)
# ---------------------------------------------------------------------------


def _om_build_call_id_to_output(messages: List[dict]) -> dict:
    """Map call_id -> output string from all function_call_output items."""
    mapping: dict = {}
    for m in messages:
        if m.get("type") == "function_call_output":
            cid = m.get("call_id", "")
            if cid:
                mapping[cid] = json.dumps(m, ensure_ascii=False)
    return mapping

def find_candidate_tool_indices_om(
    trajectory: dict,
    allowed_tool_names: Set[str],
) -> List[int]:
    """Like find_candidate_tool_indices but indexes into original_messages."""
    out: List[int] = []
    for i, item in enumerate(trajectory.get("original_messages", [])):
        if item.get("type") != "function_call":
            continue
        name = item.get("name") or ""
        if name in allowed_tool_names:
            out.append(i)
    return out


def _preview_tool_step_om(
    step: dict,
    preview_chars: int,
    call_id_to_output: dict,
) -> str:
    """Like _preview_tool_step but for an original_messages function_call item."""
    args_raw = step.get("arguments", "{}")
    try:
        ad = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        if isinstance(ad, dict):
            hint = ad.get("user_query") or ad.get("query") or ad.get("docid") or json.dumps(ad)[:200]
        else:
            hint = str(ad)[:200]
    except (json.JSONDecodeError, TypeError):
        hint = str(args_raw)[:200]
    cid = step.get("call_id", "")
    body = call_id_to_output.get(cid, "")
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)

    if len(body) > preview_chars:
        output_preview = body[:preview_chars] + "..."
    else:
        output_preview = body
    return f"args_hint={hint!r} | output_preview={output_preview!r}"


def build_catalog_lines_om(
    trajectory: dict,
    indices: Sequence[int],
    preview_chars: int,
) -> List[str]:
    """Like build_catalog_lines but indexes into original_messages."""
    messages = trajectory.get("original_messages", [])
    call_id_to_output = _om_build_call_id_to_output(messages)
    lines: List[str] = []
    for idx in indices:
        step = messages[idx]
        name = step.get("name", "?")
        prev = _preview_tool_step_om(step, preview_chars, call_id_to_output)
        lines.append(f"  index={idx}  tool={name}  {prev}")
    return lines


def _om_previous_tool_index(messages: List[dict], tool_idx: int) -> int:
    """Like previous_tool_index but scans for type=='function_call'."""
    for j in range(tool_idx - 1, -1, -1):
        if messages[j].get("type") == "function_call":
            return j
    return -1


def verbatim_excerpt_for_tool_om(
    trajectory: dict,
    tool_idx: int,
) -> str:
    """Like verbatim_excerpt_for_tool but reads from original_messages."""
    messages = trajectory.get("original_messages", [])
    if tool_idx < 0 or tool_idx >= len(messages):
        raise IndexError(f"tool_idx {tool_idx} out of range")
    step = messages[tool_idx]
    if step.get("type") != "function_call":
        raise ValueError(f"step {tool_idx} is not function_call")

    call_id_to_output = _om_build_call_id_to_output(messages)
    prev = _om_previous_tool_index(messages, tool_idx)
    parts: List[str] = []

    for i in range(prev + 1, tool_idx):
        s = messages[i]
        if s.get("type") == "reasoning":
            parts.append(json.dumps(s, ensure_ascii=False))

    # tool_name = step.get("name", "?")
    # args_raw = step.get("arguments", "")
    # if isinstance(args_raw, str):
    #     args_display = args_raw
    # else:
    #     args_display = json.dumps(args_raw, ensure_ascii=False)
    # parts.append(f"[Tool call] {tool_name}\narguments:\n{args_display}")
    parts.append(json.dumps(step, ensure_ascii=False))

    cid = step.get("call_id", "")
    out_str = call_id_to_output.get(cid, "")
    if not isinstance(out_str, str):
        out_str = json.dumps(out_str, ensure_ascii=False)
    parts.append(f"{out_str}")

    return "\n\n".join(parts)


def build_full_excerpt_om(
    trajectory: dict,
    selected_indices: Sequence[int],
    separator: str = "\n\n---\n\n",
) -> str:
    """Like build_full_excerpt but reads from original_messages."""
    chunks = [verbatim_excerpt_for_tool_om(trajectory, i) for i in selected_indices]
    return separator.join(chunks)


def main() -> None:
    from tool_call_utils import add_common_args, resolve_common_args
    ap = argparse.ArgumentParser(
        description="Select k useful tool calls via Gemini; verbatim excerpts in output."
    )
    add_common_args(ap)
    ap.add_argument(
        "--use-original-messages",
        action="store_true",
        help="Use original_messages (OpenAI function_call format) for excerpts",
    )
    args = ap.parse_args()

    paths, allowed, query_by_id, model, gen_params = resolve_common_args(args)

    if args.use_original_messages:
        find_fn, catalog_fn, excerpt_fn = (
            find_candidate_tool_indices_om,
            build_catalog_lines_om,
            build_full_excerpt_om,
        )
        ctx_fn = format_trajectory_for_prompt_orig
        req_om = True
    else:
        find_fn, catalog_fn, excerpt_fn = (
            find_candidate_tool_indices,
            build_catalog_lines,
            build_full_excerpt,
        )
        ctx_fn = None
        req_om = False

    def job(p: Path) -> dict:
        return run_one_om(
            p, model, gen_params,
            k=args.k,
            allowed_tool_names=allowed,
            preview_chars=args.preview_chars,
            context_max_chars=args.context_max_chars,
            context_reasoning_max=args.context_reasoning_max_chars,
            context_tool_max=args.context_tool_max_chars,
            dry_run=args.dry_run,
            query_by_id=query_by_id,
            find_candidates_fn=find_fn,
            build_catalog_fn=catalog_fn,
            build_excerpt_fn=excerpt_fn,
            format_context_fn=ctx_fn,
            require_original_messages=req_om,
        )

    run_pipeline(paths, args.num_threads, job, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
