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
from typing import Any, List, Optional, Sequence, Set

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
                            # truncate the content if it is too long
                            step_clone["content"][i] = step_clone["content"][i][:reasoning_max_chars] + "..."
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


def run_one(
    path: Path,
    model: Any,
    gen_params: Any,
    k: int,
    allowed_tool_names: Set[str],
    preview_chars: int,
    context_max_chars: int,
    context_reasoning_max: int,
    context_tool_max: int,
    dry_run: bool,
    use_original_messages: bool = False,
    query_by_id: Optional[dict[str, str]] = None,
) -> dict:
    with path.open(encoding="utf-8") as f:
        traj = json.load(f)

    if use_original_messages and not traj.get("original_messages"):
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": "missing_original_messages",
            "selected_indices": [],
            "rationale": "",
            "excerpt": "",
            "k_requested": k,
            "k_effective": 0,
        }

    if use_original_messages:
        candidates = find_candidate_tool_indices_om(traj, allowed_tool_names)
    else:
        candidates = find_candidate_tool_indices(traj, allowed_tool_names)
    k_eff = min(k, len(candidates))
    if k_eff == 0:
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": "no_candidate_tool_calls",
            "selected_indices": [],
            "rationale": "",
            "excerpt": "",
            "k_requested": k,
            "k_effective": 0,
        }

    if k_eff < k:
        print(
            f"[warn] {path.name}: only {len(candidates)} candidate tool calls; using k={k_eff}",
            file=sys.stderr,
        )

    if use_original_messages:
        catalog_lines = build_catalog_lines_om(traj, candidates, preview_chars)
    else:
        catalog_lines = build_catalog_lines(traj, candidates, preview_chars)
    qid_str = str(traj.get("query_id", "")).strip()
    question = ""
    if query_by_id:
        question = question_from_bcp_map(query_by_id, qid_str)
    if not question:
        question = traj.get("query") or traj.get("question") or ""
    if use_original_messages:
        context_block = format_trajectory_for_prompt_orig(
            traj,
            max_chars=context_max_chars,
            reasoning_max_chars=context_reasoning_max,
            tool_output_max_chars=context_tool_max,
        )
    else:
        context_block = format_trajectory_for_prompt(
            traj,
            max_chars=context_max_chars,
            reasoning_max_chars=context_reasoning_max,
            tool_output_max_chars=context_tool_max,
        )

    user_parts = [
        f"User question:\n{question}\n",
        f"K = {k_eff} (you must return exactly {k_eff} indices).",
        "Candidate tool calls (choose by index=...):",
        *catalog_lines,
        "\nTrajectory context (may be truncated for length; use indices from the catalog only):",
        context_block,
    ]
    user_content = "\n".join(user_parts)

    system = SYSTEM_PROMPT_TEMPLATE.format(k=k_eff)

    if dry_run:
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "dry_run": True,
            "user_content_chars": len(user_content),
            "candidate_count": len(candidates),
            "k_effective": k_eff,
        }

    print('-'*100)
    print(user_content)
    print('-'*100)
    raw = model.generate(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        params=gen_params,
    )

    try:
        parsed = parse_json_response(raw)
    except Exception as e:
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": f"json_parse_failed: {e}",
            "raw_response": raw[:2000],
            "k_effective": k_eff,
        }

    sel = parsed.get("selected_indices")
    rationale = parsed.get("rationale", "")
    if not isinstance(sel, list):
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": "selected_indices_not_a_list",
            "raw_response": raw[:2000],
            "k_effective": k_eff,
            "candidates": candidates,
        }

    try:
        sel_ints = [int(x) for x in sel]
        valid_set = set(candidates)
        ordered = validate_and_sort_indices(sel_ints, k_eff, valid_set)
    except Exception as e:
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": str(e),
            "raw_response": raw[:2000],
            "k_effective": k_eff,
            "candidates": candidates,
        }

    if use_original_messages:
        excerpt = build_full_excerpt_om(traj, ordered)
    else:
        excerpt = build_full_excerpt(traj, ordered)
    return {
        "query_id": str(traj.get("query_id", "")),
        "source_file": path.name,
        "selected_indices": ordered,
        "rationale": rationale,
        "excerpt": excerpt,
        "candidates": candidates,
        "correct_num_selected": len(ordered) == k_eff,
        "k_requested": k,
        "k_effective": k_eff,
    }


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

    def job(p: Path) -> dict:
        return run_one(
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
            use_original_messages=args.use_original_messages,
            query_by_id=query_by_id,
        )

    run_pipeline(paths, args.num_threads, job, args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
