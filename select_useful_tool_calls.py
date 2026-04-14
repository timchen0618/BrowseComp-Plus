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
import os
import re
import sys
import threading
import concurrent.futures
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kw):  # type: ignore[misc]
        return it

# Repo-local: same directory as this script
_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BCP_QUERIES_TSV = _REPO_ROOT / "topics-qrels" / "bcp" / "queries.tsv"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from portkey import Gemini25Pro, GenParams
except ImportError:
    Gemini25Pro = None  # type: ignore[misc, assignment]
    GenParams = None  # type: ignore[misc, assignment]

DEFAULT_TOOL_NAMES = (
    "local_knowledge_base_retrieval",
    "search",
    "get_document",
)

SYSTEM_PROMPT_TEMPLATE = """You are helping curate training or analysis data from web-search agent trajectories.

Your task: choose exactly {k} tool-call steps (by step index) that are the most *useful* for understanding how the agent made good retrieval decisions. Prefer:
- Steps that pivot strategy or refine queries after weak results
- Steps that retrieve clearly relevant evidence for the user question
- Diversity across the episode (early/middle/late), not redundant near-duplicates

Respond with a single JSON object only (no markdown fences), with this exact shape:
{{"selected_indices": [<int>, ...], "rationale": "<short explanation>"}}

Rules:
- "selected_indices" must list exactly {k} distinct integers, each a valid candidate index from the catalog.
- Indices must appear in ascending trajectory order in your array.
- Use only indices from the provided catalog; do not invent indices.
"""


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


def format_trajectory_for_prompt(
    trajectory: dict,
    *,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
    query_max_chars: int = 1000,
) -> str:
    """Same structure as search_agent/oss_client._format_trajectory_for_prompt (truncation optional)."""
    parts: List[str] = []
    for step in trajectory.get("result", []):
        stype = step.get("type")
        if stype == "reasoning":
            output = step.get("output", [])
            text = " ".join(str(o) for o in output) if isinstance(output, list) else str(output)
            text = text.strip()
            if text:
                if reasoning_max_chars > 0 and len(text) > reasoning_max_chars:
                    text = text[:reasoning_max_chars] + "..."
                parts.append(f"[Reasoning]: {text}")
        elif stype == "tool_call":
            tool_name = step.get("tool_name", "?")
            args = step.get("arguments", "{}")
            try:
                args_dict = json.loads(args) if isinstance(args, str) else args
                q = args_dict.get("user_query") or args_dict.get("query") or str(args)[:query_max_chars]
            except (json.JSONDecodeError, AttributeError, TypeError):
                q = str(args)[:query_max_chars]
            parts.append(f"[Tool call] {tool_name}: {q}")
            output = step.get("output")
            if output is not None:
                out_str = json.dumps(output) if not isinstance(output, str) else output
                limit = tool_output_max_chars
                parts.append(f"[Tool result]: {out_str[:limit]}{'...' if len(out_str) > limit else ''}")
        elif stype == "output_text":
            text = str(step.get("output", "")).strip()
            if text:
                parts.append(f"[Final answer]: {text}")

    result = "\n\n".join(parts) if parts else "(no trajectory steps)"
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (trajectory truncated)"
    return result


def _reasoning_text(step: dict) -> str:
    output = step.get("output", [])
    if isinstance(output, list):
        return " ".join(str(o) for o in output)
    return str(output or "")


def find_candidate_tool_indices(
    trajectory: dict,
    allowed_tool_names: Set[str],
) -> List[int]:
    out: List[int] = []
    for i, step in enumerate(trajectory.get("result", [])):
        if step.get("type") != "tool_call":
            continue
        name = step.get("tool_name") or ""
        if name in allowed_tool_names:
            out.append(i)
    return out


def _preview_tool_step(step: dict, preview_chars: int) -> str:
    args_raw = step.get("arguments", "{}")
    try:
        ad = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        if isinstance(ad, dict):
            hint = ad.get("user_query") or ad.get("query") or ad.get("docid") or json.dumps(ad)[:200]
        else:
            hint = str(ad)[:200]
    except (json.JSONDecodeError, TypeError):
        hint = str(args_raw)[:200]
    out = step.get("output")
    if out is None:
        body = ""
    elif isinstance(out, str):
        body = out
    else:
        body = json.dumps(out, ensure_ascii=False)
    if len(body) > preview_chars:
        body = body[:preview_chars] + "..."
    return f"args_hint={hint!r} | output_preview={body!r}"


def build_catalog_lines(
    trajectory: dict,
    indices: Sequence[int],
    preview_chars: int,
) -> List[str]:
    lines: List[str] = []
    result = trajectory.get("result", [])
    for idx in indices:
        step = result[idx]
        name = step.get("tool_name", "?")
        prev = _preview_tool_step(step, preview_chars)
        lines.append(f"  index={idx}  tool={name}  {prev}")
    return lines


def previous_tool_index(result: List[dict], tool_idx: int) -> int:
    for j in range(tool_idx - 1, -1, -1):
        if result[j].get("type") == "tool_call":
            return j
    return -1


def verbatim_excerpt_for_tool(
    trajectory: dict,
    tool_idx: int,
) -> str:
    """Reasoning steps since previous tool_call + full tool_call at tool_idx (exact args/output)."""
    result = trajectory.get("result", [])
    if tool_idx < 0 or tool_idx >= len(result):
        raise IndexError(f"tool_idx {tool_idx} out of range")
    step = result[tool_idx]
    if step.get("type") != "tool_call":
        raise ValueError(f"step {tool_idx} is not tool_call")

    prev = previous_tool_index(result, tool_idx)
    parts: List[str] = []

    for i in range(prev + 1, tool_idx):
        s = result[i]
        if s.get("type") == "reasoning":
            t = _reasoning_text(s).strip()
            if t:
                parts.append(f"[Reasoning]: {t}")

    tool_name = step.get("tool_name", "?")
    args_raw = step.get("arguments", "")
    if isinstance(args_raw, str):
        args_display = args_raw
    else:
        args_display = json.dumps(args_raw, ensure_ascii=False)
    parts.append(f"[Tool call] {tool_name}\narguments:\n{args_display}")

    out = step.get("output")
    if out is None:
        out_str = ""
    elif isinstance(out, str):
        out_str = out
    else:
        out_str = json.dumps(out, ensure_ascii=False)
    parts.append(f"[Tool result]:\n{out_str}")

    return "\n\n".join(parts)


def build_full_excerpt(
    trajectory: dict,
    selected_indices: Sequence[int],
    separator: str = "\n\n---\n\n",
) -> str:
    chunks = [verbatim_excerpt_for_tool(trajectory, i) for i in selected_indices]
    return separator.join(chunks)


# ---------------------------------------------------------------------------
# original_messages helpers — mirror the result-based helpers above but
# operate on the raw API message list stored in trajectory["original_messages"].
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


def parse_json_response(text: str) -> dict:
    text = text.strip()

    # Strip markdown fences if present. Handle both complete fences (```json ... ```)
    # and truncated outputs that start with ```json but never close.
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :].strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    else:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    # First try strict JSON parsing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: parse the first complete {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: recover `selected_indices` from incomplete JSON (common failure mode).
    m_sel = re.search(r"\"selected_indices\"\s*:\s*\[(.*?)\]", text, flags=re.DOTALL)
    if not m_sel:
        raise json.JSONDecodeError("Could not parse or recover selected_indices", text, 0)
    nums = re.findall(r"\b\d+\b", m_sel.group(1))
    if not nums:
        raise json.JSONDecodeError("Recovered selected_indices was empty", text, 0)
    selected_indices = [int(x) for x in nums]

    # Rationale is optional; it may be truncated, so we keep it best-effort.
    rationale = ""
    m_rat = re.search(r"\"rationale\"\s*:\s*\"(.*?)\"", text, flags=re.DOTALL)
    if m_rat:
        rationale = m_rat.group(1)

    return {"selected_indices": selected_indices, "rationale": rationale}


def validate_and_sort_indices(
    selected: List[int],
    k: int,
    valid_indices: Set[int],
) -> List[int]:
    uniq = []
    seen: Set[int] = set()
    for x in selected:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    bad = [x for x in uniq if x not in valid_indices]
    if bad:
        raise ValueError(f"Invalid indices not in candidate set: {bad}")
    uniq.sort()
    # if len(uniq) != k:
    #     raise ValueError(f"Expected {k} distinct valid indices, got {len(uniq)}: {uniq}")
    return uniq


def load_trajectory_files(
    trajectory_dir: Path,
    query_ids: Optional[Set[str]],
    max_files: Optional[int],
) -> List[Path]:
    paths = sorted(trajectory_dir.glob("*.json"))
    if query_ids is not None:
        filtered: List[Path] = []
        for p in paths:
            try:
                with p.open(encoding="utf-8") as f:
                    obj = json.load(f)
                if str(obj.get("query_id", "")) in query_ids:
                    filtered.append(p)
            except Exception:
                continue
        paths = filtered
    if max_files is not None:
        paths = paths[: max_files]
    return paths


def load_query_ids_from_output_jsonl(
    output_path: Path,
    *,
    successful_only: bool,
) -> Set[str]:
    """Read JSONL written by this script; collect query_id values.

    If successful_only is True, only include lines where the object has no \"error\" key
    (Gemini succeeded and validation passed). Failed runs can be retried on the next invocation.

    If successful_only is False, include every query_id from every valid JSON line (skip any
    trajectory we have already attempted).
    """
    found: Set[str] = set()
    if not output_path.exists():
        return found
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("query_id")
            if qid is None:
                continue
            if successful_only and "error" in obj:
                continue
            found.add(str(qid))
    return found


def read_query_id_from_trajectory(path: Path) -> Optional[str]:
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        qid = obj.get("query_id")
        return str(qid) if qid is not None else None
    except Exception:
        return None


def filter_paths_by_query_ids(
    paths: List[Path],
    skip_ids: Set[str],
) -> Tuple[List[Path], int]:
    """Drop paths whose trajectory query_id is in skip_ids. Returns (kept, n_skipped)."""
    if not skip_ids:
        return paths, 0
    kept: List[Path] = []
    skipped = 0
    for p in paths:
        qid = read_query_id_from_trajectory(p)
        if qid is not None and qid in skip_ids:
            skipped += 1
            continue
        kept.append(p)
    return kept, skipped


def load_query_id_to_text(tsv_path: Path) -> dict[str, str]:
    """Load query_id -> question text from a TSV with one tab separating id from the rest of the line."""
    out: dict[str, str] = {}
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            tab = line.find("\t")
            if tab == -1:
                continue
            qid = line[:tab].strip()
            text = line[tab + 1 :]
            if qid:
                out[qid] = text
    return out


def question_from_bcp_map(query_by_id: dict[str, str], qid: str) -> str:
    """Resolve question text; try exact id, then id without leading zeros."""
    qid = (qid or "").strip()
    if not qid or not query_by_id:
        return ""
    if qid in query_by_id:
        return query_by_id[qid]
    stripped = qid.lstrip("0")
    if stripped and stripped in query_by_id:
        return query_by_id[stripped]
    if stripped == "" and "0" in query_by_id:
        return query_by_id["0"]
    return ""


def parse_tool_names_arg(values: Optional[List[str]]) -> Set[str]:
    if not values:
        return set(DEFAULT_TOOL_NAMES)
    names: Set[str] = set()
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                names.add(part)
    return names


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
    # python select_useful_tool_calls.py --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4/ --output selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl --use-original-messages --num-threads 8
    ap = argparse.ArgumentParser(description="Select k useful tool calls via Gemini; verbatim excerpts in output.")
    ap.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Directory containing trajectory *.json files",
    )
    ap.add_argument("--k", type=int, default=5, help="Number of tool calls to select (default 5)")
    ap.add_argument(
        "--tool-names",
        action="append",
        default=None,
        help=f"Comma-separated tool_name filter (repeatable). Default: {','.join(DEFAULT_TOOL_NAMES)}",
    )
    ap.add_argument("--max-files", type=int, default=None, help="Process at most N files (after sorting)")
    ap.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Comma-separated query_id whitelist",
    )
    ap.add_argument("--output", type=Path, default=None, help="Append JSONL results here")
    ap.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Do not skip trajectories: re-run even when --output already has a successful row for that query_id",
    )
    ap.add_argument(
        "--skip-seen",
        action="store_true",
        help="Skip any query_id that appears in --output on any line (success or failure). Requires --output.",
    )
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="Do not call Gemini; report prompt sizes")
    ap.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "@vertexai-gemini-ec5413/gemini-2.5-pro"),
    )
    ap.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"),
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--preview-chars", type=int, default=1200, help="Per-tool output preview length in catalog")
    ap.add_argument("--context-max-chars", type=int, default=120_000, help="Cap full trajectory context in prompt")
    ap.add_argument("--context-reasoning-max-chars", type=int, default=0, help="Per reasoning block cap (0=none)")
    ap.add_argument("--context-tool-max-chars", type=int, default=500, help="Per tool result cap in context block")
    ap.add_argument(
        "--use-original-messages",
        action="store_true",
        help="Use original_messages from trajectory instead of the reformatted trajectory context",
    )
    ap.add_argument(
        "--queries-tsv",
        type=Path,
        default=DEFAULT_BCP_QUERIES_TSV,
        help=f"BCP topics TSV (query_id\\tquestion). Default: {DEFAULT_BCP_QUERIES_TSV}",
    )
    ap.add_argument(
        "--no-queries-tsv",
        action="store_true",
        help="Do not load --queries-tsv; use only query/question fields from each trajectory JSON",
    )

    args = ap.parse_args()
    traj_dir: Path = args.trajectory_dir
    if not traj_dir.is_dir():
        print(f"Not a directory: {traj_dir}", file=sys.stderr)
        sys.exit(1)

    qid_filter: Optional[Set[str]] = None
    if args.query_ids:
        qid_filter = {x.strip() for x in args.query_ids.split(",") if x.strip()}

    paths = load_trajectory_files(traj_dir, qid_filter, args.max_files)
    if not paths:
        print("No trajectory files to process.", file=sys.stderr)
        sys.exit(1)

    skip_ids: Set[str] = set()
    if args.skip_seen:
        if args.output is None:
            print("Warning: --skip-seen requires --output; not skipping any items.", file=sys.stderr)
        else:
            skip_ids = load_query_ids_from_output_jsonl(args.output, successful_only=False)
    elif args.output is not None and not args.no_skip_completed:
        skip_ids = load_query_ids_from_output_jsonl(args.output, successful_only=True)

    if skip_ids:
        before = len(paths)
        paths, n_skipped = filter_paths_by_query_ids(paths, skip_ids)
        kind = "seen" if args.skip_seen else "completed"
        print(
            f"Skipping {n_skipped} already-{kind} trajectory(ies) (from {args.output}). "
            f"Remaining: {len(paths)} (of {before}).",
            file=sys.stderr,
        )
    if not paths:
        print("No trajectory files left to process after skip filter.", file=sys.stderr)
        sys.exit(0)

    allowed = parse_tool_names_arg(args.tool_names)

    query_by_id: Optional[dict[str, str]] = None
    if not args.no_queries_tsv:
        qp = args.queries_tsv
        if qp.is_file():
            query_by_id = load_query_id_to_text(qp)
        else:
            print(
                f"Warning: queries TSV not found ({qp}); using trajectory query/question fields only.",
                file=sys.stderr,
            )

    model: Any = None
    gen_params: Any = None
    if not args.dry_run:
        if Gemini25Pro is None or GenParams is None:
            print(
                "portkey (portkey-ai) is required for live Gemini calls. "
                "Install with: pip install portkey-ai",
                file=sys.stderr,
            )
            sys.exit(1)
        gen_params = GenParams(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        model = Gemini25Pro(model=args.model, base_url=args.base_url or None)

    write_lock = threading.Lock()

    def append_jsonl(obj: dict) -> None:
        if args.output is None:
            return
        with write_lock:
            with args.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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

    if args.dry_run:
        for p in paths:
            r = job(p)
            print(json.dumps(r, ensure_ascii=False))
        return

    if args.num_threads <= 1:
        for p in tqdm(paths, desc="trajectories"):
            r = job(p)
            print(json.dumps(r, ensure_ascii=False))
            append_jsonl(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as ex:
            futs = [ex.submit(job, p) for p in paths]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="trajectories"):
                r = fut.result()
                print(json.dumps(r, ensure_ascii=False))
                append_jsonl(r)


if __name__ == "__main__":
    main()
