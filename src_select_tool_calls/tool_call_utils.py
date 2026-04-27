#!/usr/bin/env python3
"""Shared utilities for tool-call selection scripts.

All functions here are model-agnostic: file I/O, Gemini response parsing,
result-based trajectory formatting, and the generic OM run_one driver.
Model-specific OM helpers live in each script.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):  # type: ignore[misc]
        return it

_DIR = Path(__file__).resolve().parent         # src_select_tool_calls/
_PROJECT_ROOT = _DIR.parent                    # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_BCP_QUERIES_TSV = _PROJECT_ROOT / "topics-qrels" / "bcp" / "queries.tsv"

try:
    from portkey import Gemini25Pro, GenParams
except ImportError:
    Gemini25Pro = None  # type: ignore[misc, assignment]
    GenParams = None    # type: ignore[misc, assignment]

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


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

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
        paths = paths[:max_files]
    return paths


def load_query_ids_from_output_jsonl(
    output_path: Path,
    *,
    successful_only: bool,
) -> Set[str]:
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


# ---------------------------------------------------------------------------
# Query data
# ---------------------------------------------------------------------------

def load_query_id_to_text(tsv_path: Path) -> dict[str, str]:
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
            text = line[tab + 1:]
            if qid:
                out[qid] = text
    return out


def question_from_bcp_map(query_by_id: dict[str, str], qid: str) -> str:
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


# ---------------------------------------------------------------------------
# Gemini response parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    else:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    m_sel = re.search(r'"selected_indices"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if not m_sel:
        raise json.JSONDecodeError("Could not parse or recover selected_indices", text, 0)
    nums = re.findall(r"\b\d+\b", m_sel.group(1))
    if not nums:
        raise json.JSONDecodeError("Recovered selected_indices was empty", text, 0)
    selected_indices = [int(x) for x in nums]
    rationale = ""
    m_rat = re.search(r'"rationale"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
    if m_rat:
        rationale = m_rat.group(1)
    return {"selected_indices": selected_indices, "rationale": rationale}


def validate_and_sort_indices(
    selected: List[int],
    k: int,
    valid_indices: Set[int],
) -> List[int]:
    uniq: List[int] = []
    seen: Set[int] = set()
    for x in selected:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    bad = [x for x in uniq if x not in valid_indices]
    if bad:
        raise ValueError(f"Invalid indices not in candidate set: {bad}")
    uniq.sort()
    return uniq


# ---------------------------------------------------------------------------
# Result-based trajectory helpers
# ---------------------------------------------------------------------------

def _reasoning_text(step: dict) -> str:
    output = step.get("output", [])
    if isinstance(output, list):
        return " ".join(str(o) for o in output)
    return str(output or "")


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


def previous_tool_index(result: List[dict], tool_idx: int) -> int:
    for j in range(tool_idx - 1, -1, -1):
        if result[j].get("type") == "tool_call":
            return j
    return -1


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


def verbatim_excerpt_for_tool(
    trajectory: dict,
    tool_idx: int,
) -> str:
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
    args_display = args_raw if isinstance(args_raw, str) else json.dumps(args_raw, ensure_ascii=False)
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


def format_trajectory_for_prompt(
    trajectory: dict,
    *,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
    query_max_chars: int = 1000,
) -> str:
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


# ---------------------------------------------------------------------------
# Generic OM run_one driver
# ---------------------------------------------------------------------------

def run_one_om(
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
    query_by_id: Optional[dict[str, str]],
    find_candidates_fn: Callable[[dict, Set[str]], List[int]],
    build_catalog_fn: Callable[[dict, List[int], int], List[str]],
    build_excerpt_fn: Callable[[dict, Sequence[int]], str],
    format_context_fn: Optional[Callable[..., str]] = None,
    require_original_messages: bool = True,
) -> dict:
    """Load one trajectory, select k tool calls via Gemini, return result dict.

    find_candidates_fn, build_catalog_fn, build_excerpt_fn are model-specific parsers.
    format_context_fn overrides the context block formatter (defaults to format_trajectory_for_prompt).
    require_original_messages=False skips the missing-OM guard (for result-based mode).
    """
    with path.open(encoding="utf-8") as f:
        traj = json.load(f)

    if require_original_messages and not traj.get("original_messages"):
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

    candidates = find_candidates_fn(traj, allowed_tool_names)
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
    if len(candidates) <= k:
        excerpt = build_excerpt_fn(traj, candidates)
        # print('&'*100)
        # print(excerpt)
        # print('&'*100)
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "selected_indices": candidates,
            "rationale": "all_candidates_selected",
            "excerpt": excerpt,
            "candidates": candidates,
            "correct_num_selected": True,
            "k_requested": k,
            "k_effective": k_eff,
        }
    if k_eff < k:
        print(
            f"[warn] {path.name}: only {len(candidates)} candidate tool calls; using k={k_eff}",
            file=sys.stderr,
        )

    catalog_lines = build_catalog_fn(traj, candidates, preview_chars)
    qid_str = str(traj.get("query_id", "")).strip()
    question = ""
    if query_by_id:
        question = question_from_bcp_map(query_by_id, qid_str)
    if not question:
        question = traj.get("query") or traj.get("question") or ""

    _fmt_ctx = format_context_fn if format_context_fn is not None else format_trajectory_for_prompt
    context_block = _fmt_ctx(
        traj,
        max_chars=context_max_chars,
        reasoning_max_chars=context_reasoning_max,
        tool_output_max_chars=context_tool_max,
    )

    allowed_json = json.dumps(candidates)
    user_parts = [
        f"User question:\n{question}\n",
        f"K = {k_eff} (you must return exactly {k_eff} indices).",
        "Allowed candidate indices (choose only from this list; each value is a message index in original_messages): "
        + allowed_json,
        "Candidate tool calls (choose by index=...):",
        *catalog_lines,
        "\nTrajectory context (may be truncated for length; use indices from the catalog only):",
        context_block,
        f"\nReminder: selected_indices must be a subset of this list only: {allowed_json}",
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
        ordered = validate_and_sort_indices(sel_ints, k_eff, set(candidates))
    except Exception as e:
        return {
            "query_id": str(traj.get("query_id", "")),
            "source_file": path.name,
            "error": str(e),
            "raw_response": raw[:2000],
            "k_effective": k_eff,
            "candidates": candidates,
        }

    excerpt = build_excerpt_fn(traj, ordered)
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


# ---------------------------------------------------------------------------
# Pipeline runner (shared main loop)
# ---------------------------------------------------------------------------

def run_pipeline(
    paths: List[Path],
    num_threads: int,
    job_fn: Callable[[Path], dict],
    output_path: Optional[Path],
    dry_run: bool = False,
) -> None:
    write_lock = threading.Lock()

    def append_jsonl(obj: dict) -> None:
        if output_path is None:
            return
        with write_lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if dry_run:
        for p in paths:
            r = job_fn(p)
            print(json.dumps(r, ensure_ascii=False))
        return

    if num_threads <= 1:
        for p in tqdm(paths, desc="trajectories"):
            r = job_fn(p)
            print(json.dumps(r, ensure_ascii=False))
            append_jsonl(r)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as ex:
            futs = [ex.submit(job_fn, p) for p in paths]
            for fut in tqdm(
                concurrent.futures.as_completed(futs), total=len(futs), desc="trajectories"
            ):
                r = fut.result()
                print(json.dumps(r, ensure_ascii=False))
                append_jsonl(r)


# ---------------------------------------------------------------------------
# Shared argparse setup
# ---------------------------------------------------------------------------

def add_common_args(ap: "argparse.ArgumentParser") -> None:  # type: ignore[name-defined]
    """Add args shared by all selection scripts."""
    import argparse as _ap
    ap.add_argument("--trajectory-dir", type=Path, required=True,
                    help="Directory containing trajectory *.json files")
    ap.add_argument("--k", type=int, default=5,
                    help="Number of tool calls to select (default 5)")
    ap.add_argument("--tool-names", action="append", default=None,
                    help=f"Comma-separated tool_name filter (repeatable). "
                         f"Default: {','.join(DEFAULT_TOOL_NAMES)}")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Process at most N files (after sorting)")
    ap.add_argument("--query-ids", type=str, default=None,
                    help="Comma-separated query_id whitelist")
    ap.add_argument("--output", type=Path, default=None,
                    help="Append JSONL results here")
    ap.add_argument("--no-skip-completed", action="store_true",
                    help="Re-run even when --output already has a successful row for that query_id")
    ap.add_argument("--skip-seen", action="store_true",
                    help="Skip any query_id in --output (success or failure). Requires --output.")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not call Gemini; report prompt sizes")
    ap.add_argument("--model", type=str,
                    default="@vertexai-gemini-ec5413/gemini-2.5-pro",
                    help="Gemini model ID (also reads GEMINI_MODEL env var)")
    ap.add_argument("--base-url", type=str,
                    default="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
                    help="Portkey base URL (also reads PORTKEY_BASE_URL env var)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--preview-chars", type=int, default=1200,
                    help="Per-tool output preview length in catalog")
    ap.add_argument("--context-max-chars", type=int, default=280_000,
                    help="Cap full trajectory context in prompt")
    ap.add_argument("--context-reasoning-max-chars", type=int, default=2000,
                    help="Per reasoning block cap (0=none)")
    ap.add_argument("--context-tool-max-chars", type=int, default=3000,
                    help="Per tool result cap in context block")
    ap.add_argument("--queries-tsv", type=Path, default=DEFAULT_BCP_QUERIES_TSV,
                    help=f"BCP topics TSV (query_id\\tquestion). Default: {DEFAULT_BCP_QUERIES_TSV}")
    ap.add_argument("--no-queries-tsv", action="store_true",
                    help="Do not load --queries-tsv; use only query/question fields from each JSON")


def resolve_common_args(args: Any) -> tuple:
    """Resolve paths/skip-sets from parsed common args. Returns (paths, allowed, query_by_id, model, gen_params)."""
    import os
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
            print("Warning: --skip-seen requires --output; not skipping.", file=sys.stderr)
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
        print("No trajectory files left after skip filter.", file=sys.stderr)
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

    model_obj: Any = None
    gen_params_obj: Any = None
    if not args.dry_run:
        if Gemini25Pro is None or GenParams is None:
            print(
                "portkey (portkey-ai) is required for live Gemini calls. "
                "Install with: pip install portkey-ai",
                file=sys.stderr,
            )
            sys.exit(1)
        model_name = os.getenv("GEMINI_MODEL", args.model)
        base_url = os.getenv("PORTKEY_BASE_URL", args.base_url) or None
        gen_params_obj = GenParams(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        model_obj = Gemini25Pro(model=model_name, base_url=base_url)

    return paths, allowed, query_by_id, model_obj, gen_params_obj
