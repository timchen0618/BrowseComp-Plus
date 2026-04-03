#!/usr/bin/env python3
"""
Generate retrospective plans from agent trajectories.

For each trajectory, extracts the sequence of search queries (and brief
reasoning notes) that the agent actually issued, then calls an LLM to
produce a structured plan in the planning_prompt_v6.1.md format — as if
the plan had been written BEFORE execution.

Primary use-case: understand whether agents already plan implicitly and
whether their implicit strategies align with what a planner would produce.

Usage (vLLM):
    python generate_retrospective_plans.py \
        --input-dirs runs/tongyi/tongyi_seed0 runs/gpt-oss-120b/gpt-oss-120b_seed0 \
        --output-file retrospective_plans/tongyi_gpt_seed0.jsonl \
        --model-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3.5-122B-A10B

Usage (Anthropic API, requires ANTHROPIC_API_KEY):
    python generate_retrospective_plans.py \
        --input-dirs runs/tongyi/tongyi_seed0 \
        --output-file retrospective_plans/tongyi_seed0.jsonl \
        --backend anthropic \
        --model claude-sonnet-4-6

Usage (Portkey/Gemini, requires PORTKEY_API_KEY):
    python generate_retrospective_plans.py \
        --input-dirs runs/tongyi/tongyi_seed0 runs/gpt-oss-120b/gpt-oss-120b_seed0 \
        --output-file retrospective_plans/tongyi_gpt_seed0_gemini.jsonl \
        --backend portkey \
        --model "@vertexai-gemini-ec5413/gemini-2.5-pro" \
        --num-threads 4
"""

import argparse
import csv
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import openai
from tqdm import tqdm

T = TypeVar("T")

# ---------------------------------------------------------------------------
# System prompt for retrospective plan generation
# ---------------------------------------------------------------------------

_PLAN_FORMAT_EXAMPLE = """\
<plan>
Goal: [Restate the question concisely]
Expected answer type: [entity | date | number | list | yes-no | description]
Answer sketch: [One-sentence hypothesis about the answer structure]
Budget: [N] | Reserve: [2-3]

Steps:
1. [Major subgoal]
   1.1. [Action] | Type: explore (broad)
        Suggested query: ["exact query string used"]
        Depends on: none
        → On failure: [fallback]
   1.2. [Action] | Type: explore (reformulate)
        Suggested query: ["another query string"]
        Depends on: none
        → On failure: [fallback]
2. [Major subgoal — e.g., verify candidate]
   2.1. [Action] | Type: verify
        Suggested query: ["targeted query"]
        Depends on: step 1 — uses candidate name found in step 1
        → On failure: [fallback]
3. Synthesize findings and produce final answer | Type: reason
   Depends on: steps 1 and 2

Early termination: [condition under which to stop early]
</plan>"""

RETROSPECTIVE_SYSTEM_PROMPT = f"""\
You are an expert at reconstructing search strategies from agent trajectories.

Given a question and the ordered list of search queries (with brief reasoning
notes) that an agent actually issued while solving the question, produce a
structured RETROSPECTIVE PLAN that describes what the agent did — formatted
as if it were a plan written BEFORE execution.

### Step Types
- **explore**: Search to surface candidates or resolve unknowns.
  Subtypes: broad (initial wide search), disambiguate, reformulate (retry
  with different angle), reverse (find via a related entity).
- **verify**: Search to confirm or reject a specific candidate.
- **reason**: Derive or synthesize from gathered information (no search call).

### Rules
1. Reflect the agent's ACTUAL search trajectory — do not invent searches
   that were not made.
2. Group consecutive searches with the same logical intent into a single
   substep rather than making one substep per query.
3. Assign a Type to each substep based on what the agent was trying to do.
4. Use the actual query strings from the trajectory as "Suggested query"
   values (copy them verbatim).
5. Infer dependencies between steps from the reasoning notes.
6. Include a final reason step for synthesizing the answer.
7. Keep the plan concise — merge closely related searches.
8. Output ONLY the plan inside <plan></plan> tags, nothing else.

### Plan Format
{_PLAN_FORMAT_EXAMPLE}
"""

LEGACY_RETROSPECTIVE_SYSTEM_PROMPT = """\
You are an expert at reconstructing search strategies from agent trajectories.

Given a question and the ordered list of search queries (with brief reasoning
notes) that an agent actually issued while solving the question, produce a
RETROSPECTIVE PLAN in the legacy checklist format — as if it were written
BEFORE execution, but reflecting only what the agent actually did.

### Status markers
Since this is a retrospective plan, mark all executed steps as [x] done.
Use [!] for searches that clearly failed (returned no useful results) and
[~] for partially successful ones, if apparent from the reasoning notes.

### Rules
1. Reflect the agent's ACTUAL search trajectory — do not invent steps.
2. Group searches with the same logical intent under a single numbered step.
3. Use sub-branches (1.1, 1.2, ...) for alternative approaches tried in
   sequence (e.g., first tried English query, then Arabic).
4. Log the search count per step as (Query=N) at the end of each leaf.
5. End with a synthesis step for producing the final answer.
6. Output ONLY the plan inside <plan></plan> tags, nothing else.

### Plan Format
<plan>
1. [x] Find initial candidates using the most distinctive constraint
   1.1 [x] Broad search for candidates (Query=1)
   1.2 [x] Reformulate after weak results (Query=1)
   1.3 [x] Candidate A: [Name] — verify immediately after surfacing
      1.3.1 [x] Verify constraint X for Candidate A (Query=1)
      1.3.2 [x] Verify constraint Y for Candidate A (Query=1)
      1.3.3 [!] Verify constraint Z — failed, candidate rejected (Query=1)
   1.4 [x] Candidate B: [Name] — pivot to next candidate after A rejected
      1.4.1 [x] Verify constraint X for Candidate B (Query=1)
      1.4.2 [x] Verify constraint Y for Candidate B (Query=1)
      1.4.3 [x] Verify constraint Z for Candidate B — confirmed (Query=1)
2. [x] Verify remaining constraints on the confirmed candidate
   2.1 [x] Confirm additional detail (Query=1)
   2.2 [~] Partial evidence found, sufficient for conclusion (Query=1)
3. [x] Synthesize findings and produce final answer
</plan>
"""


# ---------------------------------------------------------------------------
# Retry / timeout helpers (same pattern as vllm_planner.py)
# ---------------------------------------------------------------------------

def log_err(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


def _call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as e:
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"request exceeded {timeout_s:.1f}s") from e
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _retry_with_deadline(fn: Callable[[], T], max_retries: int = 3,
                         per_attempt_timeout_s: float = 180.0,
                         total_deadline_s: float = 720.0) -> T:
    start = time.time()
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        if time.time() - start > total_deadline_s:
            raise TimeoutError(f"total deadline exceeded ({total_deadline_s:.1f}s)") from last_err
        try:
            return _call_with_timeout(fn, per_attempt_timeout_s)
        except Exception as e:
            log_err(f"Attempt {attempt + 1} failed: {repr(e)}")
            last_err = e
            if attempt >= max_retries:
                raise
            backoff = min(20.0, 1.0 * (2 ** attempt)) * (0.5 + random.random())
            time.sleep(backoff)
    raise RuntimeError("retry loop fell through") from last_err


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def generate_openai(client: openai.OpenAI, model: str, messages: List[Dict],
                    max_tokens: int = 4096) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.95,
    )
    return resp.choices[0].message.content or ""


def generate_anthropic(model: str, messages: List[Dict],
                       system: str, max_tokens: int = 4096) -> str:
    # Import lazily so the script works without anthropic installed
    import anthropic  # noqa: PLC0415
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def generate_portkey(model: str, messages: List[Dict],
                     base_url: Optional[str], max_tokens: int = 4096) -> str:
    # Import lazily so the script works without portkey-ai installed
    from portkey_ai import Portkey  # noqa: PLC0415
    api_key = os.environ.get("PORTKEY_API_KEY") or os.environ.get("PORTKEY_API_TOKEN", "")
    if not api_key:
        raise RuntimeError("PORTKEY_API_KEY is not set")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = Portkey(**client_kwargs)
    # Use streaming to avoid intermediate gateway 504 timeouts (same as portkey.py)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.95,
        stream=True,
    )
    return "".join(
        chunk.choices[0].delta.content
        for chunk in stream
        if chunk.choices and chunk.choices[0].delta.content
    )


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def parse_tool_args(args_str: str) -> dict:
    try:
        obj = json.loads(args_str) if args_str else {}
        # Handle double-encoded JSON (Tongyi model wraps arguments in extra quotes)
        if isinstance(obj, str):
            obj = json.loads(obj)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_search_steps(trace: dict) -> List[Dict[str, str]]:
    """
    Walk the result array and collect each search call with the reasoning
    that preceded it.

    Returns: list of {"query": str, "reasoning_before": str}
    """
    steps: List[Dict[str, str]] = []
    pending: List[str] = []

    for entry in trace.get("result", []):
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")

        if etype in ("reasoning", "assistant"):
            text = entry.get("output", "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = (text or "").strip()
            if text:
                pending.append(text[:250])  # keep first 250 chars

        elif etype == "tool_call":
            tool = entry.get("tool_name", "")
            if tool not in ("search", "local_knowledge_base_retrieval"):
                pending.clear()
                continue

            args = parse_tool_args(entry.get("arguments", "{}"))
            # Different models use different field names
            query = args.get("query", args.get("user_query", ""))
            if isinstance(query, list):
                query = query[0] if query else ""
            query = (query or "").strip()

            if query:
                steps.append({
                    "query": query,
                    # Use last two reasoning snippets for context
                    "reasoning_before": " | ".join(pending[-2:]),
                })
            pending.clear()

    return steps


def build_user_message(query_text: str, steps: List[Dict[str, str]]) -> str:
    """Build the user message describing the trajectory."""
    lines = [f"Question: {query_text}", "", "Agent's actual search trajectory:"]
    for i, step in enumerate(steps, 1):
        lines.append(f"\nSearch {i}: \"{step['query']}\"")
        if step["reasoning_before"]:
            snippet = step["reasoning_before"][:200]
            lines.append(f"  [Agent's reasoning before this search: {snippet}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-trace processing
# ---------------------------------------------------------------------------

def process_trace(
    json_path: Path,
    query_map: Dict[str, str],
    generate_fn: Callable[[List[Dict]], str],
    total_deadline_s: float = 720.0,
) -> Optional[Dict[str, Any]]:
    try:
        trace = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log_err(f"skipping {json_path.name}: {e}")
        return None

    query_id = str(trace.get("query_id", ""))
    query_text = query_map.get(query_id, "")
    if not query_text:
        log_err(f"no query text for query_id={query_id} in {json_path.name}")

    steps = extract_search_steps(trace)
    if not steps:
        log_err(f"no search queries in {json_path.name} — skipping")
        return None

    user_msg = build_user_message(query_text, steps)
    messages = [{"role": "user", "content": user_msg}]

    try:
        plan_text = _retry_with_deadline(lambda: generate_fn(messages),
                                         total_deadline_s=total_deadline_s)
    except Exception as e:
        log_err(f"LLM call failed for {json_path.name}: {e}")
        return None

    if "<plan>" not in plan_text:
        plan_text = f"<plan>\n{plan_text}\n</plan>"

    return {
        "query_id": query_id,
        "query_text": query_text,
        "source_file": json_path.name,
        "num_searches": len(steps),
        "output": plan_text,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_query_map(tsv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def load_completed_ids(output_file: Path) -> set:
    ids: set = set()
    if not output_file.exists():
        return ids
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                qid = obj.get("query_id")
                src = obj.get("source_file", "")
                if qid is not None:
                    # Key on (query_id, source_file) to allow multiple runs
                    ids.add((str(qid), src))
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate retrospective plans from agent trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs", nargs="+", required=True, type=Path,
        help="Directories containing run_*.json trajectory files",
    )
    parser.add_argument(
        "--output-file", required=True, type=Path,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--query-file", default=Path("topics-qrels/queries.tsv"), type=Path,
        help="TSV file mapping query_id to query_text",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-122B-A10B",
        help="LLM model name",
    )
    parser.add_argument(
        "--model-url", default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL (vLLM)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="vLLM port (sets --model-url to http://127.0.0.1:{port}/v1)",
    )
    parser.add_argument(
        "--backend", choices=["openai", "anthropic", "portkey"], default="openai",
        help="'openai' uses an OpenAI-compatible API (vLLM or real OpenAI); "
             "'anthropic' uses the Anthropic SDK (requires ANTHROPIC_API_KEY); "
             "'portkey' uses Portkey gateway for Gemini (requires PORTKEY_API_KEY)",
    )
    parser.add_argument(
        "--portkey-base-url",
        default="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
        help="Portkey gateway base URL (only used with --backend portkey)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max tokens to generate per plan",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Parallel LLM calls",
    )
    parser.add_argument(
        "--max-traces", type=int, default=None,
        help="Max traces per directory (for testing)",
    )
    parser.add_argument(
        "--plan-format", choices=["v6", "legacy"], default="v6",
        help="'v6' uses the structured planning_prompt_v6.1 format (Goal/Steps/Type/Depends); "
             "'legacy' uses the PROMPT_PLANNER checklist format ([ ]/[x] status markers)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip traces already present in output file",
    )
    args = parser.parse_args()

    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    # Retry deadline: Portkey/Gemini needs a longer total timeout for gateway calls
    total_deadline = 1800.0 if args.backend == "portkey" else 720.0

    # Select system prompt based on plan format
    system_prompt = (
        LEGACY_RETROSPECTIVE_SYSTEM_PROMPT
        if args.plan_format == "legacy"
        else RETROSPECTIVE_SYSTEM_PROMPT
    )

    # Build generate function
    if args.backend == "anthropic":
        def generate_fn(messages: List[Dict]) -> str:
            return generate_anthropic(args.model, messages,
                                      system_prompt, args.max_tokens)
    elif args.backend == "portkey":
        default_model = "@vertexai-gemini-ec5413/gemini-2.5-pro"
        portkey_model = args.model if args.model != "Qwen/Qwen3.5-122B-A10B" else default_model
        def generate_fn(messages: List[Dict]) -> str:
            full = [{"role": "system", "content": system_prompt}] + messages
            return generate_portkey(portkey_model, full, args.portkey_base_url, args.max_tokens)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = openai.OpenAI(base_url=args.model_url, api_key=api_key,
                               timeout=600.0)
        # Prepend system message
        def generate_fn(messages: List[Dict]) -> str:
            full = [{"role": "system", "content": system_prompt}] + messages
            return generate_openai(client, args.model, full, args.max_tokens)

    # Collect trajectory files
    query_map = load_query_map(args.query_file)
    all_files: List[Path] = []
    for d in args.input_dirs:
        if not d.is_dir():
            print(f"Error: {d} is not a directory", file=sys.stderr)
            sys.exit(1)
        files = sorted(d.glob("*.json"))
        if args.max_traces:
            files = files[: args.max_traces]
        all_files.extend(files)

    if not all_files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Resume support
    completed = load_completed_ids(args.output_file) if args.resume else set()
    remaining = []
    for p in all_files:
        if not args.resume:
            remaining.append(p)
        else:
            try:
                qid = str(json.loads(p.read_text(encoding="utf-8")).get("query_id", ""))
            except Exception:
                qid = ""
            if (qid, p.name) not in completed:
                remaining.append(p)

    print(f"Total files: {len(all_files)} | Already done: {len(all_files) - len(remaining)} | "
          f"Remaining: {len(remaining)}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    success = 0
    failed = 0

    def process_and_write(json_path: Path) -> bool:
        nonlocal success, failed
        result = process_trace(json_path, query_map, generate_fn, total_deadline)
        if result is not None:
            with write_lock:
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            success += 1
            return True
        failed += 1
        return False

    if args.num_threads <= 1:
        for p in tqdm(remaining, desc="Generating retrospective plans"):
            process_and_write(p)
    else:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as ex,
            tqdm(total=len(remaining), desc="Generating retrospective plans") as pbar,
        ):
            futures = {ex.submit(process_and_write, p): p for p in remaining}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    log_err(f"unexpected error: {e}")
                pbar.update(1)

    print(f"\nDone. Plans generated: {success} | Failed/skipped: {failed}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()


# Example invocations:
#
# vLLM (H200 node with Qwen3.5-122B already running on port 8000):
#   python generate_retrospective_plans.py \
#       --input-dirs runs/tongyi/tongyi_seed0 runs/gpt-oss-120b/gpt-oss-120b_seed0 \
#       --output-file retrospective_plans/tongyi_gpt_seed0.jsonl \
#       --model-url http://127.0.0.1:8000/v1 \
#       --model Qwen/Qwen3.5-122B-A10B \
#       --num-threads 6
#
# Portkey/Gemini:
#   PORTKEY_API_KEY=... python generate_retrospective_plans.py \
#       --input-dirs runs/tongyi/tongyi_seed0 runs/gpt-oss-120b/gpt-oss-120b_seed0 \
#       --output-file retrospective_plans/tongyi_gpt_seed0_gemini.jsonl \
#       --backend portkey \
#       --model "@vertexai-gemini-ec5413/gemini-2.5-pro" \
#       --num-threads 4
#
# Anthropic (Claude):
#   ANTHROPIC_API_KEY=... python generate_retrospective_plans.py \
#       --input-dirs runs/tongyi/tongyi_seed0 \
#       --output-file retrospective_plans/tongyi_seed0_claude.jsonl \
#       --backend anthropic \
#       --model claude-sonnet-4-6
