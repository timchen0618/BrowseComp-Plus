#!/usr/bin/env python3
"""
Evaluate whether an agent followed the plan given to it.

For each trajectory, extracts the plan (subgoals), formats the agent's
trajectory, and uses an LLM judge (via vLLM) to assess per-subgoal
completion. Outputs a score between 0 and 1.

Usage:
    # Run inference on a single directory
    python eval_plan_quality/eval_plan_following.py inference \
        --input-dir runs/Qwen3-Embedding-8B/tongyi_planning_seed0 \
        --output-file eval_plan_quality/plan_following_seed0.jsonl \
        --model Qwen/Qwen3-32B --num-threads 8

    # Run on all planning directories matching a pattern
    python eval_plan_quality/eval_plan_following.py inference \
        --input-dir runs/Qwen3-Embedding-8B/tongyi_planning_seed0 \
        --input-dir runs/Qwen3-Embedding-8B/tongyi_planning_seed1 \
        --output-file eval_plan_quality/plan_following_seeds.jsonl \
        --model Qwen/Qwen3-32B --num-threads 8

    # Compute aggregate statistics from an existing output file
    python eval_plan_quality/eval_plan_following.py compute \
        --output-file eval_plan_quality/plan_following_seed0.jsonl
"""

import argparse
import concurrent.futures
import json
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import openai
from tqdm import tqdm

T = TypeVar("T")

DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"

PLAN_PREFIX = (
    "Here is the planner's response; please follow the plan "
    "to answer the user's question: "
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing whether a search agent followed a given plan.

You will be given:
1. The full **plan** for context (showing the hierarchy of goals).
2. A list of **leaf subgoals** — the most specific, actionable items from the plan.
3. The agent's **trajectory** — a sequence of reasoning steps, tool calls (searches), and a final answer.

Your task is to determine, for EACH leaf subgoal listed, whether the agent actually completed it.

Rules:
- A subgoal is "completed" (score 1.0) if the agent clearly attempted and executed it, even if the result was unsuccessful (e.g., searched but found no results).
- A subgoal is "not completed" (score 0.0) if the agent never attempted it or skipped it entirely.
- Judge based on what the agent DID, not on whether the final answer is correct.
- If the agent accomplished the intent of a subgoal through a different approach than specified, still count it as completed.
- ONLY evaluate the leaf subgoals listed. Do NOT evaluate parent subgoals.

Output your evaluation as a JSON object with this exact schema:
```json
{
  "subgoals": [
    {
      "subgoal_id": "1.1",
      "subgoal_text": "the subgoal text",
      "completed": true,
      "evidence": "brief evidence from the trajectory",
      "score": 1.0
    }
  ],
  "reasoning": "brief summary of plan adherence"
}
```

Output ONLY the JSON object. No other text."""

JUDGE_USER_TEMPLATE = """\
## Full Plan (for context)

{plan}

## Leaf Subgoals to Evaluate

{leaf_subgoals}

## Agent Trajectory

{trajectory}

Evaluate whether the agent completed each leaf subgoal listed above."""


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

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


def _retry_with_deadline(
    fn: Callable[[], T],
    max_retries: int = 3,
    per_attempt_timeout_s: float = 180.0,
    total_deadline_s: float = 600.0,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 20.0,
) -> T:
    start = time.time()
    last_err: Optional[BaseException] = None
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        if time.time() - start > total_deadline_s:
            raise TimeoutError(
                f"total deadline exceeded ({total_deadline_s:.1f}s)"
            ) from last_err
        try:
            return _call_with_timeout(fn, per_attempt_timeout_s)
        except Exception as e:
            log(f"Attempt {attempt + 1}/{attempts} failed: {e!r}")
            last_err = e
            if attempt >= attempts - 1:
                raise
            backoff = min(max_backoff_s, base_backoff_s * (2 ** attempt))
            backoff *= 0.5 + random.random()
            time.sleep(backoff)

    raise RuntimeError("retry loop fell through") from last_err


# ---------------------------------------------------------------------------
# Plan extraction and parsing
# ---------------------------------------------------------------------------

def extract_plan(trace: dict) -> Optional[str]:
    """Return the stripped plan text from a trace, or None if no plan exists."""
    for entry in trace.get("result", []):
        if entry.get("type") != "user":
            continue
        output = entry.get("output", "")
        if isinstance(output, str) and output.startswith(PLAN_PREFIX):
            return output[len(PLAN_PREFIX):]
    return None


def parse_subgoals(plan_text: str) -> List[Dict[str, str]]:
    """Parse a plan into individual subgoals.

    Handles formats like:
        1. Do something
        1.1. Sub-item
        - [ ] Checklist item
        1. [ ] Numbered checklist
    """
    subgoals = []
    lines = plan_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match numbered items: "1.", "1.1.", "1.1", "2.2."
        numbered = re.match(
            r"^(\d+(?:\.\d+)*)\s*\.?\s*(?:\[[ x]\]\s*)?(.*)", line
        )
        # Match bullet/checklist items: "- [ ] ...", "- ..."
        bullet = re.match(r"^[-*]\s*(?:\[[ x]\]\s*)?(.*)", line)

        if numbered and numbered.group(2).strip():
            subgoals.append({
                "subgoal_id": numbered.group(1),
                "subgoal_text": numbered.group(2).strip(),
            })
        elif bullet and bullet.group(1).strip():
            subgoals.append({
                "subgoal_id": str(len(subgoals) + 1),
                "subgoal_text": bullet.group(1).strip(),
            })

    return subgoals


def find_leaf_subgoals(subgoals: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only leaf subgoals (those with no children).

    A subgoal is a leaf if no other subgoal's ID starts with its ID + '.'.
    E.g., if we have 3.1 and 3.1.1, then 3.1 is NOT a leaf but 3.1.1 is.
    """
    all_ids = {sg["subgoal_id"] for sg in subgoals}
    leaves = []
    for sg in subgoals:
        sid = sg["subgoal_id"]
        has_child = any(
            other_id != sid and other_id.startswith(sid + ".")
            for other_id in all_ids
        )
        if not has_child:
            leaves.append(sg)
    return leaves


def get_parent_id(subgoal_id: str) -> Optional[str]:
    """Return the parent ID of a subgoal, or None if it's depth 1.

    '3.1.4' -> '3.1', '3.1' -> '3', '3' -> None
    """
    parts = subgoal_id.rsplit(".", 1)
    if len(parts) == 1:
        return None
    return parts[0]


def get_depth1_id(subgoal_id: str) -> str:
    """Return the depth-1 (top-level) ancestor ID.

    '3.1.4.2' -> '3', '1.1' -> '1', '5' -> '5'
    """
    return subgoal_id.split(".")[0]


def compute_hierarchical_score(
    leaf_scores: Dict[str, float],
    all_subgoals: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compute hierarchical score by averaging leaves up to depth 1.

    1. Start with leaf scores (0.0 or 1.0 from the LLM judge).
    2. Each non-leaf node's score = average of its direct children's scores.
    3. Overall score = average of all depth-1 node scores.

    Returns dict with per-node scores, per-depth1 scores, and overall score.
    """
    all_ids = {sg["subgoal_id"] for sg in all_subgoals}

    # Initialize scores: leaves get their judge score, others start empty
    node_scores: Dict[str, float] = {}
    for sid, score in leaf_scores.items():
        node_scores[sid] = score

    # Build parent->children mapping from all subgoal IDs
    children_of: Dict[str, List[str]] = {}
    for sid in all_ids:
        pid = get_parent_id(sid)
        if pid is not None and pid in all_ids:
            children_of.setdefault(pid, []).append(sid)

    # Bottom-up: compute scores for non-leaf nodes
    # Sort by depth descending so we process deepest first
    sorted_ids = sorted(all_ids, key=lambda x: x.count("."), reverse=True)
    for sid in sorted_ids:
        if sid in node_scores:
            continue  # already has a score (leaf)
        kids = children_of.get(sid, [])
        kid_scores = [node_scores[k] for k in kids if k in node_scores]
        if kid_scores:
            node_scores[sid] = sum(kid_scores) / len(kid_scores)

    # Collect depth-1 scores
    depth1_ids = sorted({get_depth1_id(sid) for sid in all_ids})
    depth1_scores = {}
    for d1 in depth1_ids:
        if d1 in node_scores:
            depth1_scores[d1] = node_scores[d1]

    # Overall = average of depth-1 scores
    if depth1_scores:
        overall = sum(depth1_scores.values()) / len(depth1_scores)
    else:
        overall = 0.0

    return {
        "overall_score": overall,
        "depth1_scores": depth1_scores,
        "all_node_scores": node_scores,
    }


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def format_trajectory(trace: dict, max_chars: int = 12000) -> str:
    """Format a trace's result entries into a readable trajectory string.

    Truncates individual entries and the overall output to stay within
    token budget for the judge.
    """
    parts = []
    for entry in trace.get("result", []):
        entry_type = entry.get("type", "")
        output = entry.get("output", "")
        tool_name = entry.get("tool_name")
        arguments = entry.get("arguments")

        # Skip the plan injection entry itself
        if entry_type == "user" and isinstance(output, str) and output.startswith(PLAN_PREFIX):
            continue
        # Skip the "I am calling a planner" assistant message
        if entry_type == "assistant" and "calling a planner" in str(output).lower():
            continue

        if entry_type == "reasoning":
            text = output if isinstance(output, str) else str(output)
            # Truncate long reasoning
            if len(text) > 1500:
                text = text[:1500] + "... [truncated]"
            parts.append(f"[REASONING]\n{text}")

        elif entry_type == "tool_call":
            args_str = ""
            if arguments:
                if isinstance(arguments, str):
                    args_str = arguments[:300]
                else:
                    args_str = json.dumps(arguments, ensure_ascii=False)[:300]

            result_str = ""
            if isinstance(output, str):
                result_str = output[:500]
            elif isinstance(output, list):
                result_str = f"[{len(output)} results returned]"

            parts.append(
                f"[TOOL CALL: {tool_name}]\n"
                f"Arguments: {args_str}\n"
                f"Result: {result_str}"
            )

        elif entry_type == "output_text":
            text = output if isinstance(output, str) else str(output)
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"
            parts.append(f"[FINAL ANSWER]\n{text}")

        elif entry_type == "assistant":
            text = output if isinstance(output, str) else str(output)
            if text.strip():
                parts.append(f"[ASSISTANT]\n{text[:500]}")

    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n... [trajectory truncated]"
    return full


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON response."""
    raw = raw.strip()

    # Remove thinking tags if present (Qwen3 with enable_thinking)
    think_match = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
    if think_match:
        raw = think_match.group(1).strip()

    # Remove markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Find the JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # Validate required fields
    if "subgoals" not in result:
        return None

    return result


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def format_leaf_subgoals(leaves: List[Dict[str, str]]) -> str:
    """Format leaf subgoals as a numbered list for the judge prompt."""
    parts = []
    for sg in leaves:
        parts.append(f"- [{sg['subgoal_id']}] {sg['subgoal_text']}")
    return "\n".join(parts)


def evaluate_single(
    client: openai.OpenAI,
    model: str,
    plan_text: str,
    trajectory_text: str,
    leaf_subgoals: List[Dict[str, str]],
    all_subgoals: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Evaluate a single trajectory against its plan.

    The LLM judges only leaf subgoals. Hierarchical aggregation is done
    in code afterwards.
    """
    leaf_text = format_leaf_subgoals(leaf_subgoals)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                plan=plan_text,
                leaf_subgoals=leaf_text,
                trajectory=trajectory_text,
            ),
        },
    ]

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""

    raw = _retry_with_deadline(_call, max_retries=max_retries)
    parsed = parse_judge_response(raw)

    if parsed is None:
        return {
            "judge_output_raw": raw,
            "parse_error": True,
            "leaf_subgoals": [],
            "overall_score": None,
            "depth1_scores": None,
            "reasoning": None,
        }

    # Build leaf score map from judge output
    judge_leaves = parsed.get("subgoals", [])
    leaf_scores: Dict[str, float] = {}
    for sg in judge_leaves:
        sid = sg.get("subgoal_id", "")
        score = 1.0 if sg.get("completed", False) or sg.get("score", 0) > 0.5 else 0.0
        leaf_scores[sid] = score

    # Fill in any leaves the judge missed with 0.0
    for sg in leaf_subgoals:
        if sg["subgoal_id"] not in leaf_scores:
            leaf_scores[sg["subgoal_id"]] = 0.0

    # Hierarchical aggregation
    hier = compute_hierarchical_score(leaf_scores, all_subgoals)

    return {
        "judge_output_raw": raw,
        "parse_error": False,
        "leaf_subgoals": judge_leaves,
        "overall_score": hier["overall_score"],
        "depth1_scores": hier["depth1_scores"],
        "reasoning": parsed.get("reasoning"),
    }


# ---------------------------------------------------------------------------
# Load and prepare items
# ---------------------------------------------------------------------------

def load_traces(input_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load all trajectory JSON files from the given directories."""
    items = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            log(f"WARNING: {input_dir} is not a directory, skipping")
            continue

        json_files = sorted(input_dir.glob("*.json"))
        log(f"Found {len(json_files)} JSON files in {input_dir}")

        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log(f"WARNING: skipping {json_path}: {e}")
                continue

            plan_text = extract_plan(trace)
            if plan_text is None:
                continue

            subgoals = parse_subgoals(plan_text)
            if not subgoals:
                log(f"WARNING: no subgoals parsed from {json_path.name}")
                continue

            leaf_subgoals = find_leaf_subgoals(subgoals)
            trajectory_text = format_trajectory(trace)

            items.append({
                "json_path": str(json_path),
                "input_dir": str(input_dir),
                "query_id": str(trace.get("query_id", "")),
                "status": trace.get("status", ""),
                "plan_text": plan_text,
                "subgoals": subgoals,
                "leaf_subgoals": leaf_subgoals,
                "trajectory_text": trajectory_text,
                "num_subgoals": len(subgoals),
                "num_leaf_subgoals": len(leaf_subgoals),
            })

    return items


# ---------------------------------------------------------------------------
# Mode: inference
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    # Load existing results to skip already-evaluated items
    existing_keys = set()
    if args.output_file.exists() and not args.force:
        with args.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = obj.get("json_path", "")
                    if key:
                        existing_keys.add(key)
                except json.JSONDecodeError:
                    continue
        if existing_keys:
            log(f"Found {len(existing_keys)} existing results, will skip them")

    items = load_traces(args.input_dir)
    log(f"Loaded {len(items)} traces with plans")

    # Filter out already-evaluated
    items = [it for it in items if it["json_path"] not in existing_keys]
    log(f"{len(items)} traces remaining after skipping existing results")

    if not items:
        log("Nothing to evaluate")
        return

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key=args.api_key,
        timeout=600.0,
    )

    # Thread-safe file writer
    lock = threading.Lock()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output_file.open("a", encoding="utf-8")

    results = []
    errors = 0

    def _process(item: Dict[str, Any]) -> None:
        nonlocal errors
        try:
            judge_result = evaluate_single(
                client, args.model,
                item["plan_text"],
                item["trajectory_text"],
                leaf_subgoals=item["leaf_subgoals"],
                all_subgoals=item["subgoals"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )

            record = {
                "json_path": item["json_path"],
                "input_dir": item["input_dir"],
                "query_id": item["query_id"],
                "status": item["status"],
                "num_subgoals": item["num_subgoals"],
                "num_leaf_subgoals": item["num_leaf_subgoals"],
                "overall_score": judge_result["overall_score"],
                "depth1_scores": judge_result["depth1_scores"],
                "parse_error": judge_result["parse_error"],
                "leaf_subgoals": judge_result["leaf_subgoals"],
                "reasoning": judge_result["reasoning"],
            }

            with lock:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                results.append(record)

        except Exception as e:
            log(f"ERROR processing {item['json_path']}: {e!r}")
            with lock:
                errors += 1

    if args.num_threads <= 1:
        for item in tqdm(items, desc="Evaluating"):
            _process(item)
    else:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as pool,
            tqdm(total=len(items), desc="Evaluating") as pbar,
        ):
            futures = [pool.submit(_process, item) for item in items]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                pbar.update(1)

    out_f.close()

    # Print summary
    valid = [r for r in results if not r["parse_error"] and r["overall_score"] is not None]
    log(f"Done. {len(results)} evaluated, {errors} errors, {len(results) - len(valid)} parse errors")

    if valid:
        scores = [r["overall_score"] for r in valid]
        avg = sum(scores) / len(scores)
        log(f"Average plan-following score: {avg:.3f} (n={len(valid)})")

    log(f"Results written to {args.output_file}")


# ---------------------------------------------------------------------------
# Mode: compute
# ---------------------------------------------------------------------------

def run_compute(args: argparse.Namespace) -> None:
    if not args.output_file.exists():
        print(f"File not found: {args.output_file}", file=sys.stderr)
        sys.exit(1)

    records = []
    with args.output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    valid = [r for r in records if not r.get("parse_error") and r.get("overall_score") is not None]
    parse_errors = len(records) - len(valid)

    print(f"Total records: {len(records)}")
    print(f"Valid records: {len(valid)}")
    print(f"Parse errors:  {parse_errors}")
    print()

    if not valid:
        print("No valid records to analyze.")
        return

    scores = [r["overall_score"] for r in valid]
    avg = sum(scores) / len(scores)
    median = sorted(scores)[len(scores) // 2]
    min_s = min(scores)
    max_s = max(scores)

    print(f"Overall plan-following score:")
    print(f"  Mean:   {avg:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Min:    {min_s:.3f}")
    print(f"  Max:    {max_s:.3f}")
    print()

    # Per-directory breakdown
    by_dir: Dict[str, List[float]] = {}
    for r in valid:
        d = r.get("input_dir", "unknown")
        by_dir.setdefault(d, []).append(r["overall_score"])

    if len(by_dir) > 1:
        print("Per-directory breakdown:")
        for d, s in sorted(by_dir.items()):
            d_name = Path(d).name
            d_avg = sum(s) / len(s)
            print(f"  {d_name}: mean={d_avg:.3f} (n={len(s)})")
        print()

    # Per leaf subgoal completion rate (aggregate across all records)
    total_leaves = 0
    completed_leaves = 0
    for r in valid:
        for sg in r.get("leaf_subgoals", []):
            total_leaves += 1
            if sg.get("completed", False) or sg.get("score", 0) > 0.5:
                completed_leaves += 1

    if total_leaves > 0:
        print(f"Leaf subgoal completion rate: {completed_leaves}/{total_leaves} "
              f"({completed_leaves / total_leaves:.1%})")

    # Avg number of subgoals / leaves
    avg_subgoals = sum(r.get("num_subgoals", 0) for r in valid) / len(valid)
    avg_leaves = sum(r.get("num_leaf_subgoals", 0) for r in valid) / len(valid)
    print(f"Avg subgoals per plan: {avg_subgoals:.1f} (total), {avg_leaves:.1f} (leaves)")
    print()

    # Distribution histogram
    buckets = [0] * 11  # 0.0, 0.1, ..., 1.0
    for s in scores:
        idx = min(10, int(s * 10))
        buckets[idx] += 1

    print("Score distribution:")
    for i, count in enumerate(buckets):
        bar = "#" * count
        print(f"  {i * 0.1:.1f}: {bar} ({count})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether an agent followed its plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # -- inference ---------------------------------------------------------
    inf = subparsers.add_parser(
        "inference",
        help="Run LLM judge to evaluate plan following",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inf.add_argument(
        "--input-dir", required=True, type=Path, action="append",
        help="Directory containing run_*.json trace files (can be repeated)",
    )
    inf.add_argument(
        "--output-file", required=True, type=Path,
        help="Output JSONL file path",
    )
    inf.add_argument(
        "--model-url", default=DEFAULT_MODEL_URL,
        help="Base URL for the vLLM server",
    )
    inf.add_argument(
        "--port", type=int, default=None,
        help="vLLM server port (shorthand for --model-url)",
    )
    inf.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    inf.add_argument("--api-key", default="EMPTY", help="API key")
    inf.add_argument("--num-threads", type=int, default=8, help="Concurrency")
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top-p", type=float, default=0.95)
    inf.add_argument("--max-tokens", type=int, default=4096)
    inf.add_argument("--max-retries", type=int, default=3)
    inf.add_argument("--force", action="store_true", help="Re-evaluate existing")

    # -- compute -----------------------------------------------------------
    comp = subparsers.add_parser(
        "compute",
        help="Compute statistics from an existing output file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    comp.add_argument(
        "--output-file", required=True, type=Path,
        help="JSONL file with evaluation results",
    )

    args = parser.parse_args()

    if args.mode == "inference":
        run_inference(args)
    elif args.mode == "compute":
        run_compute(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

    # python eval_plan_quality/eval_plan_following.py inference --input-dir runs/gpt-oss-120b/gpt-oss-120b_planning_seed0/ --output-file eval_plan_quality/plan_following_score_gpt_planning_seed0.jsonl --model Qwen/Qwen3.5-122B-A10B