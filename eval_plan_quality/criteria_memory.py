"""
Extract, accumulate, and evolve plan-quality criteria from LLM judge outputs.

The script maintains a persistent JSON knowledge base of criteria that grows
and refines itself each time new judge results are ingested.

Modes:
  extract          — Parse judge thinking from comparison or pointwise results,
                     use an LLM to distill new criteria, and merge them into
                     the persistent knowledge base.
  show             — Display the current knowledge base.
  generate-prompts — Read the knowledge base and produce two refined prompts:
                     (1) a plan-generation prompt and (2) a plan-evaluation prompt.

Usage:
    # Extract criteria from pairwise comparison results
    python eval_plan_quality/criteria_memory.py extract \
        --judge-file eval_plan_quality/comparison_results.jsonl \
        --source-type comparison \
        --model Qwen/Qwen3-32B --num-threads 4

    # Extract criteria from pointwise scoring results
    python eval_plan_quality/criteria_memory.py extract \
        --judge-file eval_plan_quality/judge_scores.jsonl \
        --source-type pointwise \
        --model Qwen/Qwen3-32B --num-threads 4

    # Show the current knowledge base
    python eval_plan_quality/criteria_memory.py show

    # Generate refined prompts from the knowledge base
    python eval_plan_quality/criteria_memory.py generate-prompts \
        --model Qwen/Qwen3-32B
"""

import argparse
import json
import random
import re
import sys
import time
import threading
import concurrent.futures
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import openai
from tqdm import tqdm

T = TypeVar("T")

DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_KB_PATH = Path("eval_plan_quality/criteria_kb.json")
DEFAULT_PROMPTS_DIR = Path("eval_plan_quality/prompts")

BATCH_SIZE = 20  # number of judge outputs to send per extraction call


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Retry helpers (same as compare_plans.py / llm_judge.py)
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
    per_attempt_timeout_s: float = 300.0,
    total_deadline_s: float = 900.0,
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
# Knowledge base operations
# ---------------------------------------------------------------------------

def load_kb(path: Path) -> Dict[str, Any]:
    """Load the criteria knowledge base, or return a fresh one."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 1,
        "created": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "extraction_history": [],
        "criteria": [],
    }


def save_kb(kb: Dict[str, Any], path: Path) -> None:
    """Save the knowledge base to disk."""
    kb["last_updated"] = datetime.utcnow().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    log(f"Knowledge base saved to {path}")


def merge_criteria(
    existing: List[Dict[str, Any]],
    new_criteria: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge new criteria into the existing list.

    Each criterion has:
        name:        short identifier
        description: what the criterion means
        guidance:    actionable advice for plan authors
        examples:    list of concrete examples from judge reasoning
        frequency:   how many times this pattern was observed
        sources:     list of source files that contributed
        tags:        list of category tags

    Merging strategy:
        - Match by name (case-insensitive, fuzzy prefix match).
        - On match: increment frequency, append new examples (dedup),
          update description/guidance if the new version is longer,
          add source info.
        - On no match: add as new criterion.
    """
    merged = deepcopy(existing)
    name_to_idx: Dict[str, int] = {}
    for i, c in enumerate(merged):
        name_to_idx[c["name"].lower().strip()] = i

    for nc in new_criteria:
        nc_key = nc["name"].lower().strip()
        # Try exact match first, then substring match
        match_idx = name_to_idx.get(nc_key)
        if match_idx is None:
            for existing_key, idx in name_to_idx.items():
                # Fuzzy: one is a substring of the other
                if nc_key in existing_key or existing_key in nc_key:
                    match_idx = idx
                    break

        if match_idx is not None:
            ec = merged[match_idx]
            ec["frequency"] = ec.get("frequency", 1) + nc.get("frequency", 1)
            # Keep the longer/better description
            if len(nc.get("description", "")) > len(ec.get("description", "")):
                ec["description"] = nc["description"]
            if len(nc.get("guidance", "")) > len(ec.get("guidance", "")):
                ec["guidance"] = nc["guidance"]
            # Append new examples (dedup by string)
            existing_examples = set(ec.get("examples", []))
            for ex in nc.get("examples", []):
                if ex not in existing_examples:
                    ec.setdefault("examples", []).append(ex)
                    existing_examples.add(ex)
            # Merge tags
            existing_tags = set(ec.get("tags", []))
            for tag in nc.get("tags", []):
                if tag not in existing_tags:
                    ec.setdefault("tags", []).append(tag)
                    existing_tags.add(tag)
            # Merge sources
            existing_sources = set(ec.get("sources", []))
            for src in nc.get("sources", []):
                if src not in existing_sources:
                    ec.setdefault("sources", []).append(src)
                    existing_sources.add(src)
        else:
            nc.setdefault("frequency", 1)
            nc.setdefault("examples", [])
            nc.setdefault("tags", [])
            nc.setdefault("sources", [])
            merged.append(nc)
            name_to_idx[nc_key] = len(merged) - 1

    # Sort by frequency descending
    merged.sort(key=lambda c: c.get("frequency", 0), reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Extraction: parse judge outputs and distill criteria via LLM
# ---------------------------------------------------------------------------

def load_judge_outputs(path: Path, source_type: str) -> List[Dict[str, str]]:
    """Load judge thinking from a results JSONL.

    Returns list of dicts with keys: query_id, question, thinking.
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            judge_output = obj.get("judge_output", "")

            # Try to extract thinking from JSON structured output
            thinking = ""
            try:
                parsed = json.loads(judge_output)
                thinking = parsed.get("thinking", "")
            except (json.JSONDecodeError, TypeError):
                # Free-text output — the entire output IS the thinking
                thinking = judge_output

            if not thinking.strip():
                continue

            records.append({
                "query_id": str(obj.get("query_id", "")),
                "question": obj.get("question", ""),
                "thinking": thinking.strip(),
            })
    return records


EXTRACT_SYSTEM_PROMPT = """\
You are an expert at analyzing LLM judge feedback to extract recurring patterns, \
criteria, and actionable guidelines for what makes a good plan for an agentic \
search system.

You will be given a batch of judge reasoning outputs (from evaluating search plans). \
Your job is to identify the criteria, rules, and guidelines that the judge \
repeatedly uses to distinguish good plans from bad ones.

For each criterion you identify, provide:
1. A short name (2-5 words)
2. A description of what this criterion means
3. Actionable guidance for a plan author (what to do / what to avoid)
4. 1-2 concrete examples from the judge's reasoning that illustrate this criterion
5. Tags categorizing this criterion (e.g., "search_strategy", "structure", \
"robustness", "efficiency", "constraint_handling")

Output a JSON array of criteria objects:
```json
[
  {
    "name": "...",
    "description": "...",
    "guidance": "...",
    "examples": ["...", "..."],
    "tags": ["...", "..."]
  }
]
```

Focus on:
- Patterns that appear across MULTIPLE judge outputs (not one-off observations)
- Actionable advice (not vague platitudes)
- Specific failure modes the judge calls out
- What makes the winner better than the loser in pairwise comparisons
- Concrete attributes of high-scoring vs low-scoring plans

Do NOT include:
- Domain-specific facts about particular questions
- Generic advice like "be thorough" without specifics
- Criteria that only apply to one specific question type"""

EXTRACT_USER_TEMPLATE = """\
Here are {n} judge reasoning outputs from {source_type} evaluation of search plans. \
Extract the recurring criteria and guidelines.

{judge_outputs}"""


def build_extraction_batches(
    records: List[Dict[str, str]],
    batch_size: int = BATCH_SIZE,
) -> List[str]:
    """Group judge outputs into batches for extraction."""
    batches = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        parts = []
        for j, rec in enumerate(batch):
            parts.append(
                f"--- Judge Output {j + 1} (query_id={rec['query_id']}) ---\n"
                f"Question: {rec['question'][:200]}...\n\n"
                f"{rec['thinking']}\n"
            )
        batches.append("\n".join(parts))
    return batches


def extract_criteria_from_batch(
    client: openai.OpenAI,
    model: str,
    batch_text: str,
    source_type: str,
    n_records: int,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Send a batch of judge outputs to the LLM for criteria extraction."""
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACT_USER_TEMPLATE.format(
                n=n_records,
                source_type=source_type,
                judge_outputs=batch_text,
            ),
        },
    ]

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    raw = _retry_with_deadline(_call, max_retries=max_retries)

    # Parse JSON array from the response (may be wrapped in ```json ... ```)
    raw = raw.strip()
    # Remove markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Find the JSON array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        log(f"WARNING: Could not parse criteria from LLM response:\n{raw[:500]}")
        return []

    try:
        criteria = json.loads(match.group(0))
    except json.JSONDecodeError:
        log(f"WARNING: JSON parse failed:\n{match.group(0)[:500]}")
        return []

    # Validate structure
    valid = []
    for c in criteria:
        if isinstance(c, dict) and "name" in c:
            valid.append({
                "name": c.get("name", ""),
                "description": c.get("description", ""),
                "guidance": c.get("guidance", ""),
                "examples": c.get("examples", []),
                "tags": c.get("tags", []),
                "frequency": 1,
                "sources": [],
            })
    return valid


# ---------------------------------------------------------------------------
# Consolidation: after extracting from all batches, merge duplicates via LLM
# ---------------------------------------------------------------------------

CONSOLIDATE_SYSTEM_PROMPT = """\
You are an expert at organizing and deduplicating criteria for evaluating \
search plans. You will be given a list of criteria that were extracted from \
different batches of judge outputs. Many are duplicates or near-duplicates.

Your job is to consolidate them into a clean, non-redundant list where:
1. Near-duplicate criteria are merged into one (pick the best name and combine guidance)
2. Closely related criteria are kept separate only if they are genuinely distinct
3. Each criterion has clear, actionable guidance
4. Examples are preserved (take the best 2-3 per criterion)

Output a JSON array with the same schema:
```json
[
  {
    "name": "...",
    "description": "...",
    "guidance": "...",
    "examples": ["...", "..."],
    "tags": ["...", "..."],
    "frequency": <integer, sum of merged frequencies>
  }
]
```

Sort by frequency (highest first). Aim for 15-30 distinct criteria."""


def consolidate_criteria(
    client: openai.OpenAI,
    model: str,
    all_criteria: List[Dict[str, Any]],
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Use the LLM to deduplicate and consolidate extracted criteria."""
    # Prepare a compact representation for the LLM
    compact = []
    for c in all_criteria:
        compact.append({
            "name": c["name"],
            "description": c["description"],
            "guidance": c["guidance"],
            "examples": c.get("examples", [])[:2],
            "tags": c.get("tags", []),
            "frequency": c.get("frequency", 1),
        })

    messages = [
        {"role": "system", "content": CONSOLIDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here are {len(compact)} criteria extracted from multiple batches. "
                f"Consolidate them:\n\n{json.dumps(compact, indent=2, ensure_ascii=False)}"
            ),
        },
    ]

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    raw = _retry_with_deadline(_call, max_retries=max_retries)
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        log("WARNING: Consolidation parse failed, returning raw criteria")
        return all_criteria

    try:
        consolidated = json.loads(match.group(0))
    except json.JSONDecodeError:
        log("WARNING: Consolidation JSON parse failed, returning raw criteria")
        return all_criteria

    valid = []
    for c in consolidated:
        if isinstance(c, dict) and "name" in c:
            valid.append({
                "name": c.get("name", ""),
                "description": c.get("description", ""),
                "guidance": c.get("guidance", ""),
                "examples": c.get("examples", []),
                "tags": c.get("tags", []),
                "frequency": c.get("frequency", 1),
                "sources": [],
            })
    return valid


# ---------------------------------------------------------------------------
# Prompt generation from the knowledge base
# ---------------------------------------------------------------------------

GENERATE_PROMPT_SYSTEM = """\
You are an expert prompt engineer for agentic search systems. Given a set of \
empirically-derived criteria for what makes a good search plan, your job is to \
produce two polished prompts:

1. **Plan Generation Prompt**: A system prompt that instructs an LLM to create \
a high-quality search plan for a given question. It should incorporate the criteria \
as concrete guidelines the LLM must follow when constructing plans.

2. **Plan Evaluation Prompt**: A system prompt for an LLM judge that scores a \
single plan on a 1-5 scale. It should use the criteria as a detailed rubric \
with clear score anchors.

Requirements for both prompts:
- Incorporate ALL the criteria naturally (don't just list them mechanically)
- Be specific and actionable (avoid vague instructions)
- Include concrete examples of good and bad patterns where helpful
- Use the <start_system_prompt>/<end_system_prompt> and \
<start_user_prompt>/<end_user_prompt> tag format

For the plan generation prompt, the user section should have a {question} placeholder.
For the evaluation prompt, the user section should have {question} and {plan} placeholders.

Output the two prompts separated by a line of "=" characters (at least 40).
Label them clearly: "=== PLAN GENERATION PROMPT ===" and "=== PLAN EVALUATION PROMPT ==="."""


def generate_prompts(
    client: openai.OpenAI,
    model: str,
    kb: Dict[str, Any],
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> str:
    """Generate refined prompts from the knowledge base criteria."""
    criteria = kb.get("criteria", [])
    if not criteria:
        return "No criteria in knowledge base. Run 'extract' first."

    criteria_text = json.dumps(criteria, indent=2, ensure_ascii=False)

    messages = [
        {"role": "system", "content": GENERATE_PROMPT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are {len(criteria)} empirically-derived criteria for plan quality, "
                f"sorted by frequency (most commonly observed first):\n\n{criteria_text}\n\n"
                f"Generate the two prompts."
            ),
        },
    ]

    def _call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _retry_with_deadline(_call, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Mode: extract
# ---------------------------------------------------------------------------

def run_extract(args: argparse.Namespace) -> None:
    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    # Load judge outputs
    log(f"Loading judge outputs from {args.judge_file}")
    records = load_judge_outputs(args.judge_file, args.source_type)
    log(f"  → {len(records)} records with non-empty thinking")

    if not records:
        print("No judge outputs found.", file=sys.stderr)
        sys.exit(1)

    # Build batches
    batches = build_extraction_batches(records, batch_size=args.batch_size)
    log(f"  → {len(batches)} batches of up to {args.batch_size} outputs each")

    # Create client
    client = openai.OpenAI(
        base_url=args.model_url,
        api_key=args.api_key,
        timeout=600.0,
    )

    # Extract criteria from each batch (concurrent)
    all_criteria: List[Dict[str, Any]] = []
    lock = threading.Lock()
    source_label = str(args.judge_file)

    def _process_batch(batch_idx: int) -> None:
        batch_text = batches[batch_idx]
        n_records = min(args.batch_size, len(records) - batch_idx * args.batch_size)
        try:
            criteria = extract_criteria_from_batch(
                client, args.model, batch_text, args.source_type,
                n_records, max_retries=args.max_retries,
            )
            for c in criteria:
                c["sources"] = [source_label]
            with lock:
                all_criteria.extend(criteria)
        except Exception as e:
            log(f"Batch {batch_idx} failed: {e!r}")

    if args.num_threads <= 1:
        for i in tqdm(range(len(batches)), desc="Extracting"):
            _process_batch(i)
    else:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as pool,
            tqdm(total=len(batches), desc="Extracting") as pbar,
        ):
            futures = [pool.submit(_process_batch, i) for i in range(len(batches))]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                pbar.update(1)

    log(f"Extracted {len(all_criteria)} raw criteria from {len(batches)} batches")

    if not all_criteria:
        print("No criteria extracted.", file=sys.stderr)
        sys.exit(1)

    # Consolidate (deduplicate) via LLM
    log("Consolidating criteria via LLM...")
    consolidated = consolidate_criteria(
        client, args.model, all_criteria, max_retries=args.max_retries,
    )
    for c in consolidated:
        c["sources"] = [source_label]
    log(f"Consolidated to {len(consolidated)} criteria")

    # Load existing KB and merge
    kb = load_kb(args.kb_path)
    existing = kb.get("criteria", [])
    log(f"Existing KB has {len(existing)} criteria")

    kb["criteria"] = merge_criteria(existing, consolidated)
    kb["extraction_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "source_file": str(args.judge_file),
        "source_type": args.source_type,
        "records_processed": len(records),
        "raw_criteria_extracted": len(all_criteria),
        "consolidated_criteria": len(consolidated),
        "kb_criteria_after_merge": len(kb["criteria"]),
    })

    save_kb(kb, args.kb_path)
    print(f"\nDone. Knowledge base now has {len(kb['criteria'])} criteria.")
    print(f"  Saved to: {args.kb_path}")


# ---------------------------------------------------------------------------
# Mode: show
# ---------------------------------------------------------------------------

def run_show(args: argparse.Namespace) -> None:
    kb = load_kb(args.kb_path)
    criteria = kb.get("criteria", [])

    if not criteria:
        print("Knowledge base is empty. Run 'extract' first.")
        return

    print(f"Knowledge base: {args.kb_path}")
    print(f"  Created:      {kb.get('created', '?')}")
    print(f"  Last updated: {kb.get('last_updated', '?')}")
    print(f"  Extractions:  {len(kb.get('extraction_history', []))}")
    print(f"  Criteria:     {len(criteria)}")
    print()

    # Show extraction history
    history = kb.get("extraction_history", [])
    if history:
        print("Extraction history:")
        for h in history:
            print(f"  [{h['timestamp']}] {h['source_file']} "
                  f"({h['source_type']}, {h['records_processed']} records "
                  f"→ {h['consolidated_criteria']} criteria)")
        print()

    # Show criteria
    print(f"{'='*72}")
    print(f"Criteria (sorted by frequency)")
    print(f"{'='*72}")
    for i, c in enumerate(criteria, 1):
        freq = c.get("frequency", 0)
        tags = ", ".join(c.get("tags", []))
        print(f"\n{i}. {c['name']}  [freq={freq}]  [{tags}]")
        print(f"   Description: {c.get('description', '')}")
        print(f"   Guidance:    {c.get('guidance', '')}")
        examples = c.get("examples", [])
        if examples:
            print(f"   Examples:")
            for ex in examples[:3]:
                # Truncate long examples
                ex_short = ex[:200] + "..." if len(ex) > 200 else ex
                print(f"     - {ex_short}")
        sources = c.get("sources", [])
        if sources:
            print(f"   Sources: {', '.join(sources[:5])}")


# ---------------------------------------------------------------------------
# Mode: generate-prompts
# ---------------------------------------------------------------------------

def run_generate_prompts(args: argparse.Namespace) -> None:
    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    kb = load_kb(args.kb_path)
    if not kb.get("criteria"):
        print("Knowledge base is empty. Run 'extract' first.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key=args.api_key,
        timeout=600.0,
    )

    log(f"Generating prompts from {len(kb['criteria'])} criteria...")
    result = generate_prompts(client, args.model, kb, max_retries=args.max_retries)

    # Write to output file
    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    log(f"Prompts written to {output_path}")

    # Also try to split and save as separate files
    parts = re.split(r"={40,}", result)
    gen_prompt = None
    eval_prompt = None
    for part in parts:
        if "PLAN GENERATION PROMPT" in part:
            gen_prompt = part
        elif "PLAN EVALUATION PROMPT" in part:
            eval_prompt = part

    if gen_prompt:
        # Extract just the prompt content (after the header line)
        gen_prompt = re.sub(r".*PLAN GENERATION PROMPT.*\n", "", gen_prompt).strip()
        gen_path = DEFAULT_PROMPTS_DIR / "generated_plan_generation.txt"
        gen_path.write_text(gen_prompt, encoding="utf-8")
        print(f"Plan generation prompt → {gen_path}")

    if eval_prompt:
        eval_prompt = re.sub(r".*PLAN EVALUATION PROMPT.*\n", "", eval_prompt).strip()
        eval_path = DEFAULT_PROMPTS_DIR / "generated_plan_evaluation.txt"
        eval_path.write_text(eval_prompt, encoding="utf-8")
        print(f"Plan evaluation prompt → {eval_path}")

    if not gen_prompt and not eval_prompt:
        print(f"Full output (could not split) → {output_path}")

    print(f"\nDone. Generated prompts from {len(kb['criteria'])} criteria.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, accumulate, and evolve plan-quality criteria from LLM judge outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # -- extract -----------------------------------------------------------
    ext = subparsers.add_parser(
        "extract",
        help="Extract criteria from judge outputs and merge into knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ext.add_argument("--judge-file", required=True, type=Path,
                     help="JSONL file with judge outputs (comparison or pointwise)")
    ext.add_argument("--source-type", required=True,
                     choices=["comparison", "pointwise"],
                     help="Type of judge output: 'comparison' (pairwise) or 'pointwise' (scoring)")
    ext.add_argument("--kb-path", type=Path, default=DEFAULT_KB_PATH,
                     help="Path to the persistent criteria knowledge base JSON")
    ext.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                     help="Number of judge outputs per extraction batch")
    ext.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                     help="Base URL for the external vLLM server")
    ext.add_argument("--port", type=int, default=None,
                     help="vLLM server port (shorthand for --model-url)")
    ext.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    ext.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    ext.add_argument("--num-threads", type=int, default=4,
                     help="Number of concurrent extraction requests")
    ext.add_argument("--max-retries", type=int, default=3,
                     help="Max retries per request")

    # -- show --------------------------------------------------------------
    sh = subparsers.add_parser(
        "show",
        help="Display the current knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sh.add_argument("--kb-path", type=Path, default=DEFAULT_KB_PATH,
                    help="Path to the criteria knowledge base JSON")

    # -- generate-prompts --------------------------------------------------
    gp = subparsers.add_parser(
        "generate-prompts",
        help="Generate refined plan-generation and evaluation prompts from the knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gp.add_argument("--kb-path", type=Path, default=DEFAULT_KB_PATH,
                    help="Path to the criteria knowledge base JSON")
    gp.add_argument("--output-file", type=Path,
                    default=DEFAULT_PROMPTS_DIR / "generated_prompts.txt",
                    help="Output file for the generated prompts")
    gp.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                    help="Base URL for the external vLLM server")
    gp.add_argument("--port", type=int, default=None,
                    help="vLLM server port (shorthand for --model-url)")
    gp.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    gp.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    gp.add_argument("--max-retries", type=int, default=3,
                    help="Max retries per request")

    # -- dispatch ----------------------------------------------------------
    args = parser.parse_args()

    if args.mode == "extract":
        run_extract(args)
    elif args.mode == "show":
        run_show(args)
    elif args.mode == "generate-prompts":
        run_generate_prompts(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
