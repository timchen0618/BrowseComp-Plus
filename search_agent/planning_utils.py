"""Shared planning-helper utilities used by multiple agent clients.

Covers plan parsing/injection, pre-generated plan-file loading, snapshot
bookkeeping, and a retry wrapper for OpenAI-style planner calls. The
``_serialize_messages_for_planner`` helper intentionally lives in each client
because the message format (Responses API vs. react <think>/<tool_call>) is
client-specific.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


def parse_plan_from_response(response: str) -> str:
    """Extract plan content from planner response. Fallback to full response or empty."""
    if response and "<plan>" in response and "</plan>" in response:
        try:
            return response.split("<plan>")[1].split("</plan>")[0].strip()
        except IndexError:
            pass
    return response.strip() if response else ""


def inject_plan_into_messages(planner_response: str) -> list[dict]:
    """Return the two messages to inject: assistant intro and user with plan."""
    plan_content = parse_plan_from_response(planner_response)
    return [
        {
            "role": "assistant",
            "content": "I am calling a planner to plan a sequence of actions to answer the user's question.",
        },
        {
            "role": "user",
            "content": "Here is the planner's response; please follow the plan to answer the user's question: "
            + plan_content,
        },
    ]


def load_and_validate_plans(
    plan_file: str, query_tuples: list[tuple[str, str]]
) -> dict[str, str]:
    """Load pre-generated plans from a JSONL file and validate against the query list."""
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    plan_dict: dict[str, str] = {}
    with plan_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", ""))
            output = obj.get("output", "")
            if qid:
                plan_dict[qid] = output if isinstance(output, str) else json.dumps(output)

    query_ids = {qid for qid, _ in query_tuples}

    if len(plan_dict) != len(query_tuples):
        raise ValueError(
            f"Plan file has {len(plan_dict)} entries but query file has {len(query_tuples)}; "
            "lengths must match"
        )
    if set(plan_dict.keys()) != query_ids:
        missing = query_ids - set(plan_dict.keys())
        extra = set(plan_dict.keys()) - query_ids
        parts = []
        if missing:
            parts.append(f"query IDs missing from plan file: {sorted(missing)}")
        if extra:
            parts.append(f"plan file has extra IDs not in queries: {sorted(extra)}")
        raise ValueError(
            "Query IDs do not match between plan file and query file: " + "; ".join(parts)
        )

    return plan_dict


def append_plan_snapshot(
    planning_config: dict | None,
    source: str,
    parsed_text: str,
    iteration: int | None = None,
) -> None:
    """Append parsed plan to history and set canonical plan_text (last snapshot wins)."""
    if not planning_config or not (parsed_text or "").strip():
        return
    text = parsed_text.strip()
    entry: dict = {"source": source, "text": text}
    if iteration is not None:
        entry["iteration"] = iteration
    pm = planning_config.get("planning_model")
    if pm:
        entry["planning_model"] = pm
    planning_config.setdefault("plan_text_history", []).append(entry)
    planning_config["plan_text"] = text


def run_planner_with_retries(
    planner_client: Any,
    model: str,
    messages: list[dict],
    *,
    max_tries: int = 10,
    max_tokens: int = 4096,
    log_label: str = "Planning",
    verbose: bool = False,
) -> str:
    """Call an OpenAI-style chat.completions endpoint with exponential backoff.

    Returns the stripped assistant content on success, or ``""`` after all
    retries are exhausted.
    """
    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            chat_response = planner_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            content = chat_response.choices[0].message.content
            if content and content.strip():
                if verbose:
                    print(f"{log_label} call successful")
                return content.strip()
            if verbose:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")
        except Exception as e:
            if verbose:
                print(f"{log_label} error (attempt {attempt + 1}/{max_tries}): {e}")

        if attempt < max_tries - 1:
            sleep_time = min(base_sleep_time * (2**attempt) + random.uniform(0, 1), 30)
            if verbose:
                print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print(f"{log_label}: all retries exhausted, returning empty plan")
            return ""

    return ""
