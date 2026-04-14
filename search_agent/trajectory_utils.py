"""Shared helpers for loading, formatting, and summarizing agent trajectories.

Used by both oss_client.py and tongyi_client.py so the trajectory-injection
modes (traj_ext, traj_summary_ext, traj_orig_ext, traj_summary_orig_ext) share
the same plumbing.
"""

import json
import random
import time
from pathlib import Path
from typing import Callable


PROMPT_TRAJECTORY_SUMMARIZER = """
You are a research assistant. Given a question and a detailed trajectory of an agent's search process (including reasoning, tool calls, and tool results), produce a concise summary that captures:

1. **Key findings**: What information was discovered, including relevant document IDs and facts.
2. **Search strategies tried**: What queries were used and which were effective vs. ineffective.
3. **Candidates identified**: Any candidate answers or entities found during the search.
4. **Remaining gaps**: What the agent was unable to find or verify.
5. **Final answer (if any)**: The agent's conclusion and confidence level.

Be concise but preserve all actionable information that would help a new agent continue the research efficiently. Do not include raw tool outputs — summarize them instead. Output the trajectory summary in simple markdown format, covering all the above information. DO NOT directly output the answer or any tool call outputs. Your should ALWAYS output the summary in the following format: (always enclose the summary within <trajectory_summary> tags.)

<trajectory_summary>
{trajectory_summary}
</trajectory_summary>
""".strip()


def _load_and_validate_trajectories(
    trajectory_dir: str, query_tuples: list[tuple[str, str]]
) -> dict[str, dict]:
    """Load external trajectories from a directory of JSON files and validate against query list."""
    traj_path = Path(trajectory_dir)
    if not traj_path.is_dir():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    traj_dict: dict[str, dict] = {}
    for json_file in sorted(traj_path.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            qid = str(obj.get("query_id", ""))
            if qid:
                traj_dict[qid] = obj
        except Exception:
            continue

    query_ids = {qid for qid, _ in query_tuples}

    missing = query_ids - set(traj_dict.keys())
    if missing:
        raise ValueError(
            f"query IDs missing from trajectory directory: {sorted(missing)}"
        )

    return {qid: traj_dict[qid] for qid in query_ids}


def _load_trajectory_summaries(
    summary_file: str, query_tuples: list[tuple[str, str]]
) -> dict[str, str]:
    """Load pre-computed trajectory summaries from a JSONL file.

    Supports two formats:
    - ``summarize_trajectories.py`` output: field ``summary`` (may have ``<trajectory_summary>`` tags)
    - ``selected_tool_calls.jsonl`` style: field ``excerpt`` (plain text, no tags)
    """
    summaries: dict[str, str] = {}
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", ""))
            if not qid:
                continue
            text = obj.get("summary") or obj.get("excerpt") or ""
            if "<trajectory_summary>" in text and "</trajectory_summary>" in text:
                text = text.split("<trajectory_summary>", 1)[1].split("</trajectory_summary>", 1)[0]
            summaries[qid] = text.strip()

    query_ids = {qid for qid, _ in query_tuples}
    missing = query_ids - set(summaries.keys())
    if missing:
        print(f"Warning: {len(missing)} query IDs missing from summary file: {sorted(missing)[:10]}...")

    return {qid: summaries[qid] for qid in query_ids if qid in summaries}


def _format_trajectory_for_prompt(
    trajectory: dict,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
) -> str:
    """Serialize a trajectory's result steps into readable text for prompt prepending."""
    parts = []
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
                q = args_dict.get("user_query") or args_dict.get("query") or str(args)
            except (json.JSONDecodeError, AttributeError):
                q = str(args)
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


def _format_tongyi_trajectory_for_prompt(
    trajectory: dict,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
) -> str:
    """Serialize a Tongyi trajectory's result steps into readable text for prompt prepending.

    The Tongyi result schema differs from the oss schema:
    - ``reasoning.output`` is a plain string (not a list).
    - ``tool_call.arguments`` is always a JSON-encoded string.
    - The list may contain planner-injection ``user`` / ``assistant`` turns.
    """
    parts = []
    for step in trajectory.get("result", []):
        stype = step.get("type")
        if stype == "reasoning":
            text = str(step.get("output", "")).strip()
            if text:
                if reasoning_max_chars > 0 and len(text) > reasoning_max_chars:
                    text = text[:reasoning_max_chars] + "..."
                parts.append(f"[Reasoning]: {text}")
        elif stype == "tool_call":
            tool_name = step.get("tool_name", "?")
            args = step.get("arguments", "{}")
            try:
                args_dict = json.loads(args) if isinstance(args, str) else (args or {})
                if isinstance(args_dict, dict):
                    q = args_dict.get("query") or args_dict.get("user_query") or str(args)[:200]
                else:
                    q = str(args)[:200]
            except (json.JSONDecodeError, AttributeError, TypeError):
                q = str(args)[:200]
            parts.append(f"[Tool call] {tool_name}: {q}")
            output = step.get("output")
            if output:
                out_str = str(output)
                limit = tool_output_max_chars
                parts.append(
                    f"[Tool result]: {out_str[:limit]}{'...' if len(out_str) > limit else ''}"
                )
        elif stype == "output_text":
            text = str(step.get("output", "")).strip()
            if text:
                parts.append(f"[Final answer]: {text}")
        elif stype == "user":
            text = str(step.get("output", "")).strip()
            if text:
                limit = tool_output_max_chars
                parts.append(
                    f"[Injected]: {text[:limit]}{'...' if len(text) > limit else ''}"
                )
        # Skip assistant-type template turns like "I am calling a planner...".

    result = "\n\n".join(parts) if parts else "(no trajectory steps)"
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (trajectory truncated)"
    return result


def _format_original_messages_for_prompt(
    trajectory: dict,
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


def _format_original_messages_for_prompt_truncated(
    trajectory: dict,
    max_chars: int = 0,
    reasoning_max_chars: int = 3000,
    tool_output_max_chars: int = 5000,
) -> str:
    """Serialize ``original_messages`` into a JSON string, with light truncation.

    This keeps the same I/O shape as ``_format_original_messages_for_prompt``:
    input trajectory dict → output string (JSON-serialized list).

    Compared to the raw dump, it attempts to truncate very long reasoning/tool
    outputs inside the messages, similar in spirit to
    ``select_useful_tool_calls.format_trajectory_for_prompt_orig``.
    """
    msgs = trajectory.get("original_messages", [])
    if not msgs:
        return "(no original messages)"

    try:
        msgs_clone = json.loads(json.dumps(msgs, ensure_ascii=False))
    except Exception:
        # If deep-copying via JSON fails (non-serializable), fall back to raw dump.
        result = json.dumps(msgs, ensure_ascii=False)
        if max_chars > 0 and len(result) > max_chars:
            result = result[:max_chars] + "\n\n... (messages truncated)"
        return result

    for m in msgs_clone if isinstance(msgs_clone, list) else []:
        if not isinstance(m, dict):
            continue
        stype = m.get("type", "")

        if stype == "reasoning":
            content = m.get("content")
            if isinstance(content, list) and reasoning_max_chars > 0:
                for i, item in enumerate(content):
                    if isinstance(item, str) and len(item) > reasoning_max_chars:
                        content[i] = item[:reasoning_max_chars] + "..."
        elif stype == "function_call_output":
            out = m.get("output")
            if isinstance(out, str) and tool_output_max_chars > 0 and len(out) > tool_output_max_chars:
                m["output"] = out[:tool_output_max_chars] + "..."

    result = json.dumps(msgs_clone, ensure_ascii=False)
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars] + "\n\n... (messages truncated)"
    return result


def call_trajectory_summarizer(
    llm_call: Callable[[list[dict]], str],
    *,
    question: str,
    trajectory_text: str,
    system_prompt: str = PROMPT_TRAJECTORY_SUMMARIZER,
    max_tries: int = 5,
    verbose: bool = False,
) -> str:
    """Summarize a trajectory via an arbitrary LLM callable.

    ``llm_call`` takes a messages list and returns the assistant content string.
    Retries up to ``max_tries`` times; if all retries miss the
    ``<trajectory_summary>`` tags, wraps the last non-empty response with the
    tags and returns it.
    """
    base_sleep_time = 1
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nTrajectory:\n{trajectory_text}"},
    ]

    last_text = ""
    for attempt in range(max_tries):
        try:
            text = (llm_call(messages) or "").strip()
            if not text:
                if verbose:
                    print(f"Warning: Summarizer attempt {attempt + 1} received an empty response.")
            elif "<trajectory_summary>" not in text or "</trajectory_summary>" not in text:
                last_text = text
                if verbose:
                    print(
                        f"Warning: Summarizer attempt {attempt + 1} missing "
                        "<trajectory_summary> tags, retrying..."
                    )
            else:
                if verbose:
                    print("Trajectory summarization successful")
                return text
        except Exception as e:
            if verbose:
                print(f"Summarizer error (attempt {attempt + 1}/{max_tries}): {e}")

        if attempt < max_tries - 1:
            sleep_time = min(
                base_sleep_time * (2**attempt) + random.uniform(0, 1), 30
            )
            if verbose:
                print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            if last_text:
                print("Trajectory summarizer: all retries exhausted, wrapping last response with tags")
                return f"<trajectory_summary>\n{last_text}\n</trajectory_summary>"
            print("Trajectory summarizer: all retries exhausted, returning empty summary")
            return ""

    return ""
