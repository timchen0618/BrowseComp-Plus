from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from prompts import (
    format_query,
    format_query_with_trajectory,
    format_query_with_traj_summary,
)
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType
from trajectory_utils import (
    _load_and_validate_trajectories,
    _load_trajectory_summaries,
    _format_trajectory_for_prompt,
    _format_original_messages_for_prompt_truncated,
    call_trajectory_summarizer,
)
from planning_utils import (
    parse_plan_from_response,
    inject_plan_into_messages,
    load_and_validate_plans,
    append_plan_snapshot,
    run_planner_with_retries,
)


TRAJ_TRIGGERS = (
    "traj_ext",
    "traj_orig_ext",
    "traj_summary_ext",
    "traj_summary_orig_ext",
)
TRAJ_SUMMARY_TRIGGERS = ("traj_summary_ext", "traj_summary_orig_ext")


class SearchToolHandler:

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "local_knowledge_base_retrieval",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


# --- Planner helpers ---------------------------------------------------------

REASONING_TRUNCATE = 1000
TOOL_OUTPUT_TRUNCATE = 1000
CONTENT_TRUNCATE = 1000


def _serialize_messages_for_planner(messages: list) -> str:
    """Build a readable string from Responses API messages for the planner."""
    parts = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        itype = item.get("type")
        if role == "user":
            content = item.get("content", "")
            if isinstance(content, str):
                parts.append(
                    f"[User]: {content[:CONTENT_TRUNCATE]}{'...' if len(content) > CONTENT_TRUNCATE else ''}"
                )
            else:
                parts.append("[User]: (structured content)")
        elif role == "assistant":
            content = item.get("content", "")
            if isinstance(content, str):
                parts.append(
                    f"[Assistant]: {content[:CONTENT_TRUNCATE]}{'...' if len(content) > CONTENT_TRUNCATE else ''}"
                )
            else:
                parts.append("[Assistant]: (structured content)")
        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and summary:
                text = " ".join(
                    str(s.get("text", s) if isinstance(s, dict) else s)
                    for s in summary[:3]
                )
            else:
                text_parts = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text",
                        "output_text",
                        "text",
                    }:
                        text_parts.append(str(part.get("text", "")))
                text = " ".join(text_parts)
            text = (text[:REASONING_TRUNCATE] + "...") if len(text) > REASONING_TRUNCATE else text
            if text.strip():
                parts.append(f"[Reasoning]: {text.strip()}")
        elif itype == "function_call":
            name = item.get("name", "?")
            args_str = item.get("arguments", "{}")
            try:
                args = json.loads(args_str)
                q = args.get("user_query") or args.get("query") or args_str[:100]
                parts.append(f"[Tool call] {name}: {q}")
            except json.JSONDecodeError:
                parts.append(f"[Tool call] {name}: {args_str[:100]}...")
        elif itype == "function_call_output":
            output = item.get("output", "")
            if isinstance(output, str):
                out = (output[:TOOL_OUTPUT_TRUNCATE] + "...") if len(output) > TOOL_OUTPUT_TRUNCATE else output
            else:
                out = json.dumps(output)[:TOOL_OUTPUT_TRUNCATE] + "..."
            parts.append(f"[Tool result]: {out}")
        elif itype == "message":
            text_parts = []
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_parts.append(str(part.get("text", "")))
            text = " ".join(text_parts).strip()
            if text:
                parts.append(
                    f"[Message]: {text[:CONTENT_TRUNCATE]}{'...' if len(text) > CONTENT_TRUNCATE else ''}"
                )
    return "\n\n".join(parts) if parts else "(no history)"


def _planner_client(client: openai.OpenAI, planning_url: str) -> openai.OpenAI:
    """Reuse the main client if it already points at the planner URL, else make a new one."""
    if str(getattr(client, "base_url", "") or "").rstrip("/") == planning_url.rstrip("/"):
        return client
    return openai.OpenAI(base_url=planning_url, api_key="EMPTY")


def call_planner(
    client: openai.OpenAI,
    planning_model: str,
    planning_url: str,
    raw_question: str,
    *,
    plan_system_prompt: str,
    max_tries: int = 10,
    verbose: bool = False,
) -> str:
    """Call the planner model to produce a step sequence for the question."""
    messages = [
        {"role": "system", "content": plan_system_prompt},
        {"role": "user", "content": raw_question},
    ]
    return run_planner_with_retries(
        _planner_client(client, planning_url),
        planning_model,
        messages,
        max_tries=max_tries,
        log_label="Planning",
        verbose=verbose,
    )


def call_planner_mid(
    client: openai.OpenAI,
    planning_model: str,
    planning_url: str,
    raw_question: str,
    messages: list,
    *,
    plan_mid_system_prompt: str,
    max_tries: int = 10,
    verbose: bool = False,
    initial_plan: str | None = None,
) -> str:
    """Call the planner with conversation history to produce a revised plan."""
    conv_str = _serialize_messages_for_planner(messages)

    if initial_plan:
        user_content = (
            f"## Original question\n{raw_question}\n\n"
            f"## Conversation history so far\n{conv_str}\n\n"
            f"## Initial plan:\n{initial_plan}\n\n"
            "Based on the question and conversation history above, output the revised plan within <plan></plan> tags."
        )
    else:
        user_content = (
            f"## Original question\n{raw_question}\n\n"
            f"## Conversation history so far\n{conv_str}\n\n"
            "Based on the question and conversation history above, output the revised plan within <plan></plan> tags."
        )
    planner_messages = [
        {"role": "system", "content": plan_mid_system_prompt},
        {"role": "user", "content": user_content},
    ]
    return run_planner_with_retries(
        _planner_client(client, planning_url),
        planning_model,
        planner_messages,
        max_tries=max_tries,
        log_label="Mid-planning",
        verbose=verbose,
    )


def _call_mid_planner_and_inject(
    client: openai.OpenAI,
    planning_config: dict,
    messages: list,
    iteration: int,
    *,
    source: str,
    initial_plan: str | None,
    verbose: bool,
) -> bool:
    """Call mid-planner, inject plan messages into ``messages`` in place, and snapshot.

    Returns True if a plan was injected.
    """
    planner_response = call_planner_mid(
        client,
        planning_model=planning_config["planning_model"],
        planning_url=planning_config["planning_url"],
        raw_question=planning_config["raw_question"],
        messages=messages,
        plan_mid_system_prompt=planning_config["plan_mid_system_prompt"],
        verbose=planning_config.get("verbose", False),
        initial_plan=initial_plan,
    )
    if not (planner_response and planner_response.strip()):
        if verbose:
            print(f"Mid-planner returned empty at iteration {iteration}; skipping injection")
        return False
    messages.extend(inject_plan_into_messages(planner_response))
    parsed = parse_plan_from_response(planner_response)
    append_plan_snapshot(planning_config, source, parsed, iteration)
    if verbose:
        print(f"Injected {source} plan at iteration {iteration}")
    return True


# --- Main agent loop ---------------------------------------------------------

def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
    verbose: bool = False,
    planning_config: dict | None = None,
):
    tool_usage: dict[str, int] = {}
    messages = initial_request["input"]
    iteration = 1
    mid_planning_done = False
    last_reinject_at = 0
    last_revise_at = 0
    pc = planning_config or {}
    initial_plan = pc.get("plan_text", "") or None

    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            request["input"] = messages
            response = client.responses.create(**request)
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")
        messages.extend(response_dict["output"])

        if (
            len(response_dict["output"]) >= 1
            and response_dict["output"][-1]["type"] == "reasoning"
        ):
            messages.pop()
            continue

        function_calls = [
            item for item in response_dict["output"] if item["type"] == "function_call"
        ]

        last_item = response_dict["output"][-1]
        if (not function_calls) or (
            "content" in last_item
            and last_item["content"][0]["type"] == "output_text"
        ):
            return messages, tool_usage, "completed"

        new_messages = messages.copy()

        for tool_call in function_calls:
            try:
                arguments = json.loads(tool_call["arguments"])
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": result,
                    }
                )
            except Exception as e:
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": f"Error executing {tool_call['name']}: {str(e)}",
                    }
                )

        # Mid-conversation planning (one-shot at N tool calls).
        total_tools = sum(tool_usage.values())
        mid_planning_this_iter = False
        if (
            pc.get("trigger") == "after_steps"
            and total_tools >= pc.get("steps", 5)
            and not mid_planning_done
        ):
            if _call_mid_planner_and_inject(
                client,
                pc,
                new_messages,
                iteration,
                source="mid",
                initial_plan=None,
                verbose=verbose,
            ):
                mid_planning_done = True
                mid_planning_this_iter = True

        # Periodic planner revision.
        revise_every = pc.get("revise_every", 0)
        plan_text = pc.get("plan_text", "")
        if (
            revise_every > 0
            and plan_text
            and iteration - last_revise_at >= revise_every
            and not mid_planning_this_iter
            and pc.get("plan_mid_system_prompt")
            and pc.get("planning_model")
            and pc.get("raw_question") is not None
        ):
            if _call_mid_planner_and_inject(
                client,
                pc,
                new_messages,
                iteration,
                source="revise",
                initial_plan=initial_plan,
                verbose=verbose,
            ):
                last_revise_at = iteration

        # Static plan reminder.
        reinject_every = pc.get("reinject_every", 0)
        plan_text = pc.get("plan_text", "")
        if (
            reinject_every > 0
            and plan_text
            and iteration - last_reinject_at >= reinject_every
            and not mid_planning_this_iter
        ):
            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Reminder — here is the plan you were given at the start. "
                        "Use it to guide your next steps:\n" + plan_text
                    ),
                }
            )
            last_reinject_at = iteration
            if verbose:
                print(f"Re-injected plan reminder at iteration {iteration}")

        messages = new_messages
        iteration += 1

    return messages, tool_usage, "incomplete"


# --- Trajectory input builders ----------------------------------------------

def _strip_traj_summary_tags(text: str) -> str:
    return text.strip().split("<trajectory_summary>")[1].split("</trajectory_summary>")[0].strip()


def _assistant_text_from_message(message) -> str:
    """Prefer message.content; fall back to reasoning_content (vLLM GPT-OSS channel split)."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _oss_summarize_llm_call(
    client: openai.OpenAI, model: str, messages: list, max_tokens: int = 8192
) -> str:
    """Adapter that exposes the oss OpenAI client as a simple messages→text callable
    for :func:`trajectory_utils.call_trajectory_summarizer`."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return _assistant_text_from_message(resp.choices[0].message)


# Dispatch: (formatter, wants_summarization, orig_based)
_TRAJ_FORMATTERS: dict[str, tuple] = {
    "traj_ext": (_format_trajectory_for_prompt, False, False),
    "traj_orig_ext": (_format_original_messages_for_prompt_truncated, False, True),
    "traj_summary_ext": (_format_trajectory_for_prompt, True, False),
    "traj_summary_orig_ext": (_format_original_messages_for_prompt_truncated, True, True),
}


def _build_trajectory_user_content(
    qid: str,
    qtext: str,
    args,
    client: openai.OpenAI,
    trajectories_by_id: dict,
    summaries_by_id: dict,
) -> str:
    """Return the user message content for a trajectory-based planning trigger."""
    trigger = args.planning_trigger
    formatter, wants_summary, _orig_based = _TRAJ_FORMATTERS[trigger]

    if not wants_summary:
        traj = trajectories_by_id.get(qid, {})
        traj_text = formatter(
            traj,
            max_chars=args.max_chars,
            reasoning_max_chars=args.reasoning_max_chars,
            tool_output_max_chars=args.tool_output_max_chars,
        )
        if args.verbose:
            print(f"[{qid}] Prepended trajectory to prompt", flush=True)
        return format_query_with_trajectory(qtext, traj_text, args.query_template)

    # Summarization path.
    if qid in summaries_by_id:
        traj_summary = summaries_by_id[qid]
        if args.verbose:
            print(f"[{qid}] Using pre-loaded summary ({len(traj_summary)} chars)", flush=True)
    else:
        traj = trajectories_by_id.get(qid, {})
        traj_text = formatter(
            traj,
            max_chars=args.max_chars,
            reasoning_max_chars=args.reasoning_max_chars,
            tool_output_max_chars=args.tool_output_max_chars,
        )
        if args.verbose:
            print(f"[{qid}] Summarizing trajectory...", flush=True)
        raw_summary = call_trajectory_summarizer(
            lambda msgs: _oss_summarize_llm_call(client, args.model, msgs, max_tokens=8192),
            question=qtext,
            trajectory_text=traj_text,
            verbose=args.verbose,
        )
        traj_summary = _strip_traj_summary_tags(raw_summary)
    if args.verbose:
        print(f"[{qid}] Prepended trajectory summary to prompt", flush=True)
    return format_query_with_traj_summary(qtext, traj_summary, args.query_template)


# --- Persist run output ------------------------------------------------------

def _persist_response(
    out_dir: str,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    status: str,
    args,
    planning_config: dict | None,
    query_id: str | None,
):
    os.makedirs(out_dir, exist_ok=True)

    call_output_by_id: dict[str, str | dict | None] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized_results: list[dict] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue

        itype = item.get("type")

        if itype == "function_call":
            call_id = item.get("call_id")
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "output": call_output_by_id.get(call_id),
                }
            )
        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and len(summary) > 0:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text",
                        "output_text",
                        "text",
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": reasoning_output,
                }
            )
        elif itype == "message":
            text_chunks: list[str] = []
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    }
                )
        elif itype is None and "role" in item and "content" in item:
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                normalized_results.append(
                    {
                        "type": item.get("role", "user"),
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                    }
                )

    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if isinstance(tool_name, str) and ("retrieval" in tool_name.lower())
            else tool_name
        )
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(
            normalized_name, 0
        ) + int(count or 0)

    trigger = args.planning_trigger
    metadata: dict = {
        "model": initial_request.get("model"),
        "reasoning": initial_request.get("reasoning"),
        "output_dir": str(out_dir),
        "planning": args.planning,
    }
    if args.planning:
        metadata["planning_trigger"] = trigger
        if trigger in ("after_steps", "start_and_after_steps"):
            metadata["planning_steps"] = args.planning_steps
        if trigger == "start_ext" and args.planning_file:
            metadata["planning_file"] = args.planning_file
        if trigger in TRAJ_TRIGGERS and args.trajectory_dir:
            metadata["trajectory_dir"] = args.trajectory_dir
        if trigger in TRAJ_SUMMARY_TRIGGERS and args.trajectory_summary_file:
            metadata["trajectory_summary_file"] = args.trajectory_summary_file
    if args.plan_reinject_every > 0:
        metadata["plan_reinject_every"] = args.plan_reinject_every
    if args.plan_revise_every > 0:
        metadata["plan_revise_every"] = args.plan_revise_every
    if args.planning and planning_config:
        plan_text = planning_config.get("plan_text")
        if plan_text and str(plan_text).strip():
            metadata["plan_text"] = plan_text.strip()
        plan_text_history = planning_config.get("plan_text_history")
        if plan_text_history:
            metadata["plan_text_history"] = plan_text_history

    normalized_record = {
        "metadata": metadata,
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }
    if args.save_raw_messages:
        normalized_record["original_messages"] = messages

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


# --- Query processing --------------------------------------------------------

def _build_planning_config(args, qtext: str, injected_plan_text: str) -> dict | None:
    """Assemble the planning_config dict passed into the main loop."""
    if not args.planning:
        return None

    config: dict = {}
    if args.planning_trigger in ("after_steps", "start_and_after_steps"):
        config.update(
            {
                "trigger": "after_steps",
                "steps": args.planning_steps,
                "raw_question": qtext,
                "planning_model": args.planning_model,
                "planning_url": args.planning_url,
                "plan_mid_system_prompt": args.plan_mid_system_prompt,
                "verbose": args.verbose,
            }
        )

    if (args.plan_reinject_every > 0 or args.plan_revise_every > 0) and injected_plan_text:
        config["plan_text"] = injected_plan_text
        if args.plan_reinject_every > 0:
            config["reinject_every"] = args.plan_reinject_every
        if args.plan_revise_every > 0:
            config.update(
                {
                    "revise_every": args.plan_revise_every,
                    "raw_question": qtext,
                    "planning_model": args.planning_model,
                    "planning_url": args.planning_url,
                    "plan_mid_system_prompt": args.plan_mid_system_prompt,
                    "verbose": args.verbose,
                }
            )

    if injected_plan_text and args.planning_trigger in (
        "start",
        "start_and_after_steps",
        "start_ext",
    ):
        source = "start_ext" if args.planning_trigger == "start_ext" else "initial"
        append_plan_snapshot(config, source, injected_plan_text)

    return config or None


def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
):
    """Process a TSV file of (id \t query) pairs and persist responses."""
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries: list[tuple[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids: set[str] = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    plans_by_id: dict[str, str] = {}
    if args.planning and args.planning_trigger == "start_ext":
        plans_by_id = load_and_validate_plans(args.planning_file, queries)
        print(f"Loaded {len(plans_by_id)} plans from {args.planning_file}")

    summaries_by_id: dict[str, str] = {}
    if (
        args.planning
        and args.planning_trigger in TRAJ_SUMMARY_TRIGGERS
        and args.trajectory_summary_file
    ):
        summaries_by_id = _load_trajectory_summaries(args.trajectory_summary_file, queries)
        print(
            f"Loaded {len(summaries_by_id)} pre-computed summaries from {args.trajectory_summary_file}"
        )

    trajectories_by_id: dict[str, dict] = {}
    need_raw_trajectories = (
        args.planning and args.planning_trigger in TRAJ_TRIGGERS and not summaries_by_id
    )
    if need_raw_trajectories:
        trajectories_by_id = _load_and_validate_trajectories(args.trajectory_dir, queries)
        print(f"Loaded {len(trajectories_by_id)} trajectories from {args.trajectory_dir}")

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        """Build request, send, and persist response for one query."""
        if args.planning and args.planning_trigger in TRAJ_TRIGGERS:
            user_content = _build_trajectory_user_content(
                qid, qtext, args, client, trajectories_by_id, summaries_by_id
            )
        else:
            user_content = format_query(qtext, args.query_template)
        input_messages = [{"role": "user", "content": user_content}]

        injected_plan_text = ""
        if args.planning:
            if args.planning_trigger in ("start", "start_and_after_steps"):
                if args.verbose:
                    print(f"[{qid}] Planning mode (start), calling planner...", flush=True)
                planner_response = call_planner(
                    client,
                    args.planning_model,
                    args.planning_url,
                    raw_question=qtext,
                    plan_system_prompt=args.plan_system_prompt,
                    verbose=args.verbose,
                )
                input_messages.extend(inject_plan_into_messages(planner_response))
                injected_plan_text = parse_plan_from_response(planner_response)
                if args.verbose:
                    print(f"[{qid}] Injected plan into messages", flush=True)
            elif args.planning_trigger == "start_ext":
                plan_output = plans_by_id.get(qid, "")
                input_messages.extend(inject_plan_into_messages(plan_output))
                injected_plan_text = parse_plan_from_response(plan_output)
                if args.verbose:
                    print(f"[{qid}] Injected external plan into messages", flush=True)

        planning_config = _build_planning_config(args, qtext, injected_plan_text)

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": input_messages,
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        }

        try:
            messages, tool_usage, status = run_conversation_with_tools(
                client,
                initial_request,
                tool_handler,
                args.max_iterations,
                args.verbose,
                planning_config=planning_config,
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                out_dir,
                initial_request,
                messages,
                tool_usage,
                status,
                args,
                planning_config,
                query_id=qid,
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]
            for _ in as_completed(futures):
                pbar.update(1)


# --- CLI ---------------------------------------------------------------------

def _validate_planning_args(args) -> None:
    """Fail fast on invalid --planning-trigger / flag combinations."""
    if not args.planning:
        return

    trigger = args.planning_trigger

    if trigger == "start_ext" and args.planning_file is None:
        sys.exit("Error: start_ext requires --planning-file")

    if trigger in ("traj_ext", "traj_orig_ext"):
        if args.trajectory_dir is None:
            sys.exit(f"Error: {trigger} requires --trajectory-dir")
        if args.query_template not in (None, "QUERY_TEMPLATE_GIVEN_TRAJECTORY"):
            sys.exit(
                f"Error: {trigger} requires --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY "
                f"(got {args.query_template!r})"
            )

    if trigger in TRAJ_SUMMARY_TRIGGERS:
        if args.trajectory_summary_file is None and args.trajectory_dir is None:
            sys.exit(
                f"Error: {trigger} requires --trajectory-summary-file or --trajectory-dir"
            )
        if args.query_template not in (None, "QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"):
            sys.exit(
                f"Error: {trigger} requires --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY "
                f"(got {args.query_template!r})"
            )

    use_after_steps_planning = trigger in ("after_steps", "start_and_after_steps")
    if use_after_steps_planning and (args.plan_revise_every > 0 or args.plan_reinject_every > 0):
        sys.exit(
            "Error: --planning-trigger after_steps or start_and_after_steps cannot be combined "
            "with --plan-revise-every or --plan-reinject-every (choose at most one)."
        )
    if args.plan_revise_every > 0 and args.plan_reinject_every > 0:
        sys.exit(
            "Error: --plan-revise-every and --plan-reinject-every cannot be used together."
        )


def _load_plan_prompts(args) -> None:
    """Populate args.plan_system_prompt and args.plan_mid_system_prompt from files."""
    repo_root = Path(__file__).resolve().parent.parent

    args.plan_system_prompt = None
    if args.planning and args.planning_trigger in ("start", "start_and_after_steps"):
        plan_path = Path(args.plan_prompt_file)
        if not plan_path.is_absolute():
            plan_path = repo_root / plan_path
        args.plan_system_prompt = plan_path.read_text(encoding="utf-8")

    args.plan_mid_system_prompt = None
    if args.planning and (
        args.planning_trigger in ("after_steps", "start_and_after_steps")
        or args.plan_revise_every > 0
    ):
        mid_path = Path(args.plan_prompt_mid_file)
        if not mid_path.is_absolute():
            mid_path = repo_root / mid_path
        args.plan_mid_system_prompt = mid_path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Responses API with native function calling and local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="TSV file path")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name served by vLLM")
    parser.add_argument("--max-tokens", type=int, default=10000, help="max_output_tokens for Responses API")
    parser.add_argument("--output-dir", default="runs_vllm", help="Directory to store run JSON files")
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
            "QUERY_TEMPLATE_FOR_PLANNING",
            "QUERY_TEMPLATE_GIVEN_TRAJECTORY",
            "QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Parallel threads for dataset mode")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max conversation rounds with function calls")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument("--model-url", default="http://localhost:8000/v1", help="Model URL")
    parser.add_argument(
        "--planning",
        action="store_true",
        help="Enable planning: call planner model first, inject plan into conversation",
    )
    parser.add_argument("--planning-model", type=str, default=None, help="Model for planner (default: same as --model)")
    parser.add_argument("--planning-url", type=str, default=None, help="Base URL for planner (default: same as --model-url)")
    parser.add_argument(
        "--planning-trigger",
        choices=[
            "start",
            "after_steps",
            "start_and_after_steps",
            "start_ext",
            "traj_ext",
            "traj_summary_ext",
            "traj_orig_ext",
            "traj_summary_orig_ext",
        ],
        default="start",
        help="When to run planning. See --help for full list. "
        "after_steps and start_and_after_steps cannot be combined with --plan-revise-every or --plan-reinject-every.",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=5,
        help="Number of tool calls before mid-conversation planning",
    )
    parser.add_argument(
        "--planning-file",
        type=str,
        default=None,
        help="JSONL file with pre-generated plans (required when --planning-trigger=start_ext)",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help="Directory of JSON trajectory files to prepend to the prompt",
    )
    parser.add_argument("--max-chars", type=int, default=500000, help="Max total trajectory chars")
    parser.add_argument("--reasoning-max-chars", type=int, default=3000, help="Max reasoning chars per message")
    parser.add_argument("--tool-output-max-chars", type=int, default=5000, help="Max tool output chars per call")
    parser.add_argument(
        "--trajectory-summary-file",
        type=str,
        default=None,
        help="JSONL file with pre-computed trajectory summaries. "
        "Takes precedence over --trajectory-dir for traj_summary_ext and traj_summary_orig_ext.",
    )
    parser.add_argument(
        "--plan-reinject-every",
        type=int,
        default=0,
        help="Re-inject the plan as a user reminder every N iterations (0 = disabled). "
        "Mutually exclusive with --plan-revise-every and after_steps/start_and_after_steps.",
    )
    parser.add_argument(
        "--plan-revise-every",
        type=int,
        default=0,
        help="Every N iterations, call the planner to revise the plan (0 = disabled). "
        "Mutually exclusive with --plan-reinject-every and after_steps/start_and_after_steps.",
    )
    parser.add_argument(
        "--plan-prompt-file",
        type=str,
        default="prompts/planning_prompt_v2.md",
        help="Path to initial planner system prompt (relative to repo root if not absolute)",
    )
    parser.add_argument(
        "--plan-prompt-mid-file",
        type=str,
        default="prompts/planning_prompt_v2_context.md",
        help="Path to mid-conversation planner system prompt (relative to repo root if not absolute)",
    )
    parser.add_argument(
        "--save-raw-messages",
        action="store_true",
        help="Save raw messages to the output directory",
    )

    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--hf-home", type=str, help="Hugging Face home directory for caching")
    parser.add_argument("--snippet-max-tokens", type=int, default=512, help="Tokens per document snippet")
    parser.add_argument("--k", type=int, default=5, help="Top-k search results")
    parser.add_argument("--get-document", action="store_true", help="Register the get_document tool")

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    _validate_planning_args(args)

    if args.planning_model is None:
        args.planning_model = args.model
    if args.planning_url is None:
        args.planning_url = args.model_url

    _load_plan_prompts(args)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")

    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    qstr = args.query.strip()
    if not qstr.lower().endswith(".tsv"):
        sys.exit("Error: --query must point to a .tsv file (single-query mode is not supported).")
    potential_path = Path(qstr)
    if not potential_path.is_file():
        sys.exit(f"Error: TSV file not found: {potential_path}")
    print("Processing TSV dataset", potential_path)
    _process_tsv_dataset(str(potential_path), client, args, tool_handler)


if __name__ == "__main__":
    main()
