"""MiniMax-M2.5 client — vLLM chat.completions with structured tool calls.

Same flow as qwen35_client.py; separate file per user directive so per-model
defaults / future per-model divergences don't contaminate each other.

Reasoning-preservation contract: the full reasoning trace is replayed on every
turn. vLLM's minimax_m2_append_think parser extracts <think>...</think> content
into message.reasoning_content; _assistant_entry_with_preserved_reasoning
re-injects it back into content as <think>...</think> before appending to
history, which MiniMax's chat template accepts natively.

No enable_thinking chat_template_kwargs needed — MiniMax M2.5 emits reasoning
by default; passing unknown kwargs to the chat template causes errors.

Sampling params are deliberately omitted; vLLM falls back to the model's
generation_config.json (served via --generation-config auto by default).
"""

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
    format_query_with_budget,
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
    _format_original_messages_for_prompt_oss,
    call_trajectory_summarizer,
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
                "function": {
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
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
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
                    },
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


# --- Context-overflow handling ----------------------------------------------

# Matches the tongyi pattern (react_agent.py:312-337): when the context is about
# to overflow, stop making tool calls, instruct the model to produce its final
# answer from what it has, and return. Reactive 400-handling is fragile across
# vLLM versions; proactive token counting is deterministic.
FINAL_ANSWER_INSTRUCTION = (
    "You have now reached the maximum context length you can handle. "
    "Stop making tool calls and, based on all the information gathered above, "
    "provide your final answer now."
)


def _count_prompt_tokens(tokenizer, messages: list, tools: list) -> int:
    """Accurate token count via chat template; falls back to a crude estimate."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(encoded)
    except Exception:
        total = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total += len(c) // 4
        return total


# --- Main agent loop ---------------------------------------------------------

def _assistant_entry_with_preserved_reasoning(assistant_msg) -> dict:
    """Build an assistant history entry that preserves server-returned reasoning.

    vLLM's reasoning parser splits <think>...</think> off into
    message.reasoning_content. To replay the full chain on the next turn we
    inject it back into content wrapped in <think>...</think>. This mirrors
    gpt-oss's behavior (oss_client.py round-trips Responses-API reasoning
    items verbatim) and the `glm45` parser re-accepts the tagged form.
    """
    content = assistant_msg.content or ""
    # vLLM nightly returns the field as `reasoning`; older versions used
    # `reasoning_content`. Check both for forward/back compat.
    reasoning = (
        getattr(assistant_msg, "reasoning_content", None)
        or getattr(assistant_msg, "reasoning", None)
    )
    if reasoning:
        content = f"<think>\n{reasoning}\n</think>\n\n{content}".rstrip() + ("\n" if content else "")

    entry: dict = {"role": "assistant", "content": content}

    if assistant_msg.tool_calls:
        entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_msg.tool_calls
        ]

    return entry


def _force_final_answer(
    client: openai.OpenAI,
    model: str,
    messages: list,
    max_tokens: int,
) -> list:
    """Append a final-answer instruction and make one tool-free call so the
    model must produce text. The prior tool results stay intact in history so
    the model answers with their content in view."""
    messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tool_choice="none",
        max_tokens=max_tokens,
    )
    messages.append(_assistant_entry_with_preserved_reasoning(response.choices[0].message))
    return messages


def _has_unextracted_tool_call(content: str) -> bool:
    """Detect a `<tool_call>` tag that leaked into assistant.content.

    Normally vLLM's qwen3_coder tool parser strips matched
    `<tool_call>...</tool_call>` blocks out of content and emits structured
    `tool_calls`. When content retains a `<tool_call>` token, the parser either
    failed to match (typical cause: `finish_reason=length` truncated
    mid-emission) or the format drifted. Treat as broken and retry.
    """
    return bool(content) and "<minimax:tool_call>" in content


MAX_BROKEN_TOOL_RETRIES = 2


def run_conversation_with_tools(
    client: openai.OpenAI,
    model: str,
    messages: list,
    tools: list,
    max_iterations: int = 100,
    max_tokens: int = 10000,
    tokenizer=None,
    context_token_budget: int | None = None,
    verbose: bool = False,
):
    tool_usage: dict[str, int] = {}
    consecutive_errors = 0
    broken_tool_retries = 0

    for iteration in range(1, max_iterations + 1):
        # Proactive context-overflow check: before each model call, estimate the
        # prompt token count; if it would exceed the budget, force a final answer.
        if tokenizer is not None and context_token_budget is not None:
            prompt_tokens = _count_prompt_tokens(tokenizer, messages, tools)
            if prompt_tokens > context_token_budget:
                if verbose:
                    print(
                        f"[iter {iteration}] context budget exceeded "
                        f"({prompt_tokens} > {context_token_budget}); forcing final answer"
                    )
                messages = _force_final_answer(client, model, messages, max_tokens)
                return messages, tool_usage, "context_limit"

        # On a broken-tool-call retry, double max_tokens so an over-long emission
        # has room to close. Scope the bump to the retry turn only.
        effective_max_tokens = max_tokens * 2 if broken_tool_retries > 0 else max_tokens

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=effective_max_tokens,
            )
            consecutive_errors = 0
        except openai.BadRequestError as e:
            # Belt-and-suspenders for context overflow: our proactive tokenizer
            # estimate can undercount vLLM's real count by ~10K tokens. When it
            # misses, vLLM responds 400 "maximum context length exceeded".
            # Force a final answer instead of retrying (retries with the same
            # messages will fail identically).
            msg = str(e)
            if "maximum context length" in msg.lower() or "context length" in msg.lower():
                if verbose:
                    print(
                        f"[iter {iteration}] context overflow via BadRequest; "
                        f"forcing final answer"
                    )
                messages = _force_final_answer(client, model, messages, max_tokens)
                return messages, tool_usage, "context_limit"
            consecutive_errors += 1
            if verbose or consecutive_errors >= 3:
                print(f"[iter {iteration}] chat.completions BadRequest ({consecutive_errors} consecutive): {e}")
            if consecutive_errors >= 3:
                raise RuntimeError(f"3 consecutive API errors; last: {e}") from e
            continue
        except Exception as e:
            consecutive_errors += 1
            if verbose or consecutive_errors >= 3:
                print(f"[iter {iteration}] chat.completions error ({consecutive_errors} consecutive): {e}")
            if consecutive_errors >= 3:
                raise RuntimeError(f"3 consecutive API errors; last: {e}") from e
            continue

        assistant_msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        has_content = bool((assistant_msg.content or "").strip())
        has_tool_calls = bool(assistant_msg.tool_calls)

        # Broken-tool-call recovery: if content contains `<tool_call>` but the
        # server parser produced no structured tool_calls, the emission was
        # truncated or malformed. Retry (escalating max_tokens) up to N times,
        # then fall back to forcing a final answer from existing evidence.
        if (
            not has_tool_calls
            and _has_unextracted_tool_call(assistant_msg.content or "")
        ):
            if broken_tool_retries < MAX_BROKEN_TOOL_RETRIES:
                broken_tool_retries += 1
                if verbose:
                    print(
                        f"[iter {iteration}] broken <tool_call> in content "
                        f"(finish_reason={finish_reason}); "
                        f"retry {broken_tool_retries}/{MAX_BROKEN_TOOL_RETRIES}"
                    )
                continue  # do not append broken turn; resample
            if verbose:
                print(
                    f"[iter {iteration}] broken <tool_call> persists after "
                    f"{MAX_BROKEN_TOOL_RETRIES} retries; forcing final answer"
                )
            messages = _force_final_answer(client, model, messages, max_tokens)
            return messages, tool_usage, "broken_tool_call"

        # Reset retry counter on any clean response.
        broken_tool_retries = 0

        messages.append(_assistant_entry_with_preserved_reasoning(assistant_msg))

        # Reasoning-only-turn recovery: if the model emitted only reasoning
        # (no visible content, no tool_calls), drop the turn and retry.
        # Mirrors oss_client.py (Responses API pops a terminal reasoning item
        # and continues). Prevents falsely terminating with "completed" when
        # the model stalled mid-think.
        if not has_content and not has_tool_calls:
            messages.pop()
            continue

        if not has_tool_calls:
            return messages, tool_usage, "completed"

        for tc in assistant_msg.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
                result = tool_handler_ref["handler"].execute_tool(tc.function.name, arguments)
                tool_usage[tc.function.name] = tool_usage.get(tc.function.name, 0) + 1
            except Exception as e:
                result = f"Error executing {tc.function.name}: {str(e)}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return messages, tool_usage, "incomplete"


# Thread-local indirection so the tool handler reaches run_conversation_with_tools
# without threading it through every call; mirrors how oss_client.py relies on
# closure capture via _handle_single_query.
tool_handler_ref: dict = {}


# --- Trajectory input builders ----------------------------------------------

def _strip_traj_summary_tags(text: str) -> str:
    return text.strip().split("<trajectory_summary>")[1].split("</trajectory_summary>")[0].strip()


def _assistant_text_from_message(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
    )
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _summarize_llm_call(
    client: openai.OpenAI, model: str, messages: list, max_tokens: int = 8192
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return _assistant_text_from_message(resp.choices[0].message)


_TRAJ_FORMATTERS: dict[str, tuple] = {
    "traj_ext": (_format_trajectory_for_prompt, False, False),
    "traj_orig_ext": (_format_original_messages_for_prompt_oss, False, True),
    "traj_summary_ext": (_format_trajectory_for_prompt, True, False),
    "traj_summary_orig_ext": (_format_original_messages_for_prompt_oss, True, True),
}


def _build_trajectory_user_content(
    qid: str,
    qtext: str,
    args,
    client: openai.OpenAI,
    trajectories_by_id: dict,
    summaries_by_id: dict,
) -> str:
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
            lambda msgs: _summarize_llm_call(client, args.model, msgs, max_tokens=8192),
            question=qtext,
            trajectory_text=traj_text,
            verbose=args.verbose,
        )
        traj_summary = _strip_traj_summary_tags(raw_summary)
    if args.verbose:
        print(f"[{qid}] Prepended trajectory summary to prompt", flush=True)
    return format_query_with_traj_summary(qtext, traj_summary, args.query_template)


# --- Persist run output ------------------------------------------------------

def _extract_think(text: str) -> tuple[str, str]:
    """Split leading <think>...</think> (if any) from the rest of the text."""
    if not text:
        return "", ""
    t = text.lstrip()
    if not t.startswith("<think>"):
        return "", text
    end = t.find("</think>")
    if end == -1:
        return t[len("<think>"):].strip(), ""
    reasoning = t[len("<think>"):end].strip()
    remaining = t[end + len("</think>"):].lstrip()
    return reasoning, remaining


def _persist_response(
    out_dir: str,
    model: str,
    messages: list,
    tool_usage: dict,
    status: str,
    args,
    query_id: str | None,
):
    os.makedirs(out_dir, exist_ok=True)

    call_output_by_id: dict[str, str] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("role") == "tool":
            call_output_by_id[item.get("tool_call_id")] = item.get("content")

    normalized_results: list[dict] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue

        role = item.get("role")

        if role == "assistant":
            reasoning_text, remaining_text = _extract_think(item.get("content") or "")
            if reasoning_text:
                normalized_results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": [reasoning_text],
                    }
                )

            tool_calls = item.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                normalized_results.append(
                    {
                        "type": "tool_call",
                        "tool_name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                        "output": call_output_by_id.get(tc.get("id")),
                    }
                )

            if remaining_text and not tool_calls:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": remaining_text.strip(),
                    }
                )

        elif role == "user":
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                normalized_results.append(
                    {
                        "type": "user",
                        "tool_name": None,
                        "arguments": None,
                        "output": content,
                    }
                )
        # role == "tool" is captured via call_output_by_id above

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
        "model": model,
        "output_dir": str(out_dir),
    }
    if trigger in TRAJ_TRIGGERS:
        metadata["planning_trigger"] = trigger
        if args.trajectory_dir:
            metadata["trajectory_dir"] = args.trajectory_dir
        if trigger in TRAJ_SUMMARY_TRIGGERS and args.trajectory_summary_file:
            metadata["trajectory_summary_file"] = args.trajectory_summary_file

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

def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
):
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
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    summaries_by_id: dict[str, str] = {}
    if args.planning_trigger in TRAJ_SUMMARY_TRIGGERS and args.trajectory_summary_file:
        summaries_by_id = _load_trajectory_summaries(args.trajectory_summary_file, queries)
        print(
            f"Loaded {len(summaries_by_id)} pre-computed summaries from {args.trajectory_summary_file}"
        )

    trajectories_by_id: dict[str, dict] = {}
    if args.planning_trigger in TRAJ_TRIGGERS and not summaries_by_id:
        trajectories_by_id = _load_and_validate_trajectories(args.trajectory_dir, queries)
        print(f"Loaded {len(trajectories_by_id)} trajectories from {args.trajectory_dir}")

    tools = tool_handler.get_tool_definitions()
    tool_handler_ref["handler"] = tool_handler

    # Proactive context-overflow tokenizer (mirrors tongyi pattern). Load once
    # so every thread shares it. apply_chat_template is thread-safe.
    context_tokenizer = None
    context_token_budget = None
    if args.max_model_len and args.context_threshold > 0:
        try:
            context_tokenizer = AutoTokenizer.from_pretrained(args.model)
            context_token_budget = int(args.max_model_len * args.context_threshold)
            print(
                f"Loaded chat-template tokenizer for {args.model}; "
                f"context budget = {context_token_budget} "
                f"({args.context_threshold:.0%} of {args.max_model_len})"
            )
        except Exception as e:
            print(f"Warning: failed to load tokenizer for context tracking: {e}")

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        if args.planning_trigger in TRAJ_TRIGGERS:
            user_content = _build_trajectory_user_content(
                qid, qtext, args, client, trajectories_by_id, summaries_by_id
            )
        elif args.search_budget is not None:
            user_content = format_query_with_budget(qtext, args.search_budget)
        else:
            user_content = format_query(qtext, args.query_template)

        messages = [{"role": "user", "content": user_content}]

        try:
            effective_max_iter = args.search_budget if args.search_budget is not None else args.max_iterations
            messages, tool_usage, status = run_conversation_with_tools(
                client=client,
                model=args.model,
                messages=messages,
                tools=tools,
                max_iterations=effective_max_iter,
                max_tokens=args.max_tokens,
                tokenizer=context_tokenizer,
                context_token_budget=context_token_budget,
                verbose=args.verbose,
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                out_dir,
                args.model,
                messages,
                tool_usage,
                status,
                args,
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


def _validate_planning_args(args) -> None:
    trigger = args.planning_trigger
    if trigger is None:
        return

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


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MiniMax-M2.5 client: vLLM chat.completions with structured tool calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="TSV file path")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M2.5", help="Model name served by vLLM")
    parser.add_argument("--max-tokens", type=int, default=10000, help="max response tokens per chat.completions call")
    parser.add_argument("--output-dir", default="runs_vllm", help="Directory to store run JSON files")
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
            "QUERY_TEMPLATE_GIVEN_TRAJECTORY",
            "QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument("--num-threads", type=int, default=1, help="Parallel threads for dataset mode")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max conversation rounds with function calls")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model-url", default="http://localhost:8000/v1", help="Model URL")
    parser.add_argument(
        "--planning-trigger",
        choices=[
            "traj_ext",
            "traj_summary_ext",
            "traj_orig_ext",
            "traj_summary_orig_ext",
        ],
        default=None,
        help="Trajectory-based input mode.",
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
    parser.add_argument(
        "--search-budget",
        type=int,
        default=None,
        help="Max search turns; uses budget-aware prompt and overrides max_iterations",
    )
    parser.add_argument("--get-document", action="store_true", help="Register the get_document tool")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=131072,
        help="Model's served max_model_len; used together with --context-threshold to "
        "proactively force a final answer before vLLM returns HTTP 400 on context overflow. "
        "Must match the vllm serve --max-model-len value.",
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=0.75,
        help="Fraction of --max-model-len at which to abandon tool calls and request a "
        "final answer. Our apply_chat_template estimate undercounts vLLM's real "
        "count by ~10K tokens (tool-def + chat-template differences), so 0.75 gives "
        "headroom. Set to 0 to disable.",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    _validate_planning_args(args)

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
