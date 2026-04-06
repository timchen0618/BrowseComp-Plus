from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from prompts import format_query, format_query_with_trajectory, format_query_for_planning
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType


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


def _sanitize_input_items(items: list) -> list:
    """Remove channel/recipient keys with None values from input items.
    vLLM gpt-oss Responses API rejects items with 'Unknown channel: None' / 'Unknown recipient: None'.
    Output items from the API may include these fields; we strip them before sending back as input.
    """
    out = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        cleaned = {k: v for k, v in item.items() if not (k in ("channel", "recipient") and v is None)}
        if "content" in cleaned and isinstance(cleaned["content"], list):
            cleaned["content"] = [
                {k: v for k, v in c.items() if not (k in ("channel", "recipient") and v is None)}
                if isinstance(c, dict) else c
                for c in cleaned["content"]
            ]
        out.append(cleaned)
    return out


def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
    verbose: bool = False,
    planning_config: dict | None = None,
):
    tool_usage = {}
    messages = initial_request["input"]
    iteration = 1
    mid_planning_done = False
    last_reinject_at = 0  # iteration count at last plan re-injection

    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            # request["input"] = _sanitize_input_items(messages)
            request["input"] = messages
            # #region agent log
            _payload_bytes = len(json.dumps(request, default=str).encode("utf-8"))
            _input_count = len(messages)
            _input_types = [m.get("type") or m.get("role", "?") for m in messages[:10]]
            open("/scratch/hc3337/projects/BrowseComp-Plus/.cursor/debug-f24dbd.log", "a").write(
                json.dumps({"id":"pre_call","timestamp":int(time.time()*1000),"location":"oss_client.py:156","message":"Before responses.create","data":{"iteration":iteration,"model":request.get("model"),"input_count":_input_count,"input_types":_input_types,"payload_bytes":_payload_bytes,"has_reasoning":"reasoning" in request},"hypothesisId":"H1_H2_H5"}) + "\n"
            )
            # #endregion
            response = client.responses.create(**request)
        except Exception as e:
            # #region agent log
            _err_str = str(e)
            _err_type = type(e).__name__
            _body = getattr(e, "body", None) or getattr(e, "response", None)
            if hasattr(e, "response") and e.response is not None:
                try:
                    _body = e.response.json() if hasattr(e.response, "json") else str(e.response)
                except Exception:
                    _body = str(getattr(e.response, "text", ""))
            open("/scratch/hc3337/projects/BrowseComp-Plus/.cursor/debug-f24dbd.log", "a").write(
                json.dumps({"id":"err_400","timestamp":int(time.time()*1000),"location":"oss_client.py:165","message":"400/API error caught","data":{"iteration":iteration,"model":request.get("model"),"err_type":_err_type,"err_str":_err_str[:500],"response_body":str(_body)[:800] if _body else None},"hypothesisId":"H5"}) + "\n"
            )
            # #endregion
            if verbose:
                print(f"Error: {e}")
                rprint(f"Request: {request}")
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

        if (not function_calls) or ("content" in response_dict["output"][-1] and (response_dict["output"][-1]["content"][0]["type"] == "output_text")):
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
                error_msg = f"Error executing {tool_call['name']}: {str(e)}"
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": error_msg,
                    }
                )

        total_tools = sum(tool_usage.values())
        if (
            planning_config
            and planning_config.get("trigger") == "after_steps"
            and total_tools >= planning_config.get("steps", 5)
            and not mid_planning_done
        ):
            if verbose:
                print("Mid-conversation planning triggered...")
            planner_response = call_planner_mid(
                client,
                planning_model=planning_config["planning_model"],
                planning_url=planning_config["planning_url"],
                raw_question=planning_config["raw_question"],
                messages=new_messages,
                plan_mid_system_prompt=planning_config["plan_mid_system_prompt"],
                verbose=planning_config.get("verbose", False),
            )
            plan_messages = _inject_plan_into_messages(planner_response)
            new_messages.extend(plan_messages)
            mid_planning_done = True
            if verbose:
                print("Injected revised plan into messages")

        # --- Plan re-injection: periodic lightweight reminder ---
        reinject_every = (planning_config or {}).get("reinject_every", 0)
        plan_text = (planning_config or {}).get("plan_text", "")
        if (
            reinject_every > 0
            and plan_text
            and iteration - last_reinject_at >= reinject_every
        ):
            new_messages.append({
                "role": "user",
                "content": f"Reminder — here is the plan you were given at the start. "
                           f"Use it to guide your next steps:\n{plan_text}",
            })
            last_reinject_at = iteration
            if verbose:
                print(f"Re-injected plan reminder at iteration {iteration}")

        messages = new_messages
        iteration += 1

    return messages, tool_usage, "incomplete"


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
    base_sleep_time = 1
    messages = [
        {"role": "system", "content": plan_system_prompt},
        {"role": "user", "content": raw_question},
    ]

    if str(getattr(client, "base_url", "") or "").rstrip("/") == planning_url.rstrip(
        "/"
    ):
        planner_client = client
    else:
        planner_client = openai.OpenAI(base_url=planning_url, api_key="EMPTY")

    for attempt in range(max_tries):
        try:
            chat_response = planner_client.chat.completions.create(
                model=planning_model,
                messages=messages,
                max_tokens=4096,
            )
            content = chat_response.choices[0].message.content
            if content and content.strip():
                if verbose:
                    print("Planning call successful")
                return content.strip()
            if verbose:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")
        except Exception as e:
            if verbose:
                print(f"Planning error (attempt {attempt + 1}/{max_tries}): {e}")

        if attempt < max_tries - 1:
            sleep_time = min(
                base_sleep_time * (2**attempt) + random.uniform(0, 1), 30
            )
            if verbose:
                print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Planning: all retries exhausted, returning empty plan")
            return ""

    return ""


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
                parts.append(f"[User]: {content[:CONTENT_TRUNCATE]}{'...' if len(content) > CONTENT_TRUNCATE else ''}")
            else:
                parts.append("[User]: (structured content)")
        elif role == "assistant":
            content = item.get("content", "")
            if isinstance(content, str):
                parts.append(f"[Assistant]: {content[:CONTENT_TRUNCATE]}{'...' if len(content) > CONTENT_TRUNCATE else ''}")
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
                    if isinstance(part, dict) and part.get("type") in {"reasoning_text", "output_text", "text"}:
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
                parts.append(f"[Message]: {text[:CONTENT_TRUNCATE]}{'...' if len(text) > CONTENT_TRUNCATE else ''}")
    return "\n\n".join(parts) if parts else "(no history)"


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
) -> str:
    """Call the planner with conversation history to produce a revised plan."""
    conv_str = _serialize_messages_for_planner(messages)
    
    print(f"Conversation history so far: ------------------------------------------\n\n{conv_str}")
    
    user_content = f"""## Original question
{raw_question}

## Conversation history so far
{conv_str}

Based on the question and conversation history above, output the revised plan within <plan></plan> tags."""
    planner_messages = [
        {"role": "system", "content": plan_mid_system_prompt},
        {"role": "user", "content": user_content},
    ]

    if str(getattr(client, "base_url", "") or "").rstrip("/") == planning_url.rstrip("/"):
        planner_client = client
    else:
        planner_client = openai.OpenAI(base_url=planning_url, api_key="EMPTY")

    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            chat_response = planner_client.chat.completions.create(
                model=planning_model,
                messages=planner_messages,
                max_tokens=4096,
            )
            content = chat_response.choices[0].message.content
            if content and content.strip():
                if verbose:
                    print("Mid-conversation planning call successful")
                return content.strip()
            if verbose:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")
        except Exception as e:
            if verbose:
                print(f"Mid-planning error (attempt {attempt + 1}/{max_tries}): {e}")

        if attempt < max_tries - 1:
            sleep_time = min(base_sleep_time * (2**attempt) + random.uniform(0, 1), 30)
            if verbose:
                print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Mid-planning: all retries exhausted, returning empty plan")
            return ""

    return ""


def _load_and_validate_plans(
    plan_file: str, query_tuples: list[tuple[str, str]]
) -> dict[str, str]:
    """Load pre-generated plans from JSONL and validate against query list."""
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
        raise ValueError("Query IDs do not match between plan file and query file: " + "; ".join(parts))

    return plan_dict


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

    if len(traj_dict) != len(query_tuples):
        raise ValueError(
            f"Trajectory directory has {len(traj_dict)} entries but query file has {len(query_tuples)}; "
            "lengths must match"
        )
    if set(traj_dict.keys()) != query_ids:
        missing = query_ids - set(traj_dict.keys())
        extra = set(traj_dict.keys()) - query_ids
        parts = []
        if missing:
            parts.append(f"query IDs missing from trajectory directory: {sorted(missing)}")
        if extra:
            parts.append(f"trajectory directory has extra IDs not in queries: {sorted(extra)}")
        raise ValueError("Query IDs do not match between trajectory directory and query file: " + "; ".join(parts))

    return traj_dict


def _format_trajectory_for_prompt(trajectory: dict) -> str:
    """Serialize a trajectory's result steps into readable text for prompt prepending."""
    parts = []
    for step in trajectory.get("result", []):
        stype = step.get("type")
        if stype == "reasoning":
            output = step.get("output", [])
            text = " ".join(str(o) for o in output) if isinstance(output, list) else str(output)
            if text.strip():
                parts.append(f"[Reasoning]: {text.strip()}")
        elif stype == "tool_call":
            tool_name = step.get("tool_name", "?")
            args = step.get("arguments", "{}")
            try:
                args_dict = json.loads(args) if isinstance(args, str) else args
                q = args_dict.get("user_query") or args_dict.get("query") or str(args)[:100]
            except (json.JSONDecodeError, AttributeError):
                q = str(args)[:100]
            parts.append(f"[Tool call] {tool_name}: {q}")
            output = step.get("output")
            if output is not None:
                out_str = json.dumps(output) if not isinstance(output, str) else output
                parts.append(f"[Tool result]: {out_str[:500]}{'...' if len(out_str) > 500 else ''}")
        elif stype == "output_text":
            text = str(step.get("output", "")).strip()
            if text:
                parts.append(f"[Final answer]: {text}")

    return "\n\n".join(parts) if parts else "(no trajectory steps)"


def _parse_plan_from_response(response: str) -> str:
    """Extract plan content from planner response. Fallback to full response or empty."""
    if "<plan>" in response and "</plan>" in response:
        try:
            return response.split("<plan>")[1].split("</plan>")[0].strip()
        except IndexError:
            pass
    return response.strip() if response else ""


def _inject_plan_into_messages(planner_response: str) -> list[dict]:
    """Return the two messages to inject: assistant intro and user with plan."""
    plan_content = _parse_plan_from_response(planner_response)
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


def _persist_response(
    out_dir: str,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
    planning: bool = False,
    planning_trigger: str | None = None,
    planning_steps: int | None = None,
    planning_file: str | None = None,
    trajectory_dir: str | None = None,
    plan_reinject_every: int = 0,
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
            parts = item.get("content", []) or []
            text_chunks: list[str] = []
            for part in parts:
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
            # Role-based messages (e.g. planning intro, plan injection from user/assistant)
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

    metadata = {
        "model": initial_request.get("model"),
        "reasoning": initial_request.get("reasoning"),
        "output_dir": str(out_dir),
        "planning": planning,
    }
    if planning and planning_trigger:
        metadata["planning_trigger"] = planning_trigger
    if (
        planning
        and planning_trigger in ("after_steps", "start_and_after_steps")
        and planning_steps is not None
    ):
        metadata["planning_steps"] = planning_steps
    if planning and planning_trigger == "start_ext" and planning_file:
        metadata["planning_file"] = planning_file
    if planning and planning_trigger == "traj_ext" and trajectory_dir:
        metadata["trajectory_dir"] = trajectory_dir
    if plan_reinject_every > 0:
        metadata["plan_reinject_every"] = plan_reinject_every

    normalized_record = {
        "metadata": metadata,
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""

    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
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
        plans_by_id = _load_and_validate_plans(args.planning_file, queries)
        print(f"Loaded {len(plans_by_id)} plans from {args.planning_file}")

    trajectories_by_id: dict[str, dict] = {}
    if args.planning and args.planning_trigger == "traj_ext":
        trajectories_by_id = _load_and_validate_trajectories(args.trajectory_dir, queries)
        print(f"Loaded {len(trajectories_by_id)} trajectories from {args.trajectory_dir}")

    import threading

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        print('-------------Handling single query-------------------')
        """Build request, send and persist response for one query."""
        if args.planning and args.planning_trigger == "traj_ext":
            traj = trajectories_by_id.get(qid, {})
            traj_text = _format_trajectory_for_prompt(traj)
            input_messages = [
                {
                    "role": "user",
                    "content": format_query_with_trajectory(qtext, traj_text, args.query_template)
                }
            ]
            if args.verbose:
                print(f"[{qid}] Prepended external trajectory to prompt", flush=True)
        else:
            input_messages = [
                {"role": "user", "content": format_query(qtext, args.query_template)}
            ]

        planning_config = None
        injected_plan_text = ""  # track the plan text for re-injection
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
                plan_messages = _inject_plan_into_messages(planner_response)
                input_messages.extend(plan_messages)
                injected_plan_text = _parse_plan_from_response(planner_response)
                if args.verbose:
                    print(f"[{qid}] Injected plan into messages", flush=True)
            elif args.planning_trigger == "start_ext":
                plan_output = plans_by_id.get(qid, "")
                plan_messages = _inject_plan_into_messages(plan_output)
                input_messages.extend(plan_messages)
                injected_plan_text = _parse_plan_from_response(plan_output)
                if args.verbose:
                    print(f"[{qid}] Injected external plan into messages", flush=True)
            
            if args.planning_trigger in ("after_steps", "start_and_after_steps"):
                planning_config = {
                    "trigger": "after_steps",
                    "steps": args.planning_steps,
                    "raw_question": qtext,
                    "planning_model": args.planning_model,
                    "planning_url": args.planning_url,
                    "plan_mid_system_prompt": args.plan_mid_system_prompt,
                    "verbose": args.verbose,
                }
            # Set up re-injection config if enabled
            if args.plan_reinject_every > 0 and injected_plan_text:
                if planning_config is None:
                    planning_config = {}
                planning_config["reinject_every"] = args.plan_reinject_every
                planning_config["plan_text"] = injected_plan_text


        print('-------------Getting initial request-------------------')
        print("input_messages", input_messages)
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
                client, initial_request, tool_handler, args.max_iterations, args.verbose, planning_config=planning_config,
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
                query_id=qid,
                planning=args.planning,
                planning_trigger=args.planning_trigger,
                planning_steps=(
                    args.planning_steps
                    if args.planning_trigger in ("after_steps", "start_and_after_steps")
                    else None
                ),
                planning_file=(
                    args.planning_file
                    if args.planning_trigger == "start_ext"
                    else None
                ),
                trajectory_dir=(
                    args.trajectory_dir
                    if args.planning_trigger == "traj_ext"
                    else None
                ),
                plan_reinject_every=args.plan_reinject_every,
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


def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Responses API with native function calling and local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query *or* TSV file path",
    )
    parser.add_argument(
        "--model", default="openai/gpt-oss-20b", help="Model name served by vLLM"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="max_output_tokens for Responses API",
    )
    parser.add_argument(
        "--output-dir", default="runs_vllm", help="Directory to store run JSON files"
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
            "QUERY_TEMPLATE_FOR_PLANNING",
            "QUERY_TEMPLATE_GIVEN_TRAJECTORY",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Parallel threads for dataset mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--model-url", default="http://localhost:8000/v1", help="Model URL"
    )
    parser.add_argument(
        "--planning",
        action="store_true",
        help="Enable planning: call planner model first, inject plan into conversation",
    )
    parser.add_argument(
        "--planning-model",
        type=str,
        default=None,
        help="Model for planner (default: same as --model)",
    )
    parser.add_argument(
        "--planning-url",
        type=str,
        default=None,
        help="Base URL for planner (default: same as --model-url)",
    )
    parser.add_argument(
        "--planning-trigger",
        choices=["start", "after_steps", "start_and_after_steps", "start_ext", "traj_ext"],
        default="start",
        help="When to run planning: start (before loop), after_steps (mid-conversation), start_and_after_steps (both), start_ext (load plan from file), or traj_ext (prepend external trajectory to prompt)",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=5,
        help="Number of tool calls before mid-conversation planning (when --planning-trigger=after_steps or start_and_after_steps)",
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
        help="Directory of JSON trajectory files to prepend to the prompt (required when --planning-trigger=traj_ext)",
    )
    parser.add_argument(
        "--plan-reinject-every",
        type=int,
        default=0,
        help="Re-inject the plan as a reminder every N iterations (0 = disabled, e.g. 5 or 10)",
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

    # Searcher selection and shared tool options --------------------------
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models",
    )
    parser.add_argument(
        "--hf-home", type=str, help="Hugging Face home directory for caching"
    )

    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k search results to return for each query"
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.planning and args.planning_trigger == "start_ext":
        if args.planning_file is None:
            print("Error: start_ext requires --planning-file", file=sys.stderr)
            sys.exit(1)

    if args.planning and args.planning_trigger == "traj_ext":
        if args.trajectory_dir is None:
            print("Error: traj_ext requires --trajectory-file", file=sys.stderr)
            sys.exit(1)
        if args.query_template not in (None, "QUERY_TEMPLATE_GIVEN_TRAJECTORY"):
            print(
                f"Error: traj_ext requires --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY "
                f"(got {args.query_template!r})",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.planning_model is None:
        args.planning_model = args.model
    if args.planning_url is None:
        args.planning_url = args.model_url

    repo_root = Path(__file__).resolve().parent.parent
    args.plan_system_prompt = None
    if args.planning and args.planning_trigger in (
        "start",
        "start_and_after_steps",
    ):
        plan_path = Path(args.plan_prompt_file)
        if not plan_path.is_absolute():
            plan_path = repo_root / plan_path
        args.plan_system_prompt = plan_path.read_text(encoding="utf-8")

    args.plan_mid_system_prompt = None
    if args.planning and args.planning_trigger in (
        "after_steps",
        "start_and_after_steps",
    ):
        mid_path = Path(args.plan_prompt_mid_file)
        if not mid_path.is_absolute():
            mid_path = repo_root / mid_path
        args.plan_mid_system_prompt = mid_path.read_text(encoding="utf-8")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key="EMPTY",
    )

    searcher = searcher_class(args)

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    if isinstance(args.query, str):
        qstr = args.query.strip()
        print('query', qstr)
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    _process_tsv_dataset(
                        str(potential_path), client, args, tool_handler
                    )
                    return
            except OSError:
                pass

        if args.planning and args.planning_trigger in ("start_ext", "traj_ext"):
            print(
                f"Error: {args.planning_trigger} requires a TSV query file; use --query path/to/queries.tsv",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Processing single query", args.query)

    raw_question = args.query
    args.query = format_query(args.query, args.query_template)

    messages = [{"role": "user", "content": args.query}]

    planning_config = None
    injected_plan_text = ""
    if args.planning:
        if args.planning_trigger in ("start", "start_and_after_steps"):
            print("Planning mode (start), calling planner...", flush=True)
            planner_response = call_planner(
                client,
                args.planning_model,
                args.planning_url,
                raw_question=raw_question,
                plan_system_prompt=args.plan_system_prompt,
                verbose=args.verbose,
            )
            plan_messages = _inject_plan_into_messages(planner_response)
            messages.extend(plan_messages)
            injected_plan_text = _parse_plan_from_response(planner_response)
        if args.planning_trigger in ("after_steps", "start_and_after_steps"):
            planning_config = {
                "trigger": "after_steps",
                "steps": args.planning_steps,
                "raw_question": raw_question,
                "planning_model": args.planning_model,
                "planning_url": args.planning_url,
                "plan_mid_system_prompt": args.plan_mid_system_prompt,
                "verbose": args.verbose,
            }
        if args.planning_trigger == "traj_ext":
            messages= [{
                "role": "user",
                "content": format_query_for_planning(raw_question, injected_plan_text, args.query_template)
            }]
        if args.plan_reinject_every > 0 and injected_plan_text:
            if planning_config is None:
                planning_config = {}
            planning_config["reinject_every"] = args.plan_reinject_every
            planning_config["plan_text"] = injected_plan_text

    initial_request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": messages,
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
    }

    messages, tool_usage, status = run_conversation_with_tools(
        client, initial_request, tool_handler, args.max_iterations, args.verbose, planning_config=planning_config
    )

    _persist_response(
        args.output_dir, initial_request, messages, tool_usage, status, query_id=None, planning=args.planning,
        planning_trigger=args.planning_trigger,
        planning_steps=(
            args.planning_steps
            if args.planning_trigger in ("after_steps", "start_and_after_steps")
            else None
        ),
        trajectory_dir=(
            args.trajectory_dir
            if args.planning_trigger == "traj_ext"
            else None
        ),
        plan_reinject_every=args.plan_reinject_every,
    )

    rprint(messages)


if __name__ == "__main__":
    main()
