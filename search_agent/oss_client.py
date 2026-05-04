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


# --- Main agent loop ---------------------------------------------------------

def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
    verbose: bool = False,
    budget_mode: bool = False,
):
    tool_usage: dict[str, int] = {}
    messages = initial_request["input"]
    iteration = 1      # total calls (always increments)
    budget_calls = 1   # valid-call counter used only in budget_mode

    while True:
        if budget_mode:
            if budget_calls > max_iterations or iteration > 3 * max_iterations:
                break
        else:
            if iteration > max_iterations:
                break

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

        if not response_dict["output"]:
            iteration += 1
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
        has_valid_tool_call = False

        for tool_call in function_calls:
            try:
                arguments = json.loads(tool_call["arguments"])
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1
                has_valid_tool_call = True
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

        messages = new_messages
        iteration += 1
        if budget_mode and has_valid_tool_call:
            budget_calls += 1

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

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        """Build request, send, and persist response for one query."""
        if args.planning_trigger in TRAJ_TRIGGERS:
            user_content = _build_trajectory_user_content(
                qid, qtext, args, client, trajectories_by_id, summaries_by_id
            )
        elif args.search_budget is not None:
            user_content = format_query_with_budget(qtext, args.search_budget)
        else:
            user_content = format_query(qtext, args.query_template)
        input_messages = [{"role": "user", "content": user_content}]

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": input_messages,
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        }

        try:
            effective_max_iter = args.search_budget if args.search_budget is not None else args.max_iterations
            messages, tool_usage, status = run_conversation_with_tools(
                client,
                initial_request,
                tool_handler,
                effective_max_iter,
                args.verbose,
                budget_mode=(args.search_budget is not None),
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
