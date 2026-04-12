"""Summarize agent trajectories using a vLLM server.

Reads trajectory JSON files and a queries TSV, calls the LLM to produce
a concise summary per trajectory, and writes results to a single JSONL file.
Supports resuming by skipping query IDs already present in the output file.
"""

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

from search_agent.prompts import PROMPT_TRAJECTORY_SUMMARIZER


def _load_trajectories(trajectory_dir: str) -> dict[str, dict]:
    """Load trajectory JSON files from a directory, keyed by query_id."""
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
    return traj_dict


def _load_queries(tsv_path: str) -> list[tuple[str, str]]:
    """Load (query_id, query_text) pairs from a TSV file."""
    queries = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))
    return queries


def _format_trajectory_for_prompt(
    trajectory: dict,
    max_chars: int = 0,
    reasoning_max_chars: int = 0,
    tool_output_max_chars: int = 500,
) -> str:
    """Serialize a trajectory's result steps into readable text."""
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
                q = args_dict.get("user_query") or args_dict.get("query") or str(args)[:100]
            except (json.JSONDecodeError, AttributeError):
                q = str(args)[:100]
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


def _assistant_text_from_message(message) -> str:
    """Prefer message.content; fall back to reasoning_content (vLLM GPT-OSS channel split)."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def call_trajectory_summarizer(
    client: openai.OpenAI,
    model: str,
    question: str,
    trajectory_text: str,
    *,
    max_tries: int = 5,
    verbose: bool = False,
    max_tokens: int = 8192,
) -> str:
    """Call the LLM to summarize a trajectory into a concise description."""
    base_sleep_time = 1
    messages = [
        {"role": "system", "content": PROMPT_TRAJECTORY_SUMMARIZER},
        {"role": "user", "content": f"Question: {question}\n\nTrajectory:\n{trajectory_text}"},
    ]
    # print('trajectory_text:', trajectory_text[:2000])

    last_text = ""
    for attempt in range(max_tries):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            text = _assistant_text_from_message(chat_response.choices[0].message)
            if not text:
                if verbose:
                    print(f"Warning: Summarizer attempt {attempt + 1} received an empty response.")
            elif "<trajectory_summary>" not in text or "</trajectory_summary>" not in text:
                last_text = text
                if verbose:
                    print(f"Warning: Summarizer attempt {attempt + 1} missing <trajectory_summary> tags, retrying...")
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


def _load_completed_ids(output_path: Path) -> set[str]:
    """Load query IDs already present in the output JSONL file."""
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("query_id")
                if qid:
                    completed.add(str(qid))
            except json.JSONDecodeError:
                continue
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Summarize agent trajectories using a vLLM server."
    )
    parser.add_argument(
        "--trajectory-dir", required=True,
        help="Directory of trajectory JSON files",
    )
    parser.add_argument(
        "--queries-tsv", required=True,
        help="TSV file of (query_id, query_text) pairs",
    )
    parser.add_argument(
        "--output-file", required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name for the vLLM server",
    )
    parser.add_argument(
        "--model-url", default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=8,
        help="Number of concurrent threads (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum number of tokens for the trajectory summarizer (default: 8192)",
    )
    args = parser.parse_args()

    queries = _load_queries(args.queries_tsv)
    print(f"Loaded {len(queries)} queries from {args.queries_tsv}")

    trajectories = _load_trajectories(args.trajectory_dir)
    print(f"Loaded {len(trajectories)} trajectories from {args.trajectory_dir}")

    # Validate that all query IDs have trajectories
    query_ids = {qid for qid, _ in queries}
    missing = query_ids - set(trajectories.keys())
    if missing:
        print(f"Warning: {len(missing)} query IDs have no trajectory: {sorted(missing)[:10]}...")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _load_completed_ids(output_path)
    remaining = [(qid, qtext) for qid, qtext in queries if qid not in completed_ids and qid in trajectories]
    print(
        f"Processing {len(remaining)} remaining queries "
        f"(skipping {len(completed_ids)} already done)"
    )

    if not remaining:
        print("Nothing to do.")
        return

    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")

    import threading
    write_lock = threading.Lock()

    def _summarize_one(qid: str, qtext: str) -> dict:
        traj = trajectories[qid]
        traj_text = _format_trajectory_for_prompt(
            traj,
            max_chars=400000,
            reasoning_max_chars=3000,
            tool_output_max_chars=3000,
        )
        if args.verbose:
            print(f"[{qid}] Summarizing trajectory...", flush=True)

        summary = call_trajectory_summarizer(
            client,
            args.model,
            question=qtext,
            trajectory_text=traj_text,
            verbose=args.verbose,
            max_tokens=args.max_tokens,
        )

        if summary:
            print("--------------------------------")
            print(summary)
        record = {"query_id": qid, "query": qtext, "summary": summary}

        with write_lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if args.verbose:
            print(f"[{qid}] Done (summary length: {len(summary)} chars)", flush=True)

        return record

    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(_summarize_one, qid, qtext): qid
            for qid, qtext in remaining
        }
        for future in as_completed(futures):
            qid = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(remaining):
                    print(f"Progress: {completed}/{len(remaining)}", flush=True)
            except Exception as e:
                print(f"Error processing {qid}: {e}", flush=True)

    print(f"Done. {completed} summaries written to {args.output_file}")


if __name__ == "__main__":
    main()

    # python summarize_trajectories.py --trajectory-dir runs/bcp/Qwen3-Embedding-8B/first50/gpt-oss-120b/seed0/ --queries-tsv topics-qrels/bcp/queries_first50.tsv  --output-file trajectory_summaries.jsonl  --model openai/gpt-oss-120b  --model-url http://localhost:8000/v1 --num-threads 1 --verbose 
