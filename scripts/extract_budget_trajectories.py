#!/usr/bin/env python3
"""
Extract the first N tool call turns from existing trajectories to simulate budget mode.

Usage:
    python scripts/extract_budget_trajectories.py \
        --src runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
        --dst runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_main_agent_seed0 \
        --budget 5
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


def _normalize_messages_to_result(messages: list) -> list:
    """Replicate the normalization logic from oss_client.py."""
    call_output_by_id: dict = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "function_call":
            call_id = item.get("call_id")
            normalized.append({
                "type": "tool_call",
                "tool_name": item.get("name"),
                "arguments": item.get("arguments"),
                "output": call_output_by_id.get(call_id),
            })
        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and len(summary) > 0:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text", "output_text", "text"
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": reasoning_output,
            })
        elif itype == "message":
            text_chunks = []
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": text,
                })
        elif itype is None and "role" in item and "content" in item:
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                normalized.append({
                    "type": item.get("role", "user"),
                    "tool_name": None,
                    "arguments": None,
                    "output": content,
                })

    return normalized


def _extract_retrieved_docids(result: list) -> list:
    """Replicate extract_retrieved_docids_from_result from search_agent/utils.py."""
    retrieved_set = set()
    for item in result:
        if item.get("type") != "tool_call":
            continue
        tool_name = str(item.get("tool_name") or "")
        if "search" not in tool_name.lower() and "retrieval" not in tool_name.lower():
            continue
        output = item.get("output")
        parsed = None
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except Exception:
                parsed = None
        elif isinstance(output, list):
            parsed = output
        if isinstance(parsed, list):
            for elem in parsed:
                if isinstance(elem, dict) and "docid" in elem:
                    retrieved_set.add(str(elem["docid"]))
        else:
            raw = output if isinstance(output, str) else ""
            if raw:
                for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', raw):
                    retrieved_set.add(m)
                for m in re.findall(r'"docid"\s*:\s*(\d+)', raw):
                    retrieved_set.add(m)
    return list(retrieved_set)


def _truncate_messages(original_messages: list, budget: int) -> tuple[list, bool]:
    """
    Return (truncated_messages, was_truncated).

    Iterates original_messages in order, including everything up to and including
    the budget-th function_call and its immediately following function_call_output.
    Drops all subsequent messages.
    """
    fc_count = 0
    truncated = []
    i = 0

    while i < len(original_messages):
        msg = original_messages[i]

        if not isinstance(msg, dict) or msg.get("type") != "function_call":
            truncated.append(msg)
            i += 1
            continue

        # This is a function_call — include it
        fc_count += 1
        truncated.append(msg)
        i += 1

        # Include the immediately following function_call_output (if present)
        if i < len(original_messages):
            nxt = original_messages[i]
            if isinstance(nxt, dict) and nxt.get("type") == "function_call_output":
                truncated.append(nxt)
                i += 1

        if fc_count >= budget:
            # Drop everything that follows
            was_truncated = i < len(original_messages)
            return truncated, was_truncated

    return truncated, False


def _compute_tool_counts(result: list) -> dict:
    counts: dict = {}
    for item in result:
        if item.get("type") == "tool_call":
            name = str(item.get("tool_name") or "unknown")
            counts[name] = counts.get(name, 0) + 1
    return counts


def process_file(src_path: str, dst_dir: str, budget: int) -> str:
    with open(src_path, encoding="utf-8") as f:
        record = json.load(f)

    original_messages = record.get("original_messages") or []
    original_fc_count = sum(
        1 for m in original_messages
        if isinstance(m, dict) and m.get("type") == "function_call"
    )

    if original_fc_count <= budget:
        # Keep as-is: within budget, preserve full result and original status
        truncated_messages = original_messages
        was_truncated = False
    else:
        truncated_messages, was_truncated = _truncate_messages(original_messages, budget)

    result = _normalize_messages_to_result(truncated_messages)
    tool_counts = _compute_tool_counts(result)
    retrieved_docids = _extract_retrieved_docids(result)
    status = "incomplete" if was_truncated else record.get("status", "completed")

    metadata = dict(record.get("metadata") or {})
    metadata["output_dir"] = dst_dir
    metadata["extracted_from"] = src_path
    metadata["budget"] = budget

    out_record = {
        "metadata": metadata,
        "query_id": record.get("query_id"),
        "tool_call_counts": tool_counts,
        "status": status,
        "retrieved_docids": retrieved_docids,
        "result": result,
        "original_messages": truncated_messages,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    out_path = os.path.join(dst_dir, f"run_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2, default=str)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract first N tool calls from trajectories")
    parser.add_argument("--src", required=True, help="Source trajectory directory")
    parser.add_argument("--dst", required=True, help="Destination directory")
    parser.add_argument("--budget", type=int, default=5, help="Max tool calls to keep (default: 5)")
    args = parser.parse_args()

    src_dir = os.path.expanduser(args.src)
    dst_dir = os.path.expanduser(args.dst)

    if not os.path.isdir(src_dir):
        sys.exit(f"Source directory not found: {src_dir}")

    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not files:
        sys.exit(f"No JSON files found in: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    n_truncated = 0
    n_kept = 0

    for i, src_path in enumerate(files):
        out_path = process_file(src_path, dst_dir, args.budget)
        d = json.load(open(src_path))
        om = d.get("original_messages") or []
        fc = sum(1 for m in om if isinstance(m, dict) and m.get("type") == "function_call")
        if fc > args.budget:
            n_truncated += 1
        else:
            n_kept += 1
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] written → {out_path}")

    print(f"\nDone. {len(files)} files processed.")
    print(f"  Truncated (had >{args.budget} tool calls): {n_truncated}")
    print(f"  Kept as-is (had <={args.budget} tool calls): {n_kept}")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
