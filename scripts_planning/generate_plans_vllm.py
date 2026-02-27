#!/usr/bin/env python3
"""
Generate multiple plans for questions using vLLM inference with different language models.

Loads questions from a TSV file (query_id\\tquery_text), runs the planner prompt
(following react_agent.py) for each question with each specified model, extracts
the plan from <plan></plan> tags, and saves to an output TSV with an additional
plan column.

Usage:
  python scripts_planning/generate_plans_vllm.py \\
    --input topics-qrels/queries.tsv \\
    --output plans_output.tsv \\
    --model Qwen/Qwen3-8B-Instruct Qwen/Qwen3-32B
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_agent.tongyi_utils.prompts import PROMPT_PLANNER


def load_queries_tsv(tsv_path: Path) -> list[tuple[list[str], str]]:
    """
    Load queries from TSV. Returns list of (row_columns, question) where question
    is the last column (or column 1 if only 2 cols). Assumes format: query_id\\tquery_text
    or query_id\\t...\\tquery_text. The question is typically the second column.
    """
    rows = []
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            # First col = query_id, second col = question (per topics-qrels/queries.tsv)
            query_id = row[0].strip()
            question = row[1].strip() if len(row) > 1 else ""
            if not question:
                continue
            rows.append((row, question))
    return rows


def extract_plan_from_response(content: str) -> str:
    """Extract plan text from <plan>...</plan> tags. Returns empty string if not found."""
    if not content or not content.strip():
        return ""
    content = content.strip()
    match = re.search(r"<plan>\s*(.*?)\s*</plan>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content  # fallback: return raw content if no tags


def run_planning(
    llm: LLM,
    sampling_params: SamplingParams,
    system_prompt: str,
    question: str,
) -> str:
    """Run planner inference and return extracted plan."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        # Some models (e.g. Qwen3) support enable_thinking; others may not
        try:
            outputs = llm.chat(
                [messages],
                sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            outputs = llm.chat([messages], sampling_params)
    except Exception as e:
        return f"[ERROR: {e}]"
    if not outputs or not getattr(outputs[0], "outputs", None):
        return "[ERROR: No output]"
    text = outputs[0].outputs[0].text or ""
    return extract_plan_from_response(text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plans for questions using vLLM with different models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input TSV (query_id\\tquery_text or similar)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output TSV (input columns + model + plan)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        nargs="+",
        required=True,
        help="Model name(s) for vLLM (e.g. Qwen/Qwen3-8B-Instruct)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (for debugging)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    queries = load_queries_tsv(input_path)
    if args.limit:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} queries from {input_path}")

    # Planner prompt (same as react_agent.py)
    system_prompt = PROMPT_PLANNER

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)

        for model_name in args.model:
            print(f"\nLoading model: {model_name}")
            llm = LLM(
                model=model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=8192,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )

            for row, question in tqdm(queries, desc=f"Planning ({model_name})", unit="q"):
                plan = run_planning(llm, sampling_params, system_prompt, question)
                # Output: original columns + model + plan
                out_row = list(row) + [model_name, plan]
                writer.writerow(out_row)

            del llm  # free GPU before loading next model

    print(f"Saved plans to {output_path}")


if __name__ == "__main__":
    main()
