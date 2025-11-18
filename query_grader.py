#!/usr/bin/env python3
"""
Script to evaluate search queries using an LLM with vLLM.

This script uses the query grader prompt to evaluate how useful a search query is
for answering a complex question.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_system_prompt(prompt_path: Path) -> str:
    """Load the system prompt from the query grader file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_user_prompt(search_query: str, complex_question: str = None, ground_truth_documents: List[str] = None) -> str:
    """
    Format the user prompt with the search query and optional complex question.
    
    Args:
        search_query: The search query to evaluate
        complex_question: Optional complex question context (if not provided, 
                         will use a generic placeholder)
        ground_truth_documents: Optional list of ground truth documents (if not provided, 
                         will not use them to evaluate the query)
    Returns:
        Formatted user prompt string
    """
    if complex_question:
        if ground_truth_documents:
            ground_truth_documents_str = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(ground_truth_documents)])
            return f"complex_question: {complex_question}\n\nsearch_query: {search_query}\n\nground_truth_documents:\n{ground_truth_documents_str}"
        else:
            return f"complex_question: {complex_question}\n\nsearch_query: {search_query}"
    else:
        return f"search_query: {search_query}\nNote: The complex_question context is not provided. Evaluate the search_query based on its general utility and quality."


def load_and_filter_filename2queries(
    file_path: str,
    filter_keys: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Load data from filename2queries.json and filter to specified keys.
    
    Args:
        file_path: Path to the filename2queries.json file
        filter_keys: List of keys (filenames) to keep. If None, keeps all.
    
    Returns:
        List of filtered data items
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path_obj, "r", encoding="utf-8") as f:
        filename2queries = json.load(f)
    
    filtered_data = []
    
    if filter_keys:
        for key in filter_keys:
            if key in filename2queries:
                filtered_data.extend(filename2queries[key])
                print(f"Loaded {len(filename2queries[key])} items from {key}")
            else:
                print(f"Warning: Key '{key}' not found in filename2queries.json")
    else:
        # If no filter specified, return all data
        for key, data_list in filename2queries.items():
            filtered_data.extend(data_list)
            print(f"Loaded {len(data_list)} items from {key}")
    
    print(f"Total filtered items: {len(filtered_data)}")
    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a search query using LLM with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to filename2queries.json file to load data from",
    )
    parser.add_argument(
        "--filter-keys",
        type=str,
        nargs="+",
        default=None,
        help="Keys (filenames) to filter from input file. If not specified, uses default filter keys.",
    )
    parser.add_argument(
        "--complex-question",
        type=str,
        default=None,
        help="Optional complex question context for the search query",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name to use with vLLM",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="prompts/query_grader.txt",
        help="Path to the query grader prompt file",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (-1 to disable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of output tokens",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path to save the result as JSON",
    )
    parser.add_argument(
        "--save-filtered",
        type=str,
        default=None,
        help="Save filtered data to a JSON file (only used with --input-file)",
    )

    parser.add_argument(
        "--gt",
        action="store_true",
        help="Use ground truth documents to evaluate the query",
    )
    args = parser.parse_args()

    # Determine if we're processing from a file or single query
    assert args.input_file
    raw_data = read_jsonl('data/small.jsonl')
    queryid2gt = {item['query_id']: item['gold_docs'] for item in raw_data}
    
    # Default filter keys if not specified
    if args.filter_keys is None:
        filter_keys = [
            'data/run_subset/qwen3-embed-8b/gpt5.jsonl',
            'data/run_subset/qwen3-embed-8b/o3.jsonl'
        ]
    else:
        filter_keys = args.filter_keys
    
    # Load and filter data
    data_items = load_and_filter_filename2queries(args.input_file, filter_keys)
    
    if not data_items:
        print("No data items found after filtering.")
        return
    
    print(f"\nProcessing {len(data_items)} items from filtered data...")
    
    # Print sample data structure
    print("\nSample data structure:")
    if data_items:
        sample = data_items[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    print("\nData loaded and filtered successfully.")
    
    
    # Load system prompt
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    system_prompt = load_system_prompt(prompt_path)
    
    # Initialize vLLM
    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # Prepare all messages in batch
    print(f"\nSystem prompt loaded from: {prompt_path}")
    print(f"Preparing batch of {len(data_items)} items for inference...")
    
    messages_list = []
    for data_item in data_items:
        if args.gt:
            ground_truth_documents = [doc['text'] for doc in queryid2gt[data_item['query_id']]]
            user_prompt = format_user_prompt(
                data_item['search_query'], 
                data_item['complex_question']['query'],
                ground_truth_documents
            )
        else:
            user_prompt = format_user_prompt(
                data_item['search_query'], 
                data_item['complex_question']['query']
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages_list.append(messages)
    
    # Batch inference - single call to vLLM
    print("Running batch inference...")
    outputs = llm.chat(
        messages_list,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    
    # Process all outputs
    print("Processing results...")
    accuracy = 0
    for i, (data_item, output) in enumerate(zip(data_items, outputs)):
        # Extract response text
        if output and hasattr(output, "outputs") and len(output.outputs) > 0:
            response_text = output.outputs[0].text
        else:
            response_text = ""
        
        # Extract score using regex
        try:
            score_match = re.search(r'"score":\s*(\d+)', response_text)
            if score_match:
                score = int(score_match.group(1))
            else:
                # Try alternative patterns if the first one doesn't match
                score_match = re.search(r'"score":\s*"(\d+)"', response_text)
                if score_match:
                    score = int(score_match.group(1))
                else:
                    print(f"Warning: Could not extract score from response {i}")
                    score = 0
        except (AttributeError, ValueError) as e:
            print(f"Warning: Error parsing score from response {i}: {e}")
            score = 0
        
        # Calculate accuracy
        predict_positive = (score >= 6)
        is_correct = (data_item['label'] == predict_positive)
        accuracy += is_correct
        
        # if (i + 1) % 10 == 0 or i == 0:
        #     print(f"Item {i+1}/{len(data_items)}: Score={score}, Label={data_item['label']}, Correct={is_correct}")
    
    print(f"\n{'=' * 80}")
    print(f"Final Accuracy: {accuracy / len(data_items):.4f} ({accuracy}/{len(data_items)})")
    print(f"{'=' * 80}")

    # # Save to file if requested
    # if args.output:
    #     result = {
    #         "search_query": args.search_query,
    #         "complex_question": args.complex_question,
    #         "evaluation": response_text,
    #         "model": args.model,
    #     }
    #     output_path = Path(args.output)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         json.dump(result, f, indent=2, ensure_ascii=False)
    #     print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()

