"""
Load, filter, and convert trajectory run files.

Conversion: trajectory run format -> SFT format (input/output with role-based messages).
Parsing follows search_agent/tongyi_utils/react_agent.py and tongyi_client.parse_messages_to_result_array.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from compute_repeats import extract_docids_from_search_output
import json

# Default system prompt for SFT conversion (from tongyi prompts)
DEFAULT_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call the search tool one or multiple times to assist with the user query.

You are provided with the search tool, its signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts a single query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}
</tools>

For each search tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": search, "arguments": <args-json-object>}
</tool_call>

Current date: """


def read_json(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_trajectories(data_dir: Path) -> List[Dict[str, Any]]:
    # load all json files in the data_dir
    json_files = list(data_dir.glob('*.json'))
    print('num json files', len(json_files))
    trajectories = []
    for json_file in json_files:
        trajectories.append(read_json(json_file))
    return trajectories


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Parse tool arguments from run file format (may be JSON string or dict)."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"query": arguments}
    return {}


def _format_tool_call(tool_name: str, arguments: Any) -> str:
    """Format a tool call as the assistant would produce it (per react_agent)."""
    args_dict = _parse_tool_arguments(arguments)
    tool_call_obj = {"name": tool_name, "arguments": args_dict}
    return f"<tool_call>\n{json.dumps(tool_call_obj, ensure_ascii=False)}\n</tool_call>"


def _format_tool_response(output: str) -> str:
    """Format tool output as the user message (per react_agent)."""
    return f"<tool_response>\n{output}\n</tool_response>"


def convert_trajectory_to_sft(
    run_data: Dict[str, Any],
    query: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a trajectory run file to SFT format.

    Args:
        run_data: JSON object from a run file (e.g. run_*.json from tongyi_client).
            Must have "result" array. Does not contain the query text; use query param.
        query: The user's question (run files only store query_id; pass query from queries.tsv).
        system_prompt: Optional system prompt. If None, uses DEFAULT_SYSTEM_PROMPT.

    Returns:
        Dict with "input" and "output" keys, matching sample_trajectory_data.jsonl format.
        - input: system_prompt + "User: " + query
        - output: list of {"role": "assistant"|"user", "content": "..."} messages
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    result = run_data.get("result", [])
    output_messages: List[Dict[str, str]] = []
    reasoning_buffer: List[str] = []

    def flush_reasoning_as_assistant():
        if reasoning_buffer:
            content = "".join(f"<think>{r}</think>\n" for r in reasoning_buffer)
            output_messages.append({"role": "assistant", "content": content.strip()})
            reasoning_buffer.clear()

    for entry in result:
        entry_type = entry.get("type")
        entry_output = entry.get("output") or ""

        if entry_type == "reasoning":
            reasoning_buffer.append(entry_output)

        elif entry_type == "tool_call":
            tool_name = entry.get("tool_name") or "search"
            arguments = entry.get("arguments")
            tool_call_str = _format_tool_call(tool_name, arguments)
            if reasoning_buffer:
                content = "".join(f"<think>{r}</think>\n" for r in reasoning_buffer) + tool_call_str
                reasoning_buffer.clear()
            else:
                content = tool_call_str
            output_messages.append({"role": "assistant", "content": content})
            output_messages.append({
                "role": "user",
                "content": _format_tool_response(entry_output),
            })

        elif entry_type == "output_text":
            if reasoning_buffer:
                content = "".join(f"<think>{r}</think>\n" for r in reasoning_buffer) + f"<answer>\n{entry_output}\n</answer>"
                reasoning_buffer.clear()
            else:
                content = f"<answer>\n{entry_output}\n</answer>"
            output_messages.append({"role": "assistant", "content": content})

        elif entry_type == "user":
            flush_reasoning_as_assistant()
            output_messages.append({"role": "user", "content": entry_output})

        elif entry_type == "assistant":
            flush_reasoning_as_assistant()
            output_messages.append({"role": "assistant", "content": entry_output})

    flush_reasoning_as_assistant()

    return {
        "input": f"{system_prompt}User: {query}",
        "output": output_messages,
    }


def load_query_by_id(queries_tsv: str, query_id: str) -> Optional[str]:
    """Load query text from a TSV file (query_id\\tquery_text) by query_id."""
    with open(queries_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) >= 2 and parts[0].strip() == str(query_id):
                return parts[1].strip()
    return None


def filter_trajectories(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_trajectories = []
    
    for traj in trajectories:
        query_id = traj.get('query_id')
        result_array = traj.get('result', [])
        
        search_queries_list = []
        entry_id = 0
        for entry in result_array:
            if entry.get('type') == 'tool_call' and (entry.get('tool_name') == 'search' or entry.get('tool_name') == 'local_knowledge_base_retrieval'):
                # Extract search query from arguments
                search_query = None
                arguments = entry.get('arguments')
                
                if isinstance(arguments, dict):
                    search_query = arguments.get('query') or arguments.get('search_query') or arguments.get('user_query')
                elif isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, dict):
                            search_query = parsed_args.get('query') or parsed_args.get('search_query') or parsed_args.get('user_query')
                    except:
                        search_query = arguments
                
                # Extract docids from output
                output = entry.get('output')
                doc_ids = extract_docids_from_search_output(output)
                
                # Add to list if we have a search query
                if search_query is not None and len(doc_ids) > 0:
                    if isinstance(search_query, list):
                        search_query = ' '.join(search_query)
                    search_queries_list.append({
                        "search_query": search_query,
                        "doc_ids": doc_ids
                    })
                entry_id += 1
        
        # Create result entry for this trajectory
        trajectory_result = {
            "query_id": query_id,
            "search_queries": search_queries_list
        }
        
        new_trajectories.append(trajectory_result)
    return new_trajectories


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert trajectory run files to SFT format"
    )
    parser.add_argument("run_path", type=str, help="Path to run JSON files")
    parser.add_argument("--queries-tsv", default="topics-qrels/queries.tsv", help="TSV to look up query by query_id")
    parser.add_argument("--output", "-o", help="Output JSONL file (default: stdout)", default="sft/sft_trajectories.jsonl")
    parser.add_argument("--filter", action="store_true", help="Only convert trajectories that pass the filter (have search calls with docids)")

    args = parser.parse_args()

    out_file = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        for run_path in Path(args.run_path).glob("*.json"):
            with open(run_path, "r", encoding="utf-8") as f:
                run_data = json.load(f)
            if args.filter:
                run_data = filter_trajectories(run_data)

            query_id = run_data.get("query_id")
            query = None
            if query is None and query_id is not None:
                query = load_query_by_id(args.queries_tsv, query_id)
            if query is None:
                raise SystemExit("Error: --query required, or run file must have query_id and --queries-tsv must contain it")
            sft = convert_trajectory_to_sft(run_data, query)
            line = json.dumps(sft, ensure_ascii=False)
            if out_file:
                out_file.write(line + "\n")
            else:
                print(line)
    finally:
        if out_file:
            out_file.close()
            print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()