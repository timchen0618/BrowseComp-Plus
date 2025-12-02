#!/usr/bin/env python3
"""
Load trajectory files from data/decrypted_run_files directory.

This script provides utilities to load and explore trajectory files (JSONL format)
stored in the data/decrypted_run_files directory structure.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from collections import defaultdict
import csv
# from nltk import word_tokenize

def write_json(data: Dict[str, Any], file_path: Path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_tsv(data: List[Dict[str, Any]], file_path: Path):
    with open(file_path, 'w', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t',escapechar='\\')
        writer.writerows(data)

def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of JSON objects.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line in the JSONL file
    """
    trajectories = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    trajectories.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}", 
                          file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error: Failed to read {file_path}: {e}", file=sys.stderr)
    
    return trajectories


def find_trajectory_files(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all trajectory files in the data directory.
    
    Args:
        data_dir: Path to data/decrypted_run_files directory
        
    Returns:
        Dictionary mapping subdirectory names to lists of trajectory file paths
    """
    trajectory_files = defaultdict(list)
    
    if not data_dir.exists():
        print(f"Warning: Directory {data_dir} does not exist", file=sys.stderr)
        return trajectory_files
    
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            for jsonl_file in sorted(subdir.glob("*.jsonl")):
                trajectory_files[subdir.name].append(jsonl_file)
    
    return trajectory_files


def load_trajectory_file(
    file_path: Path,
    retriever: Optional[str] = None,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load trajectories from a single file.
    
    Args:
        file_path: Path to the trajectory JSONL file
        retriever: Optional retriever name (e.g., 'bm25', 'qwen3-embed-8b')
        model: Optional model name (e.g., 'gpt5', 'o3', 'oss-20b')
        
    Returns:
        List of trajectory dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")
    
    trajectories = read_jsonl(file_path)
    
    # Add metadata if provided
    if retriever or model:
        for traj in trajectories:
            if retriever:
                traj['_metadata'] = traj.get('_metadata', {})
                traj['_metadata']['retriever'] = retriever
            if model:
                traj['_metadata'] = traj.get('_metadata', {})
                traj['_metadata']['model'] = model
    
    return trajectories


def load_all_trajectories(
    data_dir: Path,
    retriever: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all trajectory files from the data directory.
    
    Args:
        data_dir: Path to data/decrypted_run_files directory
        retriever: Optional filter to load only specific retriever (e.g., 'bm25')
        model: Optional filter to load only specific model (e.g., 'gpt5')
        
    Returns:
        Dictionary mapping file paths to lists of trajectories
    """
    trajectory_files = find_trajectory_files(data_dir)
    all_trajectories = {}
    
    for retriever_name, files in trajectory_files.items():
        # Filter by retriever if specified
        if retriever and retriever_name != retriever:
            continue
            
        for file_path in files:
            # Extract model name from filename (e.g., 'gpt5.jsonl' -> 'gpt5')
            model_name = file_path.stem
            
            # Filter by model if specified
            if model and model_name != model:
                continue
            
            # Load trajectories
            trajectories = load_trajectory_file(file_path, retriever_name, model_name)
            all_trajectories[str(file_path)] = trajectories
    
    return all_trajectories


def load_trajectories_by_retriever(
    data_dir: Path,
    retriever: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all trajectory files for a specific retriever.
    
    Args:
        data_dir: Path to data/decrypted_run_files directory
        retriever: Retriever name (e.g., 'bm25', 'qwen3-embed-8b')
        
    Returns:
        Dictionary mapping model names to lists of trajectories
    """
    trajectory_files = find_trajectory_files(data_dir)
    
    if retriever not in trajectory_files:
        print(f"Warning: No files found for retriever '{retriever}'", file=sys.stderr)
        return {}
    
    trajectories_by_model = {}
    for file_path in trajectory_files[retriever]:
        model_name = file_path.stem
        trajectories = load_trajectory_file(file_path, retriever, model_name)
        trajectories_by_model[model_name] = trajectories
    
    return trajectories_by_model


def extract_docids_from_search_output(output: Any) -> List[str]:
    """
    Extract docids from search tool output.
    
    Args:
        output: Search tool output (can be list, dict, or string)
        
    Returns:
        List of docid strings
    """
    docids_set = set()
    
    if output is None:
        return []
    
    # Try to parse if output is a string
    parsed = None
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except Exception:
            parsed = None
    elif isinstance(output, (list, dict)):
        parsed = output
    
    # Extract docids from parsed structure
    if isinstance(parsed, list):
        for elem in parsed:
            if isinstance(elem, dict) and "docid" in elem:
                docids_set.add(str(elem["docid"]))
    elif isinstance(parsed, dict) and "docid" in parsed:
        docids_set.add(str(parsed["docid"]))
    
    # Fallback: regex grep docids from raw string output
    if isinstance(output, str):
        # Quoted docid values
        for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', output):
            docids_set.add(str(m))
        # Unquoted numeric docid values
        for m in re.findall(r'"docid"\s*:\s*(\d+)', output):
            docids_set.add(str(m))
        
        # <docid: 59490> there is a space between <docid: and 59490>
        for m in re.findall(r'<docid:\s*(\d+)>', output):
            docids_set.add(str(m))

    return sorted(list(docids_set))


def extract_search_queries_with_docids(trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each trajectory, collect the list of search queries and their corresponding doc_ids.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        List of trajectory dictionaries, each with a 'search_queries' field containing
        a list of dictionaries with 'search_query' and 'doc_ids' fields
    """
    results = []
    
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
        
        results.append(trajectory_result)
    return results


def print_search_results(search_results: List[Dict[str, Any]], max_results: Optional[int] = None):
    """
    Print search results in a formatted way.
    
    Args:
        search_results: List of search result dictionaries
        max_results: Maximum number of results to print (None for all)
    """
    if not search_results:
        print("No search results found.")
        return
    
    total = len(search_results)
    if max_results:
        search_results = search_results[:max_results]
        print(f"Showing {len(search_results)} of {total} search results:\n")
    else:
        print(f"Found {total} search results:\n")
    
    for i, result in enumerate(search_results, 1):
        print("=" * 80)
        print(f"Search Result #{i}")
        print("=" * 80)
        if 'source_file' in result:
            print(f"Source File: {result.get('source_file')}")
        print(f"Query ID: {result.get('query_id', 'N/A')}")
        print(f"Search Index: {result.get('search_index', 'N/A')}")
        
        search_query = result.get('search_query')
        if search_query:
            print(f"\nSearch Query: {search_query}")
        else:
            print("\nSearch Query: (not found in arguments)")
        
        output = result.get('search_output')
        if output is not None:
            print("\nSearch Results:")
            print("-" * 80)
            
            # Format output based on its type
            if isinstance(output, list):
                if len(output) == 0:
                    print("(Empty list)")
                elif isinstance(output[0], dict):
                    # List of dictionaries (likely document results)
                    for j, item in enumerate(output, 1):
                        print(f"\n  Result {j}:")
                        if isinstance(item, dict):
                            # Pretty print dictionary
                            formatted = json.dumps(item, indent=4, ensure_ascii=False)
                            for line in formatted.split('\n'):
                                print(f"    {line}")
                        else:
                            print(f"    {item}")
                else:
                    # List of strings or other types
                    for j, item in enumerate(output, 1):
                        print(f"  {j}. {item}")
            elif isinstance(output, str):
                # String output - check if it's JSON
                try:
                    parsed = json.loads(output)
                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                    print(formatted)
                except:
                    print(output)
            else:
                # Other types - convert to JSON
                formatted = json.dumps(output, indent=2, ensure_ascii=False)
                print(formatted)
        else:
            print("\nSearch Results: (No output)")
        
        print()


def get_trajectory_stats(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a list of trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not trajectories:
        return {
            'count': 0,
            'query_ids': [],
            'statuses': {},
        }
    
    query_ids = [traj.get('query_id') for traj in trajectories if 'query_id' in traj]
    statuses = defaultdict(int)
    for traj in trajectories:
        status = traj.get('status', 'unknown')
        statuses[status] += 1
    
    return {
        'count': len(trajectories),
        'query_ids': query_ids,
        'unique_query_ids': len(set(query_ids)),
        'statuses': dict(statuses),
    }


########################################################
# Similarity Functions
########################################################

def word_tokenize(query: str) -> List[str]:
    """
    Tokenize a string into a list of words.
    """
    return query.split()

def jaccard_similarity(query_1: str, query_2: str) -> float:
    """
    Compute the Jaccard similarity between two strings.
    """
    return len(set(word_tokenize(query_1)) & set(word_tokenize(query_2))) / len(set(word_tokenize(query_1)) | set(word_tokenize(query_2)))

def compute_pairwise_similarity(query_1: str, query_2: str, similarity_function: str) -> float:
    """
    Compute the pairwise similarity between two strings.
    """
    if similarity_function == 'cosine':
        return cosine_similarity(query_1, query_2)
    elif similarity_function == 'jaccard':
        return jaccard_similarity(query_1, query_2)
    elif similarity_function == 'levenshtein':
        return levenshtein_similarity(query_1, query_2)
    else:
        raise ValueError(f"Invalid similarity function: {similarity_function}")
    
def doc_id_overlap(doc_ids_1: List[str], doc_ids_2: List[str]) -> float:
    """
    Compute the overlap between two lists of docids.
    """
    # return len(set(doc_ids_1) & set(doc_ids_2)) / len(set(doc_ids_1) | set(doc_ids_2))
    return len(set(doc_ids_1) & set(doc_ids_2)) / max(len(set(doc_ids_1)), len(set(doc_ids_2)))


def cluster_queries(queries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Cluster a list of queries into a list of clusters.
    """
    unique_list = []
    for query in queries:
        if len(unique_list) == 0:
            unique_list.append([query])
        else:
            found_cluster = False
            for _cluster in unique_list:
                for candidate_query in _cluster:
                    if compute_pairwise_similarity(query['search_query'], candidate_query['search_query'], 'jaccard') > 0.4:
                        _cluster.append(query)
                        found_cluster = True
                        break
                if found_cluster:
                    break
            if not found_cluster:
                unique_list.append([query])
    return unique_list

def identify_gold_subqueries(queries: List[Dict[str, Any]], gold_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify gold subtasks from a list of queries.
    """
    gold_subqueries = []
    non_gold_subqueries = []
    gold_docs = gold_data.get('gold_docs', [])
    gold_indices = [] 
    for query in queries:
        #if any(docid in gold_docs for docid in query.get('doc_ids', [])):
        has_gold_id = False
        for docid in query.get('doc_ids', []):
            if docid in gold_docs:
                gold_subqueries.append(query)
                gold_indices.append(docid)
                has_gold_id = True
                break
        if not has_gold_id:
            non_gold_subqueries.append(query)
    return gold_subqueries, non_gold_subqueries, gold_indices


def main():
    """Example usage of the trajectory loading functions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load trajectory files from data/decrypted_run_files"
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/run_subset"), help="Path to data directory (default: data/run_subset)")
    parser.add_argument("--retriever", type=str, default=None, help="Filter by retriever name (e.g., 'bm25', 'qwen3-embed-8b')")    
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., 'gpt5', 'o3', 'oss-20b')")
    parser.add_argument("--file", type=Path, default=None, help="Load a specific trajectory file")
    parser.add_argument("--stats", action="store_true", help="Print statistics about loaded trajectories")
    parser.add_argument("--list", action="store_true", help="List all available trajectory files")
    parser.add_argument("--search", action="store_true", help="Print search tool results")
    parser.add_argument("--max_search_results", type=int, default=None, help="Maximum number of search results to print (default: all)")
    
    args = parser.parse_args()
    
    # List available files
    if args.list:
        trajectory_files = find_trajectory_files(args.data_dir)
        if not trajectory_files:
            print("No trajectory files found.")
            return
        
        print("Available trajectory files:")
        print("=" * 60)
        for retriever, files in sorted(trajectory_files.items()):
            print(f"\n{retriever}/")
            for file_path in files:
                model_name = file_path.stem
                print(f"  - {model_name}.jsonl")
        return
    
    # Load specific file
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        
        trajectories = load_trajectory_file(args.file)
        print(f"Loaded {len(trajectories)} trajectories from {args.file}")
        
        if args.search:
            # Extract search queries with doc_ids for each trajectory
            search_queries_data = extract_search_queries_with_docids(trajectories)
            
            # Print in JSON format
            print("\nSearch queries with doc_ids for each trajectory:")
            print("=" * 80)
            print(json.dumps(search_queries_data, indent=2, ensure_ascii=False))
        
        if args.stats:
            stats = get_trajectory_stats(trajectories)
            print("\nStatistics:")
            print(json.dumps(stats, indent=2))
        
        return
    
    # Load trajectories based on filters
    all_trajectories = load_all_trajectories(
        args.data_dir,
        retriever=args.retriever,
        model=args.model
    )
    
    if not all_trajectories:
        print("No trajectories found matching the criteria.")
        return
    
    # Print summary
    total_count = sum(len(trajs) for trajs in all_trajectories.values())
    print(f"Loaded {total_count} trajectories from {len(all_trajectories)} file(s):")
    
    for file_path, trajectories in sorted(all_trajectories.items()):
        print(f"  {file_path}: {len(trajectories)} trajectories")
    
    # Print search results if requested
    if args.search:
        raw_data = read_jsonl("data/small.jsonl")
        id2data = {record['query_id']: {"query": record['query'], "gold_docs": [d['docid'] for d in record['gold_docs']]} for record in raw_data}
        filename2queries = {}
        docid2text = {d['docid']: d['text'] for record in raw_data for d in record['gold_docs']}
        print('docid2text', len(docid2text))

        all_trajectory_search_data = []
        for file_path, trajectories in sorted(all_trajectories.items()):
            print(file_path)
            search_queries_data = extract_search_queries_with_docids(trajectories)
            # Add file path to each trajectory result for context
            for traj_data in search_queries_data:
                traj_data['source_file'] = file_path
            all_trajectory_search_data.extend(search_queries_data)
        
        # Print in JSON format
        print("\nSearch queries with doc_ids for each trajectory:")
        print("=" * 80)
        
        # Limit results if requested
        total_count = len(all_trajectory_search_data)
        if args.max_search_results and args.max_search_results > 0:
            all_trajectory_search_data = all_trajectory_search_data[:args.max_search_results]
            print(f"Showing {len(all_trajectory_search_data)} of {total_count} trajectories:\n")
        else:
            print(f"Found {total_count} trajectories:\n")
        
        # print(json.dumps(all_trajectory_search_data, indent=2, ensure_ascii=False))
        
        repeated_queries = []
        filename2trajectories = defaultdict(list)
        for traj in all_trajectory_search_data:
            filename2trajectories[traj['source_file']].append(traj)
        
        for filename, trajectories in filename2trajectories.items():
            filename2queries[filename] = []
            avg_pairwise_similarity = 0.0
            avg_consecutive_similarity = 0.0
            
            avg_pairwise_doc_id_overlap = 0.0
            avg_consecutive_doc_id_overlap = 0.0

            avg_repeat_perc = 0.0
            avg_gold_subqueries = 0.0
            avg_num_queries = 0.0
            for i in range(len(trajectories)):
                p_sim, c_sim, p_doc_id_overlap, c_doc_id_overlap = [], [], [], []
                # compute similarity between pairs of queries for each trajectory
                #print(len(trajectories[i]['search_queries']), filename, trajectories[i]['query_id'])
                queries = trajectories[i]['search_queries']
                #print('xxx', queries)
                gold_subqueries, non_gold_subqueries, gold_doc_indices = identify_gold_subqueries(queries, id2data[trajectories[i]['query_id']])
                for ii, q in enumerate(gold_subqueries):
                    filename2queries[filename].append({"search_query": q['search_query'], "complex_question": id2data[trajectories[i]['query_id']], "label": 1, "query_id": trajectories[i]['query_id'], "gold_doc_text": docid2text[gold_doc_indices[ii]]})
                for q in non_gold_subqueries:    
                    filename2queries[filename].append({"search_query": q['search_query'], "complex_question": id2data[trajectories[i]['query_id']], "label": 0, "query_id": trajectories[i]['query_id'], "gold_doc_text": ""})

                #print(gold_subqueries)
                avg_gold_subqueries += (len(gold_subqueries)/len(queries))
                avg_num_queries += len(queries)

                for j in range(len(queries)-1):
                    for k in range(j+1, len(queries)):
                        query_1 = queries[j]
                        query_2 = queries[k]
                        p_doc_id_overlap.append(doc_id_overlap(query_1['doc_ids'], query_2['doc_ids']))
                        #print(query_1['search_query'], query_2['search_query'], query_1['doc_ids'], query_2['doc_ids'])
                        p_sim.append(compute_pairwise_similarity(query_1['search_query'], query_2['search_query'], 'jaccard'))
                    # compute similarity between consecutive queries for each trajectory
                    # assert len(queries[j+1]['doc_ids']) == len(queries[j]['doc_ids']), (len(queries[j+1]['doc_ids']), len(queries[j]['doc_ids']), filename, trajectories[i]['query_id'], j, queries[j]['doc_ids'], queries)
                    assert len(queries[j]['doc_ids']), (len(queries[j]['doc_ids']), filename, trajectories[i]['query_id'])
                    c_doc_id_overlap.append(doc_id_overlap(queries[j+1]['doc_ids'], queries[j]['doc_ids']))
                    c_sim.append(compute_pairwise_similarity(queries[j+1]['search_query'], queries[j]['search_query'], 'jaccard'))                        
                avg_pairwise_doc_id_overlap += sum(p_doc_id_overlap) / len(p_doc_id_overlap)
                avg_consecutive_doc_id_overlap += sum(c_doc_id_overlap) / len(c_doc_id_overlap)
                avg_pairwise_similarity += sum(p_sim) / len(p_sim)
                avg_consecutive_similarity += sum(c_sim) / len(c_sim)
                clustered_queries = cluster_queries(queries)
                # print(clustered_queries)
                num_duplicates = len(queries) - len(clustered_queries)
                avg_repeat_perc += (num_duplicates/len(queries))
                # print(f"Number of duplicates: {num_duplicates}")
                for _cluster in clustered_queries:
                    if len(_cluster) > 1:
                        _qs = [_query['search_query'].replace('\t', ' ') for _query in _cluster]
                        _qs = [' '.join(q) if isinstance(q, list) else q for q in _qs]
                        # print(_qs)
                        repeated_queries.append(_qs)
            avg_pairwise_doc_id_overlap /= len(trajectories)
            avg_consecutive_doc_id_overlap /= len(trajectories)
            avg_pairwise_similarity /= len(trajectories)
            avg_consecutive_similarity /= len(trajectories)
            avg_repeat_perc /= len(trajectories)
            avg_gold_subqueries /= len(trajectories)
            avg_num_queries /= len(trajectories)
            print('--------------------------------')
            print(filename)
            print(f"Average pairwise doc id overlap: {avg_pairwise_doc_id_overlap:.2f}")
            print(f"Average consecutive doc id overlap: {avg_consecutive_doc_id_overlap:.2f}")
            print(f"Average pairwise similarity: {avg_pairwise_similarity:.2f}")
            print(f"Average consecutive similarity: {avg_consecutive_similarity:.2f}")
            print(f"Average repeat percentage: {avg_repeat_perc:.2f}")
            print(f"Average gold subqueries: {avg_gold_subqueries:.2f}")
            print(f"Average num queries: {avg_num_queries:.2f}")
            print('--------------------------------')
        
        write_json(filename2queries, 'filename2queries.json')
        write_tsv(repeated_queries, 'repeated_queries.tsv')
        return
    
    # Print statistics if requested
    if args.stats:
        print("\nStatistics per file:")
        print("=" * 60)
        for file_path, trajectories in sorted(all_trajectories.items()):
            stats = get_trajectory_stats(trajectories)
            print(f"\n{file_path}:")
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

