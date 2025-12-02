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
from typing import List, Dict, Any, Optional, Iterator, Tuple   
from collections import defaultdict
import csv
from src_utils.load_trajectories import load_all_trajectories, find_trajectory_files, write_json, write_tsv, read_jsonl
# from nltk import word_tokenize

def write_csv(csv_results: List[List[Any]], filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_results)

def load_system_prompt(prompt_path: Path) -> str:
    """Load the system prompt from the query grader file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_user_prompt(search_query_1: str, search_query_2: str) -> str:
    """
    Format the user prompt with two search queries.
    Args:  search_query_1: The first search query  |  search_query_2: The second search query
    Returns:  Formatted user prompt string
    """
    return f"search_query_1: {search_query_1}\nsearch_query_2: {search_query_2}"


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

def llm_judge_similarity(query_pairs: List[Tuple[str, str]], llm, sampling_params, system_prompt) -> List[bool]: 
    """
        Compute the similarity between two queries using the LLM judge.
    """
    messages_list = []
    for query_1, query_2 in query_pairs:
        user_prompt = format_user_prompt(query_1, query_2)
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
    results = []
    print("Processing results...")
    for output in outputs:
        if output and hasattr(output, "outputs") and len(output.outputs) > 0:
            response_text = output.outputs[0].text
        else:
            response_text = ""
        results.append(response_text == "Yes" or response_text == "yes" or response_text == "Yes." or response_text == "yes.")
    return results

def compute_pairwise_similarity(query_1: str, query_2: str, similarity_function: str, llm=None, sampling_params=None, system_prompt=None) -> float:
    """
    Compute the pairwise similarity between two strings.
    """
    if similarity_function == 'cosine':
        return cosine_similarity(query_1, query_2)
    elif similarity_function == 'jaccard':
        return jaccard_similarity(query_1, query_2)
    elif similarity_function == 'llm_judge':
        return llm_judge_similarity(query_1, query_2, llm, sampling_params, system_prompt)
    else:
        raise ValueError(f"Invalid similarity function: {similarity_function}")
    
def doc_id_overlap(doc_ids_1: List[str], doc_ids_2: List[str]) -> float:
    """
    Compute the overlap between two lists of docids.
    """
    # return len(set(doc_ids_1) & set(doc_ids_2)) / len(set(doc_ids_1) | set(doc_ids_2))
    return len(set(doc_ids_1) & set(doc_ids_2)) / max(len(set(doc_ids_1)), len(set(doc_ids_2)))


def cluster_queries(queries: List[Dict[str, Any]], 
                    similarity_threshold: float = 0.4, 
                    similarity_function: str = 'jaccard',
                    llm=None, sampling_params=None, system_prompt=None
                    ) -> List[List[Dict[str, Any]]]:
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
                    # if compute_pairwise_similarity(query['search_query'], candidate_query['search_query'], similarity_function, llm, sampling_params, system_prompt) > similarity_threshold:
                    if llm_judge_similarity([(query['search_query'], candidate_query['search_query'])], llm, sampling_params, system_prompt)[0]:
                        _cluster.append(query)
                        found_cluster = True
                        break
                if found_cluster:
                    break
            if not found_cluster:
                unique_list.append([query])
    return unique_list

def cluster_queries_by_doc_id(queries, similarity_threshold: float = 0.4) -> List[List[Dict[str, Any]]]:
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
                    if compute_pairwise_similarity(query['search_query'], candidate_query['search_query']) > similarity_threshold:
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
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Similarity threshold for clustering queries")
    parser.add_argument("--similarity_function", type=str, default='llm_judge', help="Similarity function for clustering queries")
    parser.add_argument("--output_csv", action='store_true', help="whether to output the results to a csv file.")
    args = parser.parse_args()
    
    ########################################################
    # Loading and Sanity Check
    ########################################################
    # List available files
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
    
    print("=" * 60)
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
    
    
    ########################################################
    # Initialize LLM judge
    ########################################################
    print('Initializing VLLM engine')
    from vllm import LLM, SamplingParams
    # Load system prompt
    system_prompt = "You are a helpful assistant that judges the similarity between two search queries. Respond with only 'Yes or 'No'. If the queries are similar and are likely to return the same search results, response with 'Yes'. Otherwise, respond with 'No'. Do not respond with any other text."
    
    # Initialize vLLM
    print(f"Loading model: {"Qwen/Qwen3-30B-A3B-Instruct-2507"}")
    llm = LLM(model="Qwen/Qwen3-30B-A3B-Instruct-2507", tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.8)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    
    ########################################################
    # Compute Repeats
    ########################################################
    if args.output_csv:
        csv_results = [['filename', '', 'pairwise_doc_id_overlap', 'consecutive_doc_id_overlap', 'repeat_perc_doc_id', '', 'pairwise_similarity', 'consecutive_similarity', 'repeat_perc', 'num_queries', 'gold_subqueries']]
    raw_data = read_jsonl("data/small.jsonl")
    # query id to data
    id2data = {record['query_id']: {"query": record['query'], "gold_docs": [d['docid'] for d in record['gold_docs']]} for record in raw_data}
    filename2queries = {}
    # gold doc id to text
    golddocid2text = {d['docid']: d['text'] for record in raw_data for d in record['gold_docs']}

    all_trajectory_search_data = []
    for file_path, trajectories in sorted(all_trajectories.items()):
        search_queries_data = extract_search_queries_with_docids(trajectories)
        # Add file path to each trajectory result for context
        for traj_data in search_queries_data:
            traj_data['source_file'] = file_path
        all_trajectory_search_data.extend(search_queries_data)
    
    # Print in JSON format
    print("\nSearch queries with doc_ids for each trajectory:")
    print("=" * 80)    
    print(f"Found {total_count} trajectories:\n")
    
    
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
        
        query_pairs = []
        for i in range(len(trajectories)):
            p_doc_id_overlap, c_doc_id_overlap = [], []
            queries = trajectories[i]['search_queries']
            
            # START: identify gold subqueries
            gold_subqueries, non_gold_subqueries, gold_doc_indices = identify_gold_subqueries(queries, id2data[trajectories[i]['query_id']])
            for ii, q in enumerate(gold_subqueries):
                filename2queries[filename].append({"search_query": q['search_query'], "complex_question": id2data[trajectories[i]['query_id']], "label": 1, "query_id": trajectories[i]['query_id'], "gold_doc_text": golddocid2text[gold_doc_indices[ii]]})
            for q in non_gold_subqueries:    
                filename2queries[filename].append({"search_query": q['search_query'], "complex_question": id2data[trajectories[i]['query_id']], "label": 0, "query_id": trajectories[i]['query_id'], "gold_doc_text": ""})

            avg_gold_subqueries += (len(gold_subqueries)/len(queries))
            avg_num_queries += len(queries)
            # END: identify gold subqueries
            
            # append query pairs to list
            for j in range(len(queries)-1):
                for k in range(j+1, len(queries)):
                    query_1 = queries[j]
                    query_2 = queries[k]
                    p_doc_id_overlap.append(doc_id_overlap(query_1['doc_ids'], query_2['doc_ids']))
                    query_pairs.append((query_1['search_query'], query_2['search_query']))
                # compute similarity between consecutive queries for each trajectory
                assert len(queries[j]['doc_ids']), (len(queries[j]['doc_ids']), filename, trajectories[i]['query_id'])
                c_doc_id_overlap.append(doc_id_overlap(queries[j+1]['doc_ids'], queries[j]['doc_ids']))
                query_pairs.append((queries[j+1]['search_query'], queries[j]['search_query']))
            avg_pairwise_doc_id_overlap += sum(p_doc_id_overlap) / len(p_doc_id_overlap)
            avg_consecutive_doc_id_overlap += sum(c_doc_id_overlap) / len(c_doc_id_overlap)
            
        # LLM Judge Similarity
        judge_results = llm_judge_similarity(query_pairs, llm, sampling_params, system_prompt)
            
        for i in range(len(trajectories)):
            p_sim, c_sim = [], []
            queries = trajectories[i]['search_queries']
                        
            # compute similarity between pairs of queries for each trajectory
            for j in range(len(queries)-1):
                for k in range(j+1, len(queries)):
                    query_1 = queries[j]
                    query_2 = queries[k]
                    # query_pairs.append((query_1['search_query'], query_2['search_query']))
                    p_sim.append(float(judge_results.pop(0)))
                # compute similarity between consecutive queries for each trajectory
                # query_pairs.append((queries[j+1]['search_query'], queries[j]['search_query']))
                c_sim.append(float(judge_results.pop(0)))
            avg_pairwise_similarity += sum(p_sim) / len(p_sim)
            avg_consecutive_similarity += sum(c_sim) / len(c_sim)
        assert len(judge_results) == 0, (len(judge_results), filename)
        
        
        for i in range(len(trajectories)):
            queries = trajectories[i]['search_queries']
            # START: cluster queries
            clustered_queries = cluster_queries(queries, similarity_threshold=args.similarity_threshold, similarity_function=args.similarity_function, llm=llm, sampling_params=sampling_params, system_prompt=system_prompt)
            # END: cluster queries
            
            # START: compute duplicates
            num_duplicates = len(queries) - len(clustered_queries)
            avg_repeat_perc += (num_duplicates/len(queries))
            for _cluster in clustered_queries:
                if len(_cluster) > 1:
                    _qs = [_query['search_query'].replace('\t', ' ') for _query in _cluster]
                    _qs = [' '.join(q) if isinstance(q, list) else q for q in _qs]
                    repeated_queries.append(_qs)
            # END: compute duplicates]
            
            
        # Average Statistics
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
        
        if args.output_csv:
            avg_repeat_perc_doc_id = 0.0
            csv_results.append([filename, '', avg_pairwise_doc_id_overlap, 
                                avg_consecutive_doc_id_overlap, 100*avg_repeat_perc_doc_id, '', avg_pairwise_similarity, 
                                avg_consecutive_similarity, 100*avg_repeat_perc, avg_num_queries, avg_gold_subqueries])
    
    # write_json(filename2queries, 'filename2queries.json')
    # write_tsv(repeated_queries, 'repeated_queries.tsv')
    if args.output_csv:
        write_csv(csv_results, 'csv_results.csv')
    return
    


if __name__ == "__main__":
    main()

