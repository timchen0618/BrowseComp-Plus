import json
import csv
from tqdm import tqdm
import argparse
from pathlib import Path
import json5
def read_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(filepath, data):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def read_tsv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [row for row in reader]
    
    
def write_tsv(filepath, data):
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)
    

def parse_trajectories(args):
    questions = read_tsv(args.question_file)
    id2question = {item[0]: item[1] for item in questions}
    print('loaded questions. Loading corpus...')
        
    data = read_jsonl(args.trajectory_file)
    corpus = read_tsv(args.corpus_file)
    id2item = {item[0]: item for item in corpus}  # id, text, title
    print('loaded corpus. Generating results...')
    # 500001
    out_data = []
    last_query_list = []
    data = sorted(data, key=lambda x: int(x['query_id']))
    for trajectory in tqdm(data):
        retrieved_docids = trajectory['retrieved_docids']
        all_docids = set()
        last_query = None
        has_doc_list = []
        for doc_dict in retrieved_docids:
            for query_before_parse, docids in doc_dict.items():
                all_docids.update(set(docids))
                if len(docids) > 0:
                    has_doc_list.append(True)
                else:
                    has_doc_list.append(False)
                break

        # print(all_docids)
        has_doc_list.reverse()
        retrieved_docids.reverse()
        
        for has_doc, doc_dict in zip(has_doc_list, retrieved_docids):
            if has_doc:
                last_query = json5.loads(list(doc_dict.keys())[0])['query']
                break
        
        if isinstance(last_query, list):
            last_query = ' '.join(last_query)
        # print(last_query)
        out_data.append({"question": id2question[trajectory['query_id']], 
                        "answers": [''], 
                        "ctxs": [{"id":docid, "text": id2item[docid][1], "title": id2item[docid][2]} for docid in all_docids],
                        "last_query": "" if last_query is None else last_query,
                        "qid": trajectory['query_id']})
        last_query_list.append({"question": "" if last_query is None else last_query, "qid": trajectory['query_id'], "answers": [''], "ctxs": []})
        
        
        
    print('Saving results...')
    write_jsonl(Path(args.input_dir) / 'qampari_combined_docs.jsonl', out_data)
    write_jsonl(Path(args.input_dir) / 'qampari_last_queries.jsonl', last_query_list)
    
    
def combine_last_search_and_trajectories(args):
    combined_docs = read_jsonl(Path(args.input_dir) / 'qampari_combined_docs.jsonl')
    last_query_retrieval_results = read_jsonl(args.last_query_retrieval_results)
    assert len(combined_docs) == len(last_query_retrieval_results)
    for inst, last_query_inst in zip(combined_docs, last_query_retrieval_results):
        assert inst['last_query'] == last_query_inst['question'],(inst['last_query'], last_query_inst['question'])
        if len(inst['ctxs']) < args.topk:
            inst['ctxs'].extend(last_query_inst['ctxs'][:(args.topk - len(inst['ctxs']))])
        else:
            inst['ctxs'] = inst['ctxs'][:args.topk]
    
    selected_indices = []
    for inst in combined_docs:
        assert len(inst['ctxs']) == args.topk
        for i in range(len(inst['ctxs'])):
            inst['ctxs'][i]['score'] = 1 / (i + 1 + 1e-6)
        selected_indices.append(int(inst['qid']))
    write_jsonl(Path(args.input_dir) / f'qampari_agentic_final_results_topk{args.topk}.jsonl', combined_docs)
    fw = open(Path(args.input_dir) / f'qampari_agentic_final_results_topk{args.topk}.txt', 'w')
    for index in selected_indices:
        fw.write(f'{index}\n')
    fw.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="qampari_runs/qwen3-0.6b/tongyi")
    parser.add_argument("--question-file", type=str, default="topics-qrels/qampari_dev_question_only.tsv")
    parser.add_argument("--corpus-file", type=str, default="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv")
    parser.add_argument("--trajectory-file", "-t", type=str, default="qampari_runs/qwen3-0.6b/tongyi/combined.jsonl")
    parser.add_argument("--command", type=str, default="parse_trajectories", choices=["parse_trajectories", "combine_last_search_and_trajectories", "subset"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--last-query-retrieval-results", "-l", type=str, default="qampari_runs/qwen3-0.6b/tongyi/last_query_retrieval_results.jsonl")
    
    args = parser.parse_args()
    
    if args.command == "parse_trajectories":
        parse_trajectories(args)
    elif args.command == "combine_last_search_and_trajectories":
        combine_last_search_and_trajectories(args)
    else:
        # selected_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20, 22, 23]
        selected_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 22, 23]
        data = read_jsonl("/scratch/hc3337/projects//diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl")
        out_data = [data[i] for i in selected_indices]
        write_jsonl("/scratch/hc3337/projects//diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_small_qwen.jsonl", out_data)
        raise ValueError(f"Invalid command: {args.command}")
    
    
    # Qwen3-0.6b
    # python generate_qampari_results.py
    # python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi/ -t qampari_runs/qwen3-0.6b/tongyi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries.jsonl 
    
    # Infly
    # python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi/ -t qampari_runs/infly/tongyi/combined.jsonl --command parse_trajectories
    # python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi/ -t qampari_runs/infly/tongyi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/inf/qampari_last_queries.jsonl 
    
    # Contriever
    # python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi/ -t qampari_runs/contriever/tongyi/combined.jsonl --command parse_trajectories
    # python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi/ -t qampari_runs/contriever/tongyi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/qampari_last_queries.jsonl 
