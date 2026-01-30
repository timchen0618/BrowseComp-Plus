######################### Parse trajectories, and get last queries #########################
python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi/ -t qampari_runs/contriever/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi_k20/ -t qampari_runs/contriever/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi_k100/ -t qampari_runs/contriever/tongyi_multi_k100/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi/ -t qampari_runs/infly/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi_k20/ -t qampari_runs/infly/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi_k100/ -t qampari_runs/infly/tongyi_multi_k100/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi/ -t qampari_runs/qwen3-0.6b/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi_k20/ -t qampari_runs/qwen3-0.6b/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi_k100/ -t qampari_runs/qwen3-0.6b/tongyi_multi_k100/combined.jsonl --command parse_trajectories

python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi/ -t qampari_runs/contriever_finetuned/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi_k20/ -t qampari_runs/contriever_finetuned/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi_k100/ -t qampari_runs/contriever_finetuned/tongyi_multi_k100/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi/ -t qampari_runs/infly_finetuned/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi_k20/ -t qampari_runs/infly_finetuned/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi_k100/ -t qampari_runs/infly_finetuned/tongyi_multi_k100/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b_finetuned/tongyi_multi/ -t qampari_runs/qwen3-0.6b_finetuned/tongyi_multi/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/ -t qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/ -t qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/combined.jsonl --command parse_trajectories


# Deduplicate trajectories
# Combine JSON files into a single JSONL file
# And then parse the trajectories
PYTHONPATH=. python scripts_evaluation/deduplicate_trajectories.py qampari_runs/contriever/tongyi_base/ 
PYTHONPATH=. python scripts_evaluation/deduplicate_trajectories.py qampari_runs/infly/tongyi_base/ 
PYTHONPATH=. python scripts_evaluation/deduplicate_trajectories.py qampari_runs/qwen3-0.6b/tongyi_base/ 

PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir qampari_runs/contriever/tongyi_base
PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir qampari_runs/infly/tongyi_base
PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir qampari_runs/qwen3-0.6b/tongyi_base
python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_base/ -t qampari_runs/contriever/tongyi_base/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_base/ -t qampari_runs/infly/tongyi_base/combined.jsonl --command parse_trajectories
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_base/ -t qampari_runs/qwen3-0.6b/tongyi_base/combined.jsonl --command parse_trajectories


PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir quest_runs/contriever/tongyi_multi
PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir quest_runs/qwen3-0.6b/tongyi_multi
PYTHONPATH=. python src_utils/combine_json_to_jsonl.py --input_dir quest_runs/infly/tongyi_multi

python generate_qampari_results.py --input-dir quest_runs/contriever/tongyi_multi/ -t quest_runs/contriever/tongyi_multi/combined.jsonl --command parse_trajectories  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest
python generate_qampari_results.py --input-dir quest_runs/qwen3-0.6b/tongyi_multi/ -t quest_runs/qwen3-0.6b/tongyi_multi/combined.jsonl --command parse_trajectories  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest
python generate_qampari_results.py --input-dir quest_runs/infly/tongyi_multi/ -t quest_runs/infly/tongyi_multi/combined.jsonl --command parse_trajectories  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest

python generate_qampari_results.py --input-dir quest_runs/infly/tongyi_multi/ -t quest_runs/infly/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly/quest_last_queries_tongyi_multi.jsonl  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest
python generate_qampari_results.py --input-dir quest_runs/qwen3-0.6b/tongyi_multi/ -t quest_runs/qwen3-0.6b/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/quest_last_queries_tongyi_multi.jsonl  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest
python generate_qampari_results.py --input-dir quest_runs/contriever/tongyi_multi/ -t quest_runs/contriever/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/quest_last_queries_tongyi_multi.jsonl  --question-file topics-qrels/quest_dev_question_only.tsv --data_type quest





######################### Combine last search and trajectories #########################
# python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi/ -t qampari_runs/contriever/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/qampari_last_queries_tongyi_multi.jsonl   # saved to qampari_runs/contriever/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi_k20/ -t qampari_runs/contriever/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/contriever/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_multi_k100/ -t qampari_runs/contriever/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/contriever/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl

python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi/ -t qampari_runs/infly/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly/qampari_last_queries_tongyi_multi.jsonl   # saved to qampari_runs/infly/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi_k20/ -t qampari_runs/infly/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/infly/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_multi_k100/ -t qampari_runs/infly/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/infly/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl

python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi/ -t qampari_runs/qwen3-0.6b/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries_tongyi_multi.jsonl   # saved to qampari_runs/qwen3-0.6b/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi_k20/ -t qampari_runs/qwen3-0.6b/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/qwen3-0.6b/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_multi_k100/ -t qampari_runs/qwen3-0.6b/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/qwen3-0.6b/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl

python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi/ -t qampari_runs/contriever_finetuned/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever_finetuned/qampari_last_queries_tongyi_multi.jsonl   # saved to qampari_runs/contriever_finetuned/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi_k20/ -t qampari_runs/contriever_finetuned/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever_finetuned/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/contriever_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/contriever_finetuned/tongyi_multi_k100/ -t qampari_runs/contriever_finetuned/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever_finetuned/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/contriever_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl

# python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi/ -t qampari_runs/infly_finetuned/tongyi_multi/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly_finetuned/qampari_last_queries_tongyi_multi.jsonl   # saved to qampari_runs/infly_finetuned/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi_k20/ -t qampari_runs/infly_finetuned/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly_finetuned/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/infly_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/infly_finetuned/tongyi_multi_k100/ -t qampari_runs/infly_finetuned/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly_finetuned/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/infly_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl    

# python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/ -t qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b_finetuned/qampari_last_queries_tongyi_multi_k20.jsonl   # saved to qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/ -t qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b_finetuned/qampari_last_queries_tongyi_multi_k100.jsonl   # saved to qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl


python generate_qampari_results.py --input-dir qampari_runs/contriever/tongyi_base/ -t qampari_runs/contriever/tongyi_base/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/contriever/qampari_last_queries_tongyi_base.jsonl
python generate_qampari_results.py --input-dir qampari_runs/infly/tongyi_base/ -t qampari_runs/infly/tongyi_base/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/infly/qampari_last_queries_tongyi_base.jsonl
python generate_qampari_results.py --input-dir qampari_runs/qwen3-0.6b/tongyi_base/ -t qampari_runs/qwen3-0.6b/tongyi_base/combined.jsonl --command combine_last_search_and_trajectories -l ../autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries_tongyi_base.jsonl


######################### Evaluate results #########################
ls qampari_runs/contriever/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/contriever/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/contriever/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/infly/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/infly/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/infly/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/qwen3-0.6b/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/qwen3-0.6b/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
ls qampari_runs/qwen3-0.6b/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl

# ls qampari_runs/contriever_finetuned/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/contriever_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/contriever_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/infly_finetuned/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/infly_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/infly_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/qwen3-0.6b_finetuned/tongyi_multi/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k20/qampari_agentic_final_results_topk100.jsonl
# ls qampari_runs/qwen3-0.6b_finetuned/tongyi_multi_k100/qampari_agentic_final_results_topk100.jsonl