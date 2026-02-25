import os
import json
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

import csv
def write_tsv(lines, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(lines)

# input_file = "/scratch/hc3337/projects/BrowseComp-Plus/topics-qrels/qampari_dev_question_only.tsv"
# output_dir = "/scratch/hc3337/projects/BrowseComp-Plus/topics-qrels/qampari_10_shards"
# num_shards = 10

# os.makedirs(output_dir, exist_ok=True)

# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# total = len(lines)
# chunk_size = total // num_shards
# remainder = total % num_shards

# start = 0
# for i in range(num_shards):
#     end = start + chunk_size + (1 if i < remainder else 0)
#     shard_lines = lines[start:end]
#     shard_path = os.path.join(output_dir, f"q_{i}.tsv")
#     with open(shard_path, "w", encoding="utf-8") as out_f:
#         out_f.writelines(shard_lines)
#     print(f"Wrote {len(shard_lines)} lines to {shard_path}")
#     start = end




# input_file = "data/dev_data_gt_quest.jsonl"
# output_dir = "/scratch/hc3337/projects/BrowseComp-Plus/topics-qrels/quest_17_shards"
# num_shards = 17


# input_file = "data/dev_data_gt_webqsp.jsonl"
# output_dir = "/scratch/hc3337/projects/BrowseComp-Plus/topics-qrels/webqsp_10_shards"
# num_shards = 10


# data = read_jsonl(input_file)

# os.makedirs(output_dir, exist_ok=True)

# # with open(input_file, "r", encoding="utf-8") as f:
# #     lines = f.readlines()
# lines = []
# for qid, inst in enumerate(data):
#     lines.append([str(qid), inst['question_text']])
    
# write_tsv(lines, os.path.join(output_dir, "webqsp_dev_question_only.tsv"))
    

import datasets
output_dir = "/scratch/hc3337/projects/BrowseComp-Plus/topics-qrels/nq_10_shards"
num_shards = 10


data = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')['test']

os.makedirs(output_dir, exist_ok=True)

# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()
lines = []
for qid, inst in enumerate(data):
    lines.append([str(qid), inst['question']])
    if qid >= 999:
        break
    
write_tsv(lines, os.path.join(output_dir, "nq_dev_question_only.tsv"))
    
total = len(lines)
chunk_size = total // num_shards
remainder = total % num_shards

start = 0
for i in range(num_shards):
    end = start + chunk_size + (1 if i < remainder else 0)
    shard_lines = lines[start:end]
    shard_path = os.path.join(output_dir, f"q_{i}.tsv")
    write_tsv(shard_lines, shard_path)
    print(f"Wrote {len(shard_lines)} lines to {shard_path}")
    start = end

