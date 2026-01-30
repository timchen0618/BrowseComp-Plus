import argparse
import json
import glob
import csv
import os

def read_tsv_file(file_path: str) -> list[tuple[str, str]]:
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)


def read_jsonl_file(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def main():
    # decide if the file is a single file or a glob pattern
    if args.reference_file.endswith("*") or args.reference_file.endswith("*.tsv"):
        reference_files = glob.glob(args.reference_file)
        reference_files = sorted(reference_files)
    else:
        reference_files = [args.reference_file]
        
    if args.input_dir:
        # get all the output run query ids in the input directory
        output_run_query_ids = []
        for file in glob.glob(os.path.join(args.input_dir, "*.json")):
            with open(file, "r") as f:
                data = json.load(f)
                output_run_query_ids.append(data["query_id"])
            
        print(f"Found {len(output_run_query_ids)} output run query ids")
        
    if args.jsonl_file:
        output_run_query_ids = [item["query_id"] for item in read_jsonl_file(args.jsonl_file)]
        print(f"Found {len(output_run_query_ids)} output run query ids from {args.jsonl_file}")
        
    for reference_file in reference_files:
        print(f"Processing {reference_file}...")
        reference_data = read_tsv_file(reference_file)
        query_ids = [row[0] for row in reference_data]
        
        missing_query_ids = set(query_ids) - set(output_run_query_ids)
        missing_query_ids = sorted(list(missing_query_ids))
        print(f"***Missing {len(missing_query_ids)} query ids in {reference_file}:***\n {missing_query_ids}\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--reference_file", type=str, default="topics-qrels/bcp_10_shards/*")
    args = parser.parse_args()
        
        
    main()

