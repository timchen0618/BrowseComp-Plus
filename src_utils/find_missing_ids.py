import argparse
import json
import glob
import csv
import os


def get_leaf_dirs_with_json(root_path: str) -> list[str]:
    """Find directories that directly contain JSON files. Stops descending when such dirs are found."""
    results = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if any(f.endswith(".json") for f in filenames):
            results.append(dirpath)
            dirnames.clear()  # Don't recurse into subdirs
    return sorted(results)


def read_tsv_file(file_path: str) -> list[tuple[str, str]]:
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)


def read_jsonl_file(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def collect_query_ids_from_dir(dir_path: str) -> list[str]:
    """Collect query_id from each JSON file in the directory."""
    ids = []
    for path in glob.glob(os.path.join(dir_path, "*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                ids.append(data["query_id"])
        except (json.JSONDecodeError, KeyError):
            pass  # Skip invalid or malformed files
    return ids


def main():
    # decide if the file is a single file or a glob pattern
    if args.reference_file.endswith("*") or args.reference_file.endswith("*.tsv"):
        reference_files = glob.glob(args.reference_file)
        reference_files = sorted(reference_files)
    else:
        reference_files = [args.reference_file]

    if args.recursive:
        if not args.input_dir:
            parser.error("--recursive requires --input_dir")
        if args.jsonl_file:
            parser.error("--recursive cannot be used with --jsonl_file")
        dirs_to_check = get_leaf_dirs_with_json(args.input_dir)
        print(f"Found {len(dirs_to_check)} leaf directories with JSON files")
        for leaf_dir in dirs_to_check:
            output_run_query_ids = collect_query_ids_from_dir(leaf_dir)
            print(f"\n--- Checking {leaf_dir} ({len(output_run_query_ids)} query ids) ---")
            for reference_file in reference_files:
                reference_data = read_tsv_file(reference_file)
                query_ids = [row[0] for row in reference_data]
                missing_query_ids = set(query_ids) - set(output_run_query_ids)
                missing_query_ids = sorted(missing_query_ids)
                print(f"***Missing {len(missing_query_ids)} query ids in {reference_file}:***\n {missing_query_ids}\n")
        return

    if args.input_dir:
        output_run_query_ids = collect_query_ids_from_dir(args.input_dir)
        print(f"Found {len(output_run_query_ids)} output run query ids")

    if args.jsonl_file:
        output_run_query_ids = [item["query_id"] for item in read_jsonl_file(args.jsonl_file)]
        print(f"Found {len(output_run_query_ids)} output run query ids from {args.jsonl_file}")

    for reference_file in reference_files:
        print(f"Processing {reference_file}...")
        reference_data = read_tsv_file(reference_file)
        query_ids = [row[0] for row in reference_data]
        missing_query_ids = set(query_ids) - set(output_run_query_ids)
        missing_query_ids = sorted(missing_query_ids)
        print(f"***Missing {len(missing_query_ids)} query ids in {reference_file}:***\n {missing_query_ids}\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--reference_file", type=str, default="topics-qrels/bcp_10_shards/*")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively check every subfolder; stops at dirs containing JSON files")
    args = parser.parse_args()
        
        
    main()

