#!/usr/bin/env python3
"""
Split a selected_tool_calls JSONL from selected_tool_calls/all/ into per-split
sharded files under selected_tool_calls/{split}/{model}/seed{seed}/.

Usage:
    python src_utils/split_selected_tool_calls.py \
        --input selected_tool_calls/all/selected_tool_calls.jsonl \
        --model gpt-oss-120b \
        --seed 0

Output layout (matches SBATCH --trajectory-summary-file references):
    selected_tool_calls/full/{model}/seed{seed}/{stem}_{i}.jsonl          (i=0..9)
    selected_tool_calls/first50/{model}/seed{seed}/{stem}_first50.jsonl
    selected_tool_calls/test150/{model}/seed{seed}/{stem}_test150_{i}.jsonl  (i=0..2)
    selected_tool_calls/train680/{model}/seed{seed}/{stem}_train680_{i}.jsonl (i=0..7)
"""
import argparse
import json
from pathlib import Path

# split_name -> (tsv_path, num_shards, file_suffix)
SPLITS = {
    "full":     ("topics-qrels/bcp/queries.tsv",          10, ""),
    "first50":  ("topics-qrels/bcp/queries_first50.tsv",   1, "first50"),
    "test150":  ("topics-qrels/bcp/queries_test150.tsv",   3, "test150"),
    "train680": ("topics-qrels/bcp/queries_train680.tsv",  8, "train680"),
}


def load_qids(tsv_path):
    qids = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                qids.append(line.split("\t")[0])
    return qids


def main():
    ap = argparse.ArgumentParser(description="Split selected_tool_calls JSONL into per-split sharded files.")
    ap.add_argument("--input", required=True, help="Source JSONL (e.g. selected_tool_calls/all/selected_tool_calls.jsonl)")
    ap.add_argument("--model", required=True, help="Model name (e.g. gpt-oss-120b)")
    ap.add_argument("--seed", required=True, type=int, help="Seed number")
    ap.add_argument("--output-base", default="selected_tool_calls", help="Base output directory (default: selected_tool_calls)")
    args = ap.parse_args()

    input_path = Path(args.input)
    stem = input_path.stem
    output_base = Path(args.output_base)

    records = {}
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["query_id"]] = rec
    print(f"Loaded {len(records)} records from {input_path}")

    for split_name, (tsv_path, num_shards, suffix) in SPLITS.items():
        qids = load_qids(tsv_path)
        split_records = [records[qid] for qid in qids if qid in records]
        missing = [qid for qid in qids if qid not in records]
        if missing:
            print(f"  WARNING [{split_name}]: {len(missing)} query IDs not found in input")

        out_dir = output_base / split_name / args.model / f"seed{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if num_shards == 1:
            fname = f"{stem}_{suffix}.jsonl" if suffix else f"{stem}.jsonl"
            out_path = out_dir / fname
            with open(out_path, "w") as f:
                for rec in split_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"  {out_path}: {len(split_records)} records")
        else:
            chunk = len(split_records) // num_shards
            remainder = len(split_records) % num_shards
            start = 0
            for i in range(num_shards):
                end = start + chunk + (1 if i < remainder else 0)
                shard = split_records[start:end]
                fname = f"{stem}_{suffix}_{i}.jsonl" if suffix else f"{stem}_{i}.jsonl"
                out_path = out_dir / fname
                with open(out_path, "w") as f:
                    for rec in shard:
                        f.write(json.dumps(rec) + "\n")
                print(f"  {out_path}: {len(shard)} records")
                start = end


if __name__ == "__main__":
    main()
