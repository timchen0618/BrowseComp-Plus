#!/usr/bin/env python3
"""
Add 'search_counts' field to records in a JSONL file.

The 'search_counts' field will contain the number of search tool calls,
extracted from tool_call_counts["search"].
"""

import json
import sys
from pathlib import Path


def add_search_counts(jsonl_path: Path, output_path: Path = None) -> int:
    """
    Add search_counts field to all records in a JSONL file.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output JSONL file (default: overwrite input)
        
    Returns:
        Number of records processed
    """
    if not jsonl_path.exists():
        print(f"Error: JSONL file {jsonl_path} does not exist", file=sys.stderr)
        return 0
    
    output_path = output_path or jsonl_path
    
    print(f"Processing {jsonl_path}...")
    
    records = []
    total_count = 0
    
    # Read all records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                obj = json.loads(line)
                
                # Extract search count from tool_call_counts
                tool_call_counts = obj.get("tool_call_counts", {})
                search_count = tool_call_counts.get("search", 0)
                
                # Add search_counts field
                obj["search_counts"] = int(search_count)
                
                records.append(obj)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
    
    # Write records back
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Processed {len(records)} records, added search_counts field to all")
    if output_path != jsonl_path:
        print(f"Saved to {output_path}")
    
    return len(records)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add 'search_counts' field to records in a JSONL file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="data/decrypted_run_files/bm25/oss-20b.jsonl",
        help="Input JSONL file (default: data/decrypted_run_files/bm25/oss-20b.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: overwrite input)",
    )
    
    args = parser.parse_args()
    
    add_search_counts(args.input, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()

