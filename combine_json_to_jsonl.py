#!/usr/bin/env python3
"""
Combine all JSON files in a directory into a single JSONL file.

Each JSON file should contain a single JSON object, which will be written
as one line in the output JSONL file.
"""

import json
import sys
from pathlib import Path
from typing import List


def combine_json_files(input_dir: Path, output_file: Path) -> None:
    """
    Combine all .json files in input_dir into a single .jsonl file.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Path to output JSONL file
    """
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Find all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {input_dir}", file=sys.stderr)
        return
    
    print(f"Found {len(json_files)} JSON files to combine")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    combined_count = 0
    error_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as in_f:
                    data = json.load(in_f)
                
                # Write as a single line (JSONL format)
                json.dump(data, out_f, ensure_ascii=False)
                out_f.write("\n")
                combined_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON file {json_file}: {e}", file=sys.stderr)
                error_count += 1
            except Exception as e:
                print(f"Error: Failed to process {json_file}: {e}", file=sys.stderr)
                error_count += 1
    
    print(f"Successfully combined {combined_count} files into {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors", file=sys.stderr)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine all JSON files in a directory into a single JSONL file"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default="runs/bm25/oss-20b",
        help="Directory containing JSON files (default: runs/bm25/oss-20b)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (default: <input_dir>/combined.jsonl)",
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_file = args.output or (input_dir / "combined.jsonl")
    
    combine_json_files(input_dir, output_file)


if __name__ == "__main__":
    main()

