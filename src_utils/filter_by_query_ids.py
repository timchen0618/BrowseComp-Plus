#!/usr/bin/env python3
"""
Filter files to keep only queries that match the query IDs in a reference JSONL file.

This script:
1. Reads query IDs from a reference JSONL file
2. Filters JSON files in a directory to keep only those with matching query IDs
3. Optionally filters other JSONL files as well
"""

import json
import sys
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict


def extract_query_ids_from_jsonl(jsonl_path: Path) -> Set[str]:
    """
    Extract all query IDs from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        Set of query IDs (as strings)
    """
    query_ids = set()
    
    if not jsonl_path.exists():
        print(f"Error: Reference file {jsonl_path} does not exist", file=sys.stderr)
        return query_ids
    
    print(f"Reading query IDs from {jsonl_path}...")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                query_id = obj.get("query_id")
                if query_id is not None:
                    # Convert to string for consistent comparison
                    query_ids.add(str(query_id))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error processing line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
    
    print(f"Found {len(query_ids)} unique query IDs")
    return query_ids


def filter_json_files(input_dir: Path, query_ids: Set[str], backup: bool = True) -> int:
    """
    Filter JSON files in a directory, keeping only those with matching query IDs.
    
    Args:
        input_dir: Directory containing JSON files
        query_ids: Set of query IDs to keep
        backup: If True, create .bak files before deletion
        
    Returns:
        Number of files kept
    """
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Warning: Directory {input_dir} does not exist or is not a directory", file=sys.stderr)
        return 0
    
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0
    
    print(f"\nFiltering {len(json_files)} JSON files in {input_dir}...")
    
    kept_count = 0
    removed_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            query_id = data.get("query_id")
            query_id_str = str(query_id) if query_id is not None else None
            
            if query_id_str in query_ids:
                kept_count += 1
            else:
                # Remove file (or create backup)
                if backup:
                    backup_path = json_file.with_suffix(json_file.suffix + ".bak")
                    json_file.rename(backup_path)
                    print(f"  Moved {json_file.name} -> {json_file.name}.bak (query_id: {query_id_str})")
                else:
                    json_file.unlink()
                    print(f"  Removed {json_file.name} (query_id: {query_id_str})")
                removed_count += 1
                
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)
    
    print(f"Kept {kept_count} files, removed {removed_count} files")
    return kept_count


def filter_jsonl_file(jsonl_path: Path, query_ids: Set[str], output_path: Path = None) -> int:
    """
    Filter a JSONL file to keep only records with matching query IDs.
    
    Args:
        jsonl_path: Path to input JSONL file
        query_ids: Set of query IDs to keep
        output_path: Path to output JSONL file (default: overwrite input)
        
    Returns:
        Number of records kept
    """
    if not jsonl_path.exists():
        print(f"Warning: JSONL file {jsonl_path} does not exist", file=sys.stderr)
        return 0
    
    output_path = output_path or jsonl_path
    
    print(f"\nFiltering JSONL file {jsonl_path}...")
    
    kept_records = []
    total_count = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                obj = json.loads(line)
                query_id = obj.get("query_id")
                query_id_str = str(query_id) if query_id is not None else None
                
                if query_id_str in query_ids:
                    kept_records.append(obj)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing line {line_num} in {jsonl_path}: {e}", file=sys.stderr)
    
    # Sort records by query_id (as integers, from small to large)
    def get_query_id_int(record):
        """Helper function to extract query_id as integer for sorting."""
        query_id = record.get("query_id")
        if query_id is None:
            return float('inf')  # Put records without query_id at the end
        try:
            return int(query_id)
        except (ValueError, TypeError):
            # If conversion fails, try string comparison as fallback
            try:
                return int(str(query_id))
            except (ValueError, TypeError):
                return float('inf')  # Put non-numeric query_ids at the end
    
    kept_records.sort(key=get_query_id_int)
    
    # Write filtered and sorted records
    with open(output_path, "w", encoding="utf-8") as f:
        for record in kept_records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Kept {len(kept_records)} out of {total_count} records (sorted by query_id)")
    if output_path != jsonl_path:
        print(f"Saved filtered file to {output_path}")
    
    return len(kept_records)


def filter_jsonl_directory(jsonl_dir: Path, query_ids: Set[str], exclude_file: Path = None) -> dict:
    """
    Filter all JSONL files in a directory (recursively), excluding a specific file.
    
    Args:
        jsonl_dir: Directory containing JSONL files (searches recursively)
        query_ids: Set of query IDs to keep
        exclude_file: Path to exclude from filtering (e.g., the reference file)
        
    Returns:
        Dictionary with counts of files processed
    """
    if not jsonl_dir.exists() or not jsonl_dir.is_dir():
        print(f"Warning: Directory {jsonl_dir} does not exist or is not a directory", file=sys.stderr)
        return {"processed": 0, "total_records_kept": 0}
    
    # Find all JSONL files recursively
    jsonl_files = list(jsonl_dir.rglob("*.jsonl"))
    
    # Exclude the reference file if specified
    if exclude_file:
        exclude_file = exclude_file.resolve()
        jsonl_files = [f for f in jsonl_files if f.resolve() != exclude_file]
    
    if not jsonl_files:
        print(f"No JSONL files found in {jsonl_dir}")
        return {"processed": 0, "total_records_kept": 0}
    
    print(f"\nFound {len(jsonl_files)} JSONL files to filter in {jsonl_dir}")
    
    total_kept = 0
    processed = 0
    
    for jsonl_file in sorted(jsonl_files):
        records_kept = filter_jsonl_file(jsonl_file, query_ids)
        total_kept += records_kept
        processed += 1
    
    print(f"\nProcessed {processed} JSONL files, kept {total_kept} total records")
    return {"processed": processed, "total_records_kept": total_kept}


def calculate_average_search_calls(jsonl_dir: Path, query_ids: Set[str]) -> Dict[str, float]:
    """
    Calculate the average number of search calls per file across queries.
    
    Args:
        jsonl_dir: Directory containing JSONL files (searches recursively)
        query_ids: Set of query IDs to consider (only queries in this set will be counted)
        
    Returns:
        Dictionary mapping file paths (relative to jsonl_dir) to average search call counts
    """
    if not jsonl_dir.exists() or not jsonl_dir.is_dir():
        print(f"Warning: Directory {jsonl_dir} does not exist or is not a directory", file=sys.stderr)
        return {}
    
    # Find all JSONL files recursively
    jsonl_files = list(jsonl_dir.rglob("*.jsonl"))
    
    if not jsonl_files:
        return {}
    
    # Dictionary to store search counts per file: {file_path: [list of search counts]}
    file_search_counts: Dict[str, List[int]] = defaultdict(list)
    
    print(f"\nCalculating average search calls per file from {len(jsonl_files)} files...")
    
    jsonl_dir_resolved = jsonl_dir.resolve()
    
    for jsonl_file in sorted(jsonl_files):
        # Get relative path from jsonl_dir (e.g., "bm25/gpt5.jsonl")
        try:
            file_path = str(jsonl_file.relative_to(jsonl_dir_resolved))
        except ValueError:
            # If file is not relative to jsonl_dir, use full path
            file_path = str(jsonl_file)
        
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        query_id = obj.get("query_id")
                        query_id_str = str(query_id) if query_id is not None else None
                        
                        # Only count queries that match the query_ids set
                        if query_id_str in query_ids:
                            # Extract search count from tool_call_counts
                            if 'tool_call_counts' in obj:
                                tool_call_counts = obj.get("tool_call_counts", {})
                                search_count = tool_call_counts.get("search", 0)
                            elif 'search_counts' in obj:
                                search_count = obj.get("search_counts", 0)
                            else:
                                search_count = 0
                                
                            # Convert to int (handle various types)
                            try:
                                search_count = int(search_count)
                            except (ValueError, TypeError):
                                search_count = 0
                            
                            file_search_counts[file_path].append(search_count)
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {jsonl_file}: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num} in {jsonl_file}: {e}", file=sys.stderr)
                        
        except Exception as e:
            print(f"Warning: Error reading {jsonl_file}: {e}", file=sys.stderr)
    
    # Calculate averages per file
    file_averages: Dict[str, float] = {}
    for file_path, search_counts in file_search_counts.items():
        if len(search_counts) > 0:
            average = sum(search_counts) / len(search_counts)
            file_averages[file_path] = average
    
    return file_averages


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter files to keep only queries matching IDs from a reference JSONL file"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default="data/decrypted_run_files/bm25/oss-20b.jsonl",
        help="Reference JSONL file containing query IDs to keep (default: data/decrypted_run_files/bm25/oss-20b.jsonl)",
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        default="runs/bm25/oss-20b",
        help="Directory containing JSON files to filter (default: runs/bm25/oss-20b)",
    )
    parser.add_argument(
        "--jsonl_files",
        type=Path,
        nargs="*",
        help="Additional JSONL files to filter (optional)",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=Path,
        default="data/decrypted_run_files",
        help="Directory containing JSONL files to filter recursively (default: data/decrypted_run_files)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create .bak backup files when removing JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files",
    )
    
    args = parser.parse_args()
    
    # Extract query IDs from reference file
    query_ids = extract_query_ids_from_jsonl(args.reference)
    
    if not query_ids:
        print("Error: No query IDs found in reference file", file=sys.stderr)
        sys.exit(1)
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be modified")
    
    # Filter JSON files in directory
    if args.json_dir and not args.dry_run:
        filter_json_files(args.json_dir, query_ids, backup=not args.no_backup)
    elif args.dry_run and args.json_dir:
        # Dry run: just count what would be kept/removed
        json_files = list(args.json_dir.glob("*.json")) if args.json_dir.exists() else []
        kept = 0
        removed = 0
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                query_id = str(data.get("query_id")) if data.get("query_id") else None
                if query_id in query_ids:
                    kept += 1
                else:
                    removed += 1
            except:
                pass
        print(f"\nWould keep {kept} JSON files, remove {removed} JSON files")
    
    # Filter JSONL files in directory
    if args.jsonl_dir and not args.dry_run:
        filter_jsonl_directory(args.jsonl_dir, query_ids, exclude_file=args.reference)
    elif args.dry_run and args.jsonl_dir:
        # Dry run: count what would be filtered
        jsonl_files = list(args.jsonl_dir.rglob("*.jsonl")) if args.jsonl_dir.exists() else []
        if args.reference:
            exclude_file = args.reference.resolve()
            jsonl_files = [f for f in jsonl_files if f.resolve() != exclude_file]
        print(f"\nWould filter {len(jsonl_files)} JSONL files in {args.jsonl_dir}")
    
    # Filter additional JSONL files (if specified)
    if args.jsonl_files and not args.dry_run:
        for jsonl_file in args.jsonl_files:
            filter_jsonl_file(jsonl_file, query_ids)
    elif args.dry_run and args.jsonl_files:
        print(f"\nWould filter {len(args.jsonl_files)} additional JSONL files")
    
    # Calculate and print average search calls per file
    print("\n" + "=" * 80)
    print("Average Number of Search Calls per File")
    print("=" * 80)
    
    file_averages = calculate_average_search_calls(args.jsonl_dir, query_ids)
    
    if file_averages:
        # Sort by file path for consistent output
        sorted_files = sorted(file_averages.items())
        
        print(f"\nAverage search calls across {len(query_ids)} queries:\n")
        for file_path, avg_count in sorted_files:
            print(f"  {file_path:40s}: {avg_count:.2f}")
    else:
        print("\nNo data found to calculate averages.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

    """
    Filter files to keep only queries that match the query IDs in a reference JSONL file.

    This script:
    1. Reads query IDs from a reference JSONL file
    2. Filters JSON files in a directory to keep only those with matching query IDs
    3. Optionally filters other JSONL files as well
    """
