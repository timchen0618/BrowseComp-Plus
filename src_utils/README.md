# src_utils/

Data loading, filtering, repair, and preprocessing utilities. These are general-purpose building blocks called by the shell wrappers in `scripts/` or used standalone.

---

## Existing Utilities

| File | Purpose |
|------|---------|
| `load_trajectories.py` | Core JSONL loading and file discovery |
| `filter_by_query_ids.py` | Subset trajectories by query IDs |
| `add_search_counts.py` | Augment trajectories with search call statistics |
| `shard.py` | Partition large datasets into shards |
| `combine_json_to_jsonl.py` | Batch-convert individual JSON files to JSONL |

---

## New Utilities (added April 14, 2026)

### `filter_empty_runs.py`

Scans trajectory JSON directories and identifies "empty" runs — files that contain only the initial system/user prompt with no agent reasoning, tool calls, or outputs. Supports four modes:

| Mode | Behavior |
|------|----------|
| `list` | Report counts only (optionally write path lists) |
| `copy-nonempty` | Copy non-empty files to a clean output directory |
| `move-empty` | Move empty files aside into a separate directory |
| `delete-empty` | Permanently delete empty files |

```bash
python src_utils/filter_empty_runs.py \
    --input-dirs runs/.../seed4 runs/.../seed5 \
    --mode list --write-lists
```

### `filter_empty_selected_tool_calls.py`

Filters `selected_tool_calls` JSONL files by removing records where no tool calls were selected (empty `selected_indices`). Includes a recovery mechanism: if `selected_indices` is missing or malformed, attempts to recover it from the `raw_response` field by extracting embedded JSON or regex-matching the indices array.

```bash
python src_utils/filter_empty_selected_tool_calls.py \
    --input selected_tool_calls/file.jsonl \
    --output selected_tool_calls/file.nonempty.jsonl \
    --dropped selected_tool_calls/file.dropped.jsonl
```

### `filter_selected_tool_calls_by_error.py`

Drops rows from a selected-tool-calls JSONL whose `error` field matches a specified string. Useful for removing known failure modes (e.g., `no_candidate_tool_calls`) before downstream processing.

```bash
python src_utils/filter_selected_tool_calls_by_error.py \
    --input file.jsonl \
    --output file.filtered.jsonl \
    --drop-error "no_candidate_tool_calls" \
    --write-dropped file.dropped.jsonl
```

### `parse_eval_out.py`

Parses SLURM evaluation output logs (`sbatch_outputs/eval.out`) to extract per-run metrics — accuracy, recall, average search calls — and writes them to a CSV. Uses regex to match the structured output blocks produced by `scripts_evaluation/evaluate_run.py`.

```bash
python -c "from src_utils.parse_eval_out import save_to_csv; save_to_csv('sbatch_outputs/eval.out', 'eval_summary.csv')"
```

### `repair_selected_tool_calls_jsonl.py`

Repairs selected-tool-calls JSONL files where `selected_indices` is missing or malformed due to LLM response parsing failures. Recovery strategies (in order):

1. Extract a complete JSON object from `raw_response` (stripping markdown fences).
2. Regex-match `"selected_indices": [...]` from truncated or incomplete JSON.

Writes a new file with repaired records; reports total/repaired/preserved counts.

```bash
python src_utils/repair_selected_tool_calls_jsonl.py \
    --input file.jsonl --output file.repaired.jsonl
```
