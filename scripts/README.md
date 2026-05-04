# scripts/

Helper and analysis scripts for the selected-tool-calls pipeline, run monitoring, and visualization. Added April 15, 2026.

---

## Selected Tool Calls — Validation & Augmentation

These scripts operate on `selected_tool_calls/*.jsonl` files produced by `select_useful_tool_calls.py`.

### `augment_selected_tool_calls_with_candidates_seed0.py`

Augments `selected_tool_calls.jsonl` in-place with two new fields per row:
- **`candidates`**: the full list of candidate tool-call indices reconstructed from seed0 trajectories.
- **`correct_num_selected`**: boolean — whether `len(selected_indices) == k_effective` and all indices are valid candidates.

Creates a `.bak` backup before modifying. Imports `find_candidate_tool_indices` from `select_useful_tool_calls.py`.

```bash
python scripts/augment_selected_tool_calls_with_candidates_seed0.py \
    --selected-jsonl selected_tool_calls/selected_tool_calls.jsonl \
    --seed0-dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed0
```

### `check_selected_against_candidates_seed0.py`

Validates that `selected_indices` in every row are a subset of the candidate tool-call indices from seed0 trajectories. Outputs a per-row status summary (ok / bad_indices / load_error / parse_error / schema_error) and optionally dumps invalid rows with catalog previews.

```bash
python scripts/check_selected_against_candidates_seed0.py \
    --dump-invalid --write-jsonl status.jsonl
```

### `compute_selected_tool_calls_stats.py`

Computes summary statistics for one or more selected-tool-calls JSONL files:
- % valid instances (relaxed or strict mode)
- % with `correct_num_selected` present and true
- Average number of selected indices per row
- Top invalid reasons breakdown

```bash
python scripts/compute_selected_tool_calls_stats.py \
    selected_tool_calls/selected_tool_calls.jsonl \
    selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl \
    --validity relaxed --json
```

---

## Shell Wrappers

Thin wrappers that invoke `src_utils/` scripts with project-specific paths.

### `filter_empty_runs_gpt_oss_120b_seeds4_7.sh`

Lists (and optionally copies/moves) empty trajectory files in gpt-oss-120b seeds 4–7.

### `filter_empty_selected_tool_calls_gpt_oss_120b.sh`

Filters rows with no selected tool calls from the gpt-oss-120b original-messages JSONL, writing a clean output and a dropped-records file.

### `repair_selected_tool_calls_gpt_oss_120b.sh`

Repairs gpt-oss-120b selected-tool-calls JSONL by recovering `selected_indices` from `raw_response` where the JSON parse originally failed.

---

## Visualization

### `plot_search_call_distribution.py`

Plots a multi-panel histogram of search-call counts per trajectory across matched folders. Each subplot corresponds to a trajectory variant (e.g., full trajectory, summary, selected tools) with bins: 0, 1–5, 6–10, …, >30.

```bash
python scripts/plot_search_call_distribution.py \
    --pattern "runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_*" \
    --output figures/search_call_distribution_gpt-oss-120b.png
```

### `plot_selected_position_histogram.py`

Plots the distribution of where selected tool calls fall among the candidate list (1st, 2nd, …, >20th candidate). Compares Tags vs. Original Messages variants side by side.

```bash
python scripts/plot_selected_position_histogram.py
# Output: figures/selected_candidate_position_distribution.png
```

---

## Dataset Splitting & Sharding

### `split_bcp_test150.py`

Samples a reproducible `test150` / `train680` split of the 830-query BrowseComp-Plus dataset. Uses a fixed random seed (default `42`) and writes matching query TSVs under `topics-qrels/bcp/` plus JSONL ground-truth subsets under `data/`. Streams the 2 GB source JSONL once and copies each line verbatim, so the split is both memory-safe and byte-identical to the original records.

Outputs (default names):
- `topics-qrels/bcp/queries_test150.tsv` — 150 rows
- `topics-qrels/bcp/queries_train680.tsv` — 680 rows
- `data/browsecomp_plus_decrypted_test150.jsonl` — 150 records
- `data/browsecomp_plus_decrypted_train680.jsonl` — 680 records
- `topics-qrels/bcp/queries_test150_qids.txt` — 150 query IDs, one per line (only when `--test-qids-out` is passed)

```bash
python scripts/split_bcp_test150.py \
    --test-size 150 \
    --seed 42 \
    --test-name test150 \
    --train-name train680 \
    --test-qids-out auto
```

Key flags:
- `--test-size N`: number of test examples to sample (default `150`).
- `--seed N`: RNG seed for the shuffle (default `42`; rerunning with the same seed is idempotent).
- `--test-name NAME` / `--train-name NAME`: label appended to output filenames (e.g. `queries_{NAME}.tsv`, `browsecomp_plus_decrypted_{NAME}.jsonl`).
- `--queries-tsv PATH` / `--data-jsonl PATH`: override the source queries TSV and ground-truth JSONL.
- `--test-qids-out PATH | auto | -`: optional path to write the sampled test query IDs (one per line, in the order they appear in the source TSV). `auto` or `-` defaults to `topics-qrels/bcp/queries_{test_name}_qids.txt`. `--test-indices-out` is accepted as a legacy alias.

### `shard_queries_tsv.py`

Splits any queries TSV into `N` contiguous shards `q_0.tsv` .. `q_{N-1}.tsv` in a target directory, matching the existing `topics-qrels/bcp/bcp_10_shards/` layout. Row order is preserved; when the row count doesn't divide evenly, the first `N % num_shards` shards get one extra row.

```bash
# Shard the test split into 3 (50 rows each)
python scripts/shard_queries_tsv.py \
    --input topics-qrels/bcp/queries_test150.tsv \
    --out-dir topics-qrels/bcp/bcp_test150_3_shards \
    --num-shards 3

# Shard the train split into 8 (~85 rows each)
python scripts/shard_queries_tsv.py \
    --input topics-qrels/bcp/queries_train680.tsv \
    --out-dir topics-qrels/bcp/bcp_train680_8_shards \
    --num-shards 8
```

Typical companion to `split_bcp_test150.py`. Produces:
- `topics-qrels/bcp/bcp_test150_3_shards/q_{0,1,2}.tsv` (50 rows each)
- `topics-qrels/bcp/bcp_train680_8_shards/q_{0..7}.tsv` (~85 rows each)

Use as `--reference_file "topics-qrels/bcp/bcp_test150_3_shards/*"` (or `bcp_train680_8_shards/*`) with the shard-aware tools.

---

## Run Monitoring

### `update_submit_missing.py`

Parses `find_missing_ids.py` output (piped or from a file) and generates a formatted `MISSING` dict for pasting into `submit_missing.py`. Supports both recursive mode (auto-detects run names from `--- Checking` headers) and single-directory mode (requires `--run-name`).

```bash
# Full split (10 shards)
python src_utils/find_missing_ids.py --recursive \
    --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b \
    --reference_file "topics-qrels/bcp/bcp_10_shards/*" \
    | python scripts/update_submit_missing.py

# Test split (3 shards)
python src_utils/find_missing_ids.py --recursive \
    --input_dir runs/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b \
    --reference_file "topics-qrels/bcp/bcp_test150_3_shards/*" \
    | python scripts/update_submit_missing.py

# Train split (8 shards)
python src_utils/find_missing_ids.py --recursive \
    --input_dir runs/bcp/Qwen3-Embedding-8B/train680/gpt-oss-120b \
    --reference_file "topics-qrels/bcp/bcp_train680_8_shards/*" \
    | python scripts/update_submit_missing.py
```
