# Agent Run Workflow

End-to-end guide for submitting agent runs on SLURM, finding incomplete shards, resubmitting them, and evaluating results.

## Overview

```
1. Configure runs  ──>  2. Submit to SLURM  ──>  3. Find missing IDs
       ▲                                                  │
       │                                                  ▼
       └──── 4. Resubmit missing shards  <───────────────┘
                                                          │
                                          5. Evaluate  <──┘
```

---

## Phase 1 — Configure and Submit Runs

### Fresh runs (full shard array)

Edit the shell variables at the top of the SBATCH template directly:

- **Full split (10 shards):** `run_qwen3_planning.SBATCH` — uses `#SBATCH --array=0-9`, one shard per TSV in `topics-qrels/{dataset}/{dataset}_10_shards/q_{i}.tsv`
- **First-50 split (single job):** `run_qwen3_first50.SBATCH` — runs against `topics-qrels/{dataset}/queries_first50.tsv`

Key variables in both templates:

| Variable | Purpose | Examples |
|----------|---------|----------|
| `MODEL_NAME` | Agent LLM | `gpt-oss-120b`, `tongyi` |
| `mode` | Run mode | `org`, `planning_v4`, `traj_ext`, `traj_summary_ext` |
| `seed` | Random seed | `0`, `1`, ... |
| `dataset` | Benchmark | `bcp`, `frames`, `musique` |

Submit directly:

```bash
sbatch run_qwen3_planning.SBATCH    # full split
sbatch run_qwen3_first50.SBATCH     # first50 split
```

### Batch submission via `submit_missing.py`

For submitting multiple runs at once (or resubmitting missing shards), use `submit_missing.py`. It reads a SBATCH template, patches the relevant variables per run, and submits each as a separate job.

Edit the `MISSING` dict (full split) or `MISSING_FIRST50` / `MISSING_FRAMES_FIRST50` / `MISSING_MUSIQUE_FIRST50` dicts:

```python
# Full split — value is a list of shard indices
MISSING = {
    "gpt-oss-120b_planning_v4_seed0": list(range(10)),   # all 10 shards
    "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0": [3, 5], # only shards 3 and 5
}

# First50 split — value is None (model/mode/seed auto-parsed from name)
MISSING_FIRST50 = {
    "gpt-oss-120b_planning_v4_seed0": None,
}
```

How the keys in `MISSING` dicts map to actual folder names: 
`runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/planning_v4_seed0/` -> `"gpt-oss-120b_planning_v4_seed0"`  
`runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_ext_gpt-oss-120b_seed0/` -> `"gpt-oss-120b_traj_ext_gpt-oss-120b_seed0"`  
Basically prepend the model name in front of the leaf folder name. 

Run name format: `{model}_{mode}_seed{N}`. The `parse_run_name()` function extracts model, mode, and seed automatically.

```bash
python submit_missing.py            # dry-run — prints what would be submitted
python submit_missing.py --submit   # actually submits to SLURM (200s delay between jobs)
```

### Needed Runs

| Run name (leaf folder) | `submit_missing.py` key | Seed |
|------------------------|-------------------------|------|
| `seed0` | `gpt-oss-120b_seed0` | 0 |
| `traj_orig_ext_gpt-oss-120b_seed0` | `gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0` | 0 |
| `traj_summary_orig_ext_gpt-oss-120b_seed0` | `gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0` | 0 |
| `traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0` | `gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0` | 0 |

Note: 
1. You need to run `seed0` first, in order to get a set of trajectories to run the remaining ones. Be sure to check for empty JSON files and re-run them.
2. For `traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0`, you need to also run `portkey.py` to get the "selected_tools" (see instructions below). Remember to run `seed0` with `--save-raw-messages` so that `original_messages` is present. 

### Getting Selected Tools

The `traj_summary_orig_ext_selected_tools` runs require a pre-computed JSONL of curated tool-call excerpts. These are produced by `select_useful_tool_calls.py`, which uses Gemini 2.5 Pro (via Portkey) to pick the k most useful tool calls from each trajectory.

**Prerequisites:**

- A completed `seed0` (or other baseline) run with `--save-raw-messages` so that `original_messages` is present in each trajectory JSON.
- `PORTKEY_API_KEY` set in your environment (or passed via `--api-key`).
- `pip install portkey-ai` in your environment.

**How it works:**

1. For each trajectory JSON, the script finds all candidate tool-call indices (search / get_document calls).
2. It builds a catalog of those candidates (with argument hints and output previews) and sends it along with the full trajectory context to Gemini.
3. Gemini returns a JSON response with `selected_indices` — the k most useful tool calls.
4. The script extracts verbatim excerpts (reasoning + exact args + full output) for the selected indices and writes one JSONL row per query.

**Usage (full split, 10 shards):**

```bash
export PORTKEY_API_KEY="..."

# Process all trajectories from seed4 (or whichever seed has original_messages)
python select_useful_tool_calls.py \
  --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
  --output selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
  --use-original-messages \
  --num-threads 8 \
  --queries-tsv topics-qrels/bcp/queries.tsv
```

For first50:

```bash
python select_useful_tool_calls.py \
  --trajectory-dir runs/bcp/Qwen3-Embedding-8B/first50/gpt-oss-120b/seed4 \
  --output selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_first50.jsonl \
  --use-original-messages \
  --num-threads 8 \
  --queries-tsv topics-qrels/bcp/queries_first50.tsv
```

**Key flags:**

| Flag | Purpose |
|------|---------|
| `--trajectory-dir` | Directory of trajectory JSON files to process |
| `--output` | Output JSONL path (appends; auto-resumes by skipping completed query IDs) |
| `--use-original-messages` | Read from `original_messages` (raw API messages) instead of the normalized `result[]` |
| `--k` | Number of tool calls to select per trajectory (default: 5) |
| `--num-threads` | Parallel Gemini calls |
| `--dry-run` | Report prompt sizes without calling Gemini |
| `--queries-tsv` | TSV file for question text lookup (default: `topics-qrels/bcp/queries.tsv`) |

**Post-processing:**

After generating the selected tool calls JSONL, shard it into per-shard files for the SBATCH array jobs:

```bash
# The SBATCH template expects files like:
#   selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired_{0..9}.jsonl
```

Each shard will have the same query ids as the sharded query files (`topics-qrels/bcp/bcp_10_shards/q_*.tsv`).  
The sharded files are referenced by `run_qwen3_planning.SBATCH` in the `traj_summary_orig_ext_selected_tools` case block via `--trajectory-summary-file`.

---

## Phase 2 — Monitor and Collect Runs

### Output locations

- **SLURM logs:** `sbatch_outputs/{run_name}.out`
- **Trajectories:** `runs/{dataset}/{retriever}/{split}/{model}/{run_name}/{query_id}.json`

For the full BCP split with Qwen3-Embedding-8B, a typical path is:

```
runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/planning_v4_seed0/Q001.json
```

### Monitoring

Check SLURM job status:

```bash
squeue -u $USER
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed
```

---

## Phase 3 — Find Missing IDs

Use `src_utils/find_missing_ids.py` to compare completed trajectory files against the reference query TSVs.

### Recursive scan (full split, all runs under a model directory)

```bash
python src_utils/find_missing_ids.py \
  --recursive \
  --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b \
  --reference_file "topics-qrels/bcp/bcp_10_shards/*"
```

This walks every leaf directory containing JSON files and reports which query IDs are missing from each shard TSV. Example output:

```
Found 15 leaf directories with JSON files

--- Checking runs/bcp/.../planning_v4_seed0 (78 query ids) ---
***Missing 2 query ids in topics-qrels/bcp/bcp_10_shards/q_3.tsv:***
 ['Q123', 'Q456']

--- Checking runs/bcp/.../seed0 (80 query ids) ---
All query files complete for runs/bcp/.../seed0
```

### Single-directory scan

```bash
python src_utils/find_missing_ids.py \
  --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed0 \
  --reference_file "topics-qrels/bcp/bcp_10_shards/*"
```
This only scans the specific directory (`seed0` in this case) provided.

### Scan against a single reference file (first50 split)

```bash
python src_utils/find_missing_ids.py \
  --input_dir runs/bcp/Qwen3-Embedding-8B/first50/gpt-oss-120b/seed0 \
  --reference_file "topics-qrels/bcp/queries_first50.tsv"
```
This only scans the specific directory (`seed0` in this case) provided, against first 50 queries of bcp.

---

## Phase 4 — Resubmit Missing Shards

### Option A: Manual

Read the output from Phase 3, then edit the `MISSING` dict in `submit_missing.py` by hand.

### Option B: Automated with `scripts/update_submit_missing.py`

Pipe the output of `find_missing_ids.py` directly into the bridging script:

```bash
# Pipe mode — prints a MISSING dict you can paste into submit_missing.py
python src_utils/find_missing_ids.py --recursive \
  --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b \
  --reference_file "topics-qrels/bcp/bcp_10_shards/*" \
  | python scripts/update_submit_missing.py

# Or from a saved file
python src_utils/find_missing_ids.py --recursive \
  --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b \
  --reference_file "topics-qrels/bcp/bcp_10_shards/*" \
  > missing_output.txt

python scripts/update_submit_missing.py --input missing_output.txt
```

Then paste the printed dict into `submit_missing.py` and submit:

```bash
python submit_missing.py --submit
```

(The `MISSING` and `MISSING_FIRST50` dicts are at the very top of `submit_missing.py`.)

---

## Phase 5 — Evaluate

Once all runs are complete, evaluate them with `scripts_evaluation/evaluate_run.py`. This uses Qwen3-32B as an LLM judge via local vLLM.

### Single run

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/planning_v4_seed0 \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --eval_dir evals/ \
  --tensor_parallel_size 1
```

For first50 runs, use the first50 ground truth file:

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/bcp/Qwen3-Embedding-8B/first50/gpt-oss-120b/planning_v4_seed0 \
  --ground_truth data/browsecomp_plus_decrypted_first50.jsonl \
  --eval_dir evals/ \
  --tensor_parallel_size 1
```

### Batch evaluation via SLURM

Edit `eval.SBATCH` to list the runs you want to evaluate, then submit:

```bash
sbatch eval.SBATCH
```

### Output

Each evaluation produces:

- `evals/{dataset}/{retriever}/{split}/{model}/{run_name}/evaluation_summary.json` — accuracy, recall, calibration error, tool call stats
- `evals/{dataset}/{retriever}/{split}/{model}/{run_name}/detailed_results.csv` — per-query breakdown

### Aggregate scores

```bash
python scripts_evaluation/aggregate_score.py evals/bcp/
```

This scans all `evaluation_summary.json` files under the root and writes `aggregated_scores.csv`.

---

## Quick Reference

| Step | Command |
|------|---------|
| Dry-run submission | `python submit_missing.py` |
| Submit to SLURM | `python submit_missing.py --submit` |
| Find missing (recursive) | `python src_utils/find_missing_ids.py -r --input_dir <dir> --reference_file "topics-qrels/bcp/bcp_10_shards/*"` |
| Auto-generate MISSING dict | `python src_utils/find_missing_ids.py -r --input_dir <dir> \| python scripts/update_submit_missing.py` |
| Evaluate a run | `python scripts_evaluation/evaluate_run.py --input_dir <run_dir> --tensor_parallel_size 1` |
| Aggregate scores | `python scripts_evaluation/aggregate_score.py evals/bcp/` |
