# `sft/` — Supervised Fine-Tuning

Train a base LLM on search-tool-call excerpts mined from prior agent runs.
The primary training stack is [Axolotl](https://github.com/axolotl-ai-cloud/axolotl);
an alternative [`sft/hf_trainer/`](hf_trainer/) path uses Hugging Face `Trainer` + PEFT
on the same `{"messages": ...}` JSONL.

---

## Directory layout

```
sft/
├── README.md                                     # (this file)
├── axolotl/
│   ├── prepare_dataset.py                        # selected-tool-calls → messages JSONL
│   ├── qwen3_30b_a3b_search_sft.yaml             # Axolotl config for Qwen3-30B-A3B
│   ├── qwen3.5_4b_search_sft.yaml                # Axolotl config for Qwen3.5-4B
│   ├── run_axolotl.sh                            # end-to-end Axolotl launcher
│   ├── multi_input_config_best_of_random_selection.json  # multi-input config example
│   ├── data_process.sh                           # convenience wrapper for multi-input mode c
│   ├── match_sft.py                              # combine gpt-oss + qwen converted messages
│   ├── upload_sft_diff.py                        # push sft_diff_combined.jsonl to HF Hub
│   └── data/
│       ├── raw/
│       │   └── {run_name}/                       # one subdir per dataset variant
│       │       ├── train.jsonl
│       │       └── val.jsonl
│       └── prepared/
│           └── {run_name}/                       # Axolotl tokenized cache per variant
└── checkpoints/                                  # saved adapters (one subdir per run)
```

---

## Pipeline

```
selected_tool_calls/*.jsonl    runs/.../<source_file>.json
               \                    /
                v                  v
       sft/axolotl/prepare_dataset.py
         (--template gpt-oss | qwen)
         (--split bcp-train680-test150 | random)
         (--multi-input <config.json> --mode {a,b,c,d})
                       |
                       v
       sft/axolotl/data/raw/{run_name}/{train,val}.jsonl
                       |
                       v
            axolotl preprocess               (tokenize + cache → data/prepared/{run_name}/)
                       |
                       v
      accelerate launch -m axolotl.cli.train (multi-GPU FSDP)
                       |
                       v
  sft/checkpoints/${RUN_NAME}/                   (LoRA adapter + tokenizer)
```

---

## Quick start

### Running via SBATCH (cluster)

`sft_train.SBATCH` at the repo root wraps the full three-stage pipeline in a single SLURM job (2× H200 GPU, 5 h wall time). It picks single-input or multi-input mode based on whether `MULTI_INPUT` is set, then runs data preparation → Axolotl preprocess → training in sequence.

**Key variables** (all overridable via `sbatch --export=ALL,...`):

| Variable | Default | Purpose |
|---|---|---|
| `RUN_NAME` | `random_selection` | Names the data and checkpoint subdirectories |
| `CONFIG` | `sft/axolotl/qwen3.5_4b_search_sft.yaml` | Axolotl YAML; swap to the `…30b…yaml` for 30B |
| `INPUT` | `selected_tool_calls/all/gpt-oss-120b/seed0/…random_seed42.jsonl` | Selected-tool-calls JSONL (single-input mode only) |
| `TRAJECTORY_FOLDER` | `runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4` | Source trajectory dir (single-input mode only) |
| `EVAL_FOLDER` | `evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4` | Eval results dir for success filtering (single-input mode only) |
| `MULTI_INPUT` | *(empty)* | Path to multi-input JSON config; non-empty activates multi-input mode |
| `MODE` | `a` | Multi-input selection mode (`a`/`b`/`c`/`d`) |
| `SPLIT` | `bcp-train680-test150` | Train/val split strategy |
| `SEED` | `42` | Random seed for splitting and selection |

`DATA_DIR` and `PREPARED_DIR` are auto-derived from `RUN_NAME` (`sft/axolotl/data/raw/${RUN_NAME}` and `…/prepared/${RUN_NAME}`). Checkpoints land in `sft/checkpoints/${RUN_NAME}/`.

**Submit commands:**

```bash
# Single-input mode (uses all defaults above)
sbatch sft_train.SBATCH

# Single-input mode — override paths without editing the file
sbatch --export=ALL,INPUT=path/to/selected.jsonl,TRAJECTORY_FOLDER=runs/...,EVAL_FOLDER=evals/...,RUN_NAME=my_run \
    sft_train.SBATCH

# Multi-input mode (mode d — skip query_ids with no successful run)
sbatch -J sft_mode_d \
    --export=ALL,MULTI_INPUT=sft/axolotl/multi_input_config_best_of_random_selection.json,MODE=d,RUN_NAME=best_of_4_random_selection_mode_d \
    sft_train.SBATCH
```

Inside the job, three stages run in sequence — described in the subsections below.

---

### Single-input mode

Takes one selected-tool-calls JSONL, filters by per-trajectory success via `--eval-folder`, and trains.

**Stage 1 — Data preparation:** Convert selected tool calls to `messages` JSONL.

```bash
python sft/axolotl/prepare_dataset.py \
    --input selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl \
    --trajectory-folder runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    --eval-folder evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    --output-dir sft/axolotl/data/raw/gemini_2.5_pro_selection \
    --template qwen \
    --split bcp-train680-test150 \
    --seed 42
```

Reads each record from `--input`, loads the corresponding trajectory from `--trajectory-folder`, and filters to successful trajectories using eval results from `--eval-folder`. Writes `train.jsonl` and `val.jsonl` to `--output-dir`. Pass `--keep-failed` to also include unsuccessful trajectories.

**Stage 2 — Axolotl preprocess:** Tokenize and cache the dataset.

```bash
DATA_DIR=sft/axolotl/data/raw/gemini_2.5_pro_selection \
PREPARED_DIR=sft/axolotl/data/prepared/gemini_2.5_pro_selection \
RUN_NAME=qwen3.5-4b-sft-gemini-selection \
    axolotl preprocess sft/axolotl/qwen3.5_4b_search_sft.yaml
```

Reads from `DATA_DIR`, tokenizes using the chat template and sequence length defined in the YAML, and writes the binary cache to `PREPARED_DIR`. Re-run this if you change `chat_template`, `sequence_len`, or dataset contents.

**Stage 3 — SFT training:** Launch multi-GPU FSDP training via Accelerate.

```bash
DATA_DIR=sft/axolotl/data/raw/gemini_2.5_pro_selection \
PREPARED_DIR=sft/axolotl/data/prepared/gemini_2.5_pro_selection \
RUN_NAME=qwen3.5-4b-sft-gemini-selection \
    accelerate launch -m axolotl.cli.train sft/axolotl/qwen3.5_4b_search_sft.yaml
```

Loads the tokenized cache from `PREPARED_DIR` and runs LoRA fine-tuning. The adapter and tokenizer are saved to `sft/checkpoints/${RUN_NAME}/`.

### Multi-input mode

Aggregates candidates from multiple tool-call runs and selects one per `query_id`
according to `--mode`. Provide a JSON config listing input specs (see below).

**Stage 1 — Data preparation:** Aggregate and select across multiple input runs.

```bash
# data_process.sh wraps the most common call (mode c, qwen template, bcp split)
bash sft/axolotl/data_process.sh
```

Or manually:

```bash
python sft/axolotl/prepare_dataset.py \
    --multi-input sft/axolotl/multi_input_config_best_of_random_selection.json \
    --mode d \
    --output-dir sft/axolotl/data/raw/best_of_4_random_selection_mode_d \
    --template qwen \
    --split bcp-train680-test150 \
    --seed 42
```

Reads each entry in the config JSON, pools candidates across all listed runs, applies the selection strategy for the chosen mode (e.g. mode d picks the shortest successful run per `query_id`, skipping query IDs with no successful run), and writes `train.jsonl` and `val.jsonl` to `--output-dir`.

**Stage 2 — Axolotl preprocess:** Tokenize and cache (same as single-input mode).

```bash
DATA_DIR=sft/axolotl/data/raw/best_of_4_random_selection_mode_d \
PREPARED_DIR=sft/axolotl/data/prepared/best_of_4_random_selection_mode_d \
RUN_NAME=best_of_4_random_selection_mode_d \
    axolotl preprocess sft/axolotl/qwen3.5_4b_search_sft.yaml
```

**Stage 3 — SFT training:** Launch multi-GPU FSDP training (same as single-input mode).

```bash
DATA_DIR=sft/axolotl/data/raw/best_of_4_random_selection_mode_d \
PREPARED_DIR=sft/axolotl/data/prepared/best_of_4_random_selection_mode_d \
RUN_NAME=best_of_4_random_selection_mode_d \
    accelerate launch -m axolotl.cli.train sft/axolotl/qwen3.5_4b_search_sft.yaml
```

---

## `prepare_dataset.py` reference

### Modes

| Flag | Description |
|---|---|
| `--input` + `--trajectory-folder` + `--eval-folder` | **Single-input mode**: one JSONL, one source folder, filter by eval success. |
| `--multi-input <config.json>` + `--mode {a,b,c,d}` | **Multi-input mode**: aggregate candidates across multiple runs, then select. |

`--multi-input` and `--input` are mutually exclusive.

### Multi-input config format

```json
[
  {
    "input": "selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_....jsonl",
    "trajectory_folder": "runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4",
    "subsequent_folder": "runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/subsequent_seed0/",
    "subsequent_eval_folder": "evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/subsequent_seed0/"
  },
  ...
]
```

`subsequent_eval_folder` is optional; if absent all candidates are treated as unsuccessful.

### Multi-input selection modes

| Mode | Behavior |
|---|---|
| `a` | One per `query_id` — shortest successful run; if none succeed, shortest run overall. |
| `b` | All successful runs per `query_id`; if none succeed, exactly one at random. |
| `c` | One per `query_id` — shortest successful run; if none succeed, exactly one at random. |
| `d` | One per `query_id` — shortest successful run; skip `query_id` entirely if none succeed. |

### Output templates

| `--template` | Tool name | Arg key | Reasoning format |
|---|---|---|---|
| `gpt-oss` | `local_knowledge_base_retrieval` | `user_query` | plain text |
| `qwen` (default) | `search` | `query` | `<think>...</think>` wrapping |

Use `--template qwen` (`gpt-oss` is legacy).

### Train/val split

| `--split` | Behavior |
|---|---|
| `bcp-train680-test150` (default) | Assigns examples by fixed BCP split: train→680 query IDs, val→150 query IDs (seed 42, `scripts/split_bcp_test150.py`). |
| `random` | Shuffles and holds out `--val-size` fraction (default 0.1). |

### Summary line fields

**Single-input:** `read`, `kept`, `dropped_not_success`, `dropped_schema`, `dropped_missing_source`, `dropped_bad_excerpt`

**Multi-input:** `total_query_ids`, `query_ids_with_success`, `selected_examples`, `dropped_schema`, `dropped_missing_source`, `dropped_bad_excerpt` (+ `query_ids_skipped_no_success` for mode d)

---

## Axolotl configs

Both configs use `${DATA_DIR}`, `${PREPARED_DIR}`, and `${RUN_NAME}` env vars so
multiple dataset variants can share the same YAML without editing.

### Qwen3-30B-A3B (`qwen3_30b_a3b_search_sft.yaml`)

| Setting | Value | Why |
|---|---|---|
| `base_model` | `Qwen/Qwen3-30B-A3B` | Qwen3 MoE — LoRA targets all expert linears via `lora_target_linear: true`. |
| `chat_template` | `qwen3` | Pulls the tokenizer's Jinja template; renders `<tool_call>` / `<tool_response>` correctly. |
| `roles_to_train` | `["assistant"]` | Only assistant tokens contribute to loss. |
| `train_on_eos` | `turn` | Model learns where each assistant turn ends. |
| `sample_packing` | `false` | Packing leaks supervision across tool-call boundaries. |
| `sequence_len` | `4096` | Covers system + question + a handful of retrieved snippets. |
| `num_epochs` | `3` | — |
| `adapter` | `lora` (r=16, α=32) | Parameter-efficient. |
| `fsdp_transformer_layer_cls_to_wrap` | `Qwen3MoeDecoderLayer` | Correct wrap target for Qwen3-MoE. |
| `fsdp_config.activation_checkpointing` | `true` | MoE activation checkpointing via fsdp_config (not HF gradient_checkpointing). |
| `output_dir` | `sft/checkpoints/${RUN_NAME}` | Set `RUN_NAME` env var per run. |

### Qwen3.5-4B (`qwen3.5_4b_search_sft.yaml`)

| Setting | Value | Why |
|---|---|---|
| `base_model` | `Qwen/Qwen3.5-4B` | Smaller dense model — faster iteration. |
| `chat_template` | `qwen3` | Same as 30B. |
| `sequence_len` | `32768` | Longer context for full retrieval snippets. |
| `num_epochs` | `10` | More passes to compensate for smaller model capacity. |
| `gradient_checkpointing` | `true` (HF) | Dense model — use HF gradient_checkpointing; fsdp activation_checkpointing is off. |
| `fsdp_transformer_layer_cls_to_wrap` | `Qwen3_5DecoderLayer` | Dense decoder class (not MoE). |
| `fsdp_offload_params` | `true` | Offload params to CPU to reduce VRAM pressure. |
| `output_dir` | `sft/checkpoints/${RUN_NAME}` | Set `RUN_NAME` env var per run. |

### Switching models

Update `base_model` and `fsdp_transformer_layer_cls_to_wrap` to the matching decoder-layer class.
If the new model's chat template doesn't natively emit `<tool_call>` tokens, set `chat_template`
accordingly or supply a custom Jinja via `chat_template_jinja`.

---

## Utility scripts

### `match_sft.py`

One-off script that cross-references original excerpt records with their converted
`messages` counterparts from both gpt-oss and qwen templates. Outputs a combined JSONL
with fields `query_id`, `excerpt`, `messages_json` (gpt-oss), `messages_json_qwen`.

```bash
python sft/axolotl/match_sft.py
# → sft/axolotl/data/raw/data_qwen/sft_diff_combined.jsonl
```

Paths are hardcoded at the top of the file; edit before running.

### `upload_sft_diff.py`

Uploads `sft_diff_combined.jsonl` to HuggingFace Hub as a dataset.

```bash
python sft/axolotl/upload_sft_diff.py
# → pushes to timchen0618/browsecomp-plus-sft-diff-v1
```

Deletes and recreates the repo for a clean slate on each run.

---

## Input data format

`prepare_dataset.py` consumes **selected-tool-calls JSONL** files. Each line:

```json
{
  "source_file": "run_XXXXXXXXXXXXZ.json",
  "excerpt": "<JSON item 1>\n\n<JSON item 2>\n\n...",
  "selected_indices": [2],
  "...": "other metadata"
}
```

Where `excerpt` is a sequence of OpenAI Responses-API items
(`reasoning`, `function_call`, `function_call_output`) joined by blank lines.
`source_file` is resolved against `--trajectory-folder` to obtain the original run JSON.

### What happens during conversion

1. Loads `<trajectory_folder>/<source_file>` and takes
   `original_messages[0]` verbatim as a single merged `user` turn
   (contains both system prompt and the `User: <question>` line).
2. Walks the excerpt items:
   - `reasoning` → accumulated into the current assistant buffer (wrapped in `<think>...</think>` for `--template qwen`).
   - `function_call` → rendered as `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`,
     then the assistant buffer is flushed. Tool name/arg key are rewritten for `--template qwen`.
   - `function_call_output` → flushes any pending assistant, then emits
     a user turn wrapped in `<tool_response>...</tool_response>`.
3. Keeps the example iff at least one assistant turn contains `<tool_call>`.

Source trajectories are cached after first load.

### Output shape

One JSON object per line under `sft/axolotl/data/raw/{run_name}/`:

```json
{"messages": [
  {"role": "user",      "content": "<system+question merged>"},
  {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
  {"role": "user",      "content": "<tool_response>...</tool_response>"},
  {"role": "assistant", "content": "..."}
]}
```

---

## Running on the cluster

Set `DATA_DIR`, `PREPARED_DIR`, and `RUN_NAME` in the SBATCH preamble, then run
`axolotl preprocess` and `accelerate launch` as above. See `write_sbatch.py` at the
repo root for the standard SLURM template. Typical resources: `2× A100`, `10 CPU`,
`300 GB RAM`, `12–48 h`.

Additional env vars:

- `WANDB_API_KEY` — W&B logging
- `HF_TOKEN` — if the base model is gated
- `ACCELERATE_USE_FSDP=true` — belt-and-suspenders for FSDP detection

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `--trajectory-folder is required with --input` | Single-input mode called without required args | Pass `--trajectory-folder` and `--eval-folder`. |
| `--eval-folder is required with --input` | `--eval-folder` missing in single-input mode | Point at a directory of `*_eval.json` files for the run. |
| `--mode is required with --multi-input` | Multi-input called without mode | Add `--mode {a,b,c,d}`. |
| `query_id not found in --eval-folder` | Record's `query_id` has no matching `*_eval.json` | Ensure eval folder covers all records in `--input`. |
| `No usable examples found` | All records failed validation | Check `dropped_*` counters; verify folder paths and JSONL format. |
| `dropped_missing_source=N` with N>0 | `source_file` names don't resolve under `--trajectory-folder` | Confirm you're pointing at the seed dir where those run files live. |
| `dropped_bad_excerpt=N` with N>0 | Excerpt had no `function_call` item | Usually safe to ignore; those records are low-quality. |
| OOM with Qwen3-30B-A3B | Memory pressure | Keep `gradient_checkpointing: false` + `fsdp_config.activation_checkpointing: true` (not both). |
| OOM with Qwen3.5-4B | Memory pressure | Enable `fsdp_offload_params: true` (already set); lower `sequence_len` if needed. |
| `CheckpointError` bf16 vs fp32 | HF gradient_checkpointing + FSDP conflict | Use `gradient_checkpointing_kwargs.use_reentrant: false` (already set in 4B YAML). |
| `assistant_masks` mismatch | Tokenizer template mismatch | Set `chat_template` to match your model, or supply `chat_template_jinja`. |

---

## Outputs

- `sft/axolotl/data/raw/{run_name}/train.jsonl`, `val.jsonl` — regenerable from input; safe to delete.
- `sft/axolotl/data/prepared/{run_name}/` — Axolotl tokenized cache. Delete if you change `chat_template`, `sequence_len`, or dataset contents.
- `sft/checkpoints/${RUN_NAME}/` — LoRA adapter, tokenizer, trainer state.
---

## Data provenance

- `selected_tool_calls/*.jsonl` — tool-call selection stage that picks interesting rounds from existing runs.
- `runs/bcp/<retriever>/full/<model>/<seed>/` — matching source trajectories (`original_messages[0]` holds the system+question prompt reused during conversion).
- `selected_tool_calls/all/gpt-oss-120b/seed0/` — per-seed random-selection JSONL files used by `multi_input_config_best_of_random_selection.json`.
