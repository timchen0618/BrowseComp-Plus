# `sft/` — Supervised Fine-Tuning

Train a base LLM (default: `Qwen/Qwen3-30B-A3B`) to produce good
search-tool calls using hand-selected good excerpts mined from prior agent
runs. Training framework is
[Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

---

## Directory layout

```
sft/
├── README.md                             # (this file)
├── axolotl/
│   ├── prepare_dataset.py                # selected-tool-calls -> Axolotl messages JSONL
│   ├── qwen3_30b_a3b_search_sft.yaml     # Axolotl training config
│   ├── run_axolotl.sh                    # end-to-end launcher
│   └── data/                             # generated train.jsonl / val.jsonl
└── checkpoints/                          # saved models (one subdir per run)
```

---

## Pipeline

```
selected_tool_calls/*.jsonl    runs/.../<source_file>.json
               \                    /
                v                  v
       sft/axolotl/prepare_dataset.py
                       |
                       v
       sft/axolotl/data/{train,val}.jsonl     (messages list per line)
                       |
                       v
            axolotl preprocess               (tokenize + cache)
                       |
                       v
      accelerate launch -m axolotl.cli.train (multi-GPU FSDP)
                       |
                       v
  sft/checkpoints/axolotl-qwen3-30b-a3b/     (LoRA adapter + tokenizer)
```

---

## Quick start

One-shot via the wrapper (requires two env vars):

```bash
INPUT=selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl \
TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    bash sft/axolotl/run_axolotl.sh
```

Or step by step:

```bash
# 1) Convert excerpts -> Axolotl messages JSONL
python sft/axolotl/prepare_dataset.py \
    --input selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl \
    --trajectory-folder runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    --output-dir sft/axolotl/data \
    --val-size 0.1 --seed 42

# 2) Tokenize + cache
axolotl preprocess sft/axolotl/qwen3_30b_a3b_search_sft.yaml

# 3) Train (multi-GPU via accelerate / FSDP)
accelerate launch -m axolotl.cli.train sft/axolotl/qwen3_30b_a3b_search_sft.yaml
```

Wrapper env vars: `INPUT`, `TRAJECTORY_FOLDER` (required); `DATA_DIR`,
`CONFIG`, `VAL_SIZE`, `SEED` (optional).

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
(`reasoning`, `function_call`, `function_call_output`) joined by blank
lines. `source_file` is resolved against `--trajectory-folder` to obtain
the original run JSON.

### What happens during conversion

1. Loads `<trajectory_folder>/<source_file>` and takes
   `original_messages[0]` verbatim as a single merged `user` turn
   (contains both system prompt and the `User: <question>` line).
2. Walks the excerpt items:
   - `reasoning` → accumulated into the current assistant buffer.
   - `function_call` → rendered as
     `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`,
     then the assistant buffer is flushed.
   - `function_call_output` → flushes any pending assistant, then emits
     a user turn wrapped in `<tool_response>...</tool_response>`.
3. Keeps the example iff at least one assistant turn contains
   `<tool_call>`. `<answer>` is **not** required — excerpts are single
   (or few) tool-call rounds meant to train tool-use.

Source trajectories are cached after first load, so repeated `source_file`
lookups are free even on large input files.

### Output shape (what Axolotl sees)

One JSON object per line under `sft/axolotl/data/`:

```json
{"messages": [
  {"role": "user",      "content": "<system+question merged>"},
  {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
  {"role": "user",      "content": "<tool_response>...</tool_response>"},
  {"role": "assistant", "content": "..."}
]}
```

The printed `summary:` line lists: `read`, `kept`, `dropped_schema`,
`dropped_missing_source`, `dropped_bad_excerpt`.

---

## Axolotl config highlights

Config lives at
[`sft/axolotl/qwen3_30b_a3b_search_sft.yaml`](axolotl/qwen3_30b_a3b_search_sft.yaml).
Settings worth knowing when you tweak runs:

| Setting | Value | Why |
|---|---|---|
| `base_model` | `Qwen/Qwen3-30B-A3B` | Qwen3 MoE — LoRA targets the expert linears automatically via `lora_target_linear: true`. |
| `chat_template` | `qwen3` | Pulls the tokenizer's Jinja template, which already renders `<tool_call>` / `<tool_response>` correctly. |
| `datasets[].type` | `chat_template` | Standard Axolotl type for role/content messages. |
| `roles_to_train` | `["assistant"]` | Only assistant tokens contribute to loss; user / tool turns get `label=-100`. |
| `train_on_eos` | `turn` | Model learns where each assistant turn ends — critical for respecting the tool-call budget. |
| `sample_packing` | `false` | Packing would concatenate unrelated trajectories into one sequence and leak supervision across tool-call boundaries. |
| `sequence_len` | `8192` | Enough to hold system + question + a handful of retrieved snippets. Raise if you see truncation warnings. |
| `adapter` | `lora` (r=16, α=32) | Parameter-efficient. Delete `adapter` + `lora_*` keys to do a full fine-tune. |
| `fsdp_transformer_layer_cls_to_wrap` | `Qwen3MoeDecoderLayer` | Correct wrap target for Qwen3-MoE; update when switching to another architecture. |
| `fsdp_use_orig_params` | `false` | Required for LoRA + FSDP. |
| `output_dir` | `sft/checkpoints/axolotl-qwen3-30b-a3b` | Change this per run so you don't clobber checkpoints. |

### Switching to a different base model

Update `base_model` and (if it's MoE) `fsdp_transformer_layer_cls_to_wrap`
to the matching decoder-layer class. If the new model's chat template
doesn't natively emit `<tool_call>` tokens, set `chat_template` to one
that does (`qwen3`, `llama3`, …) or supply a custom Jinja via
`chat_template_jinja`.

---

## Running on the cluster

Wrap `sft/axolotl/run_axolotl.sh` in the same SLURM template used for
other BrowseComp-Plus jobs (see `write_sbatch.py` at the repo root).
Typical resources: `2× A100`, `10 CPU`, `300 GB RAM`, `12–48 h`.

Env vars worth setting in the SBATCH preamble:

- `WANDB_API_KEY` — W&B logging
- `HF_TOKEN` — if the base model is gated
- `ACCELERATE_USE_FSDP=true` — belt-and-suspenders for FSDP detection

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `argparse: the following arguments are required: --input, --trajectory-folder` | Wrapper or script called without required inputs | Set `INPUT` and `TRAJECTORY_FOLDER`. |
| `--trajectory-folder not found or not a dir: ...` | Wrong path or file instead of folder | Pass the directory that directly contains the `run_*.json` source files. |
| `No usable examples found in ...` | All records failed schema / source-lookup / excerpt validation | Check the printed `dropped_*` counters. |
| `dropped_missing_source=N` with N>0 | `source_file` names don't resolve under `--trajectory-folder` | Double-check you're pointing at the seed directory where those run files actually live. |
| `dropped_bad_excerpt=N` with N>0 | Excerpt had no `function_call` item | Usually fine to ignore; such records are low-quality by construction. |
| OOM with Qwen3-30B-A3B | Activation/memory pressure | Lower `sequence_len`, keep `gradient_checkpointing` on, or switch to a smaller base. |
| `CheckpointError` about bf16 vs fp32 metadata | Gradient-checkpointing + bf16 quirk | Ensure `gradient_checkpointing_kwargs.use_reentrant: false` (already set in the YAML). |
| `assistant_masks` mismatch errors | Tokenizer's template doesn't match `chat_template: qwen3` | Set `chat_template` to match your model, or supply `chat_template_jinja`. |

---

## Data provenance

- `selected_tool_calls/*.jsonl` — produced by a tool-call selection
  stage that picks interesting rounds from existing runs and stores them
  as `{source_file, excerpt, ...}` records.
- `runs/bcp/<retriever>/full/<model>/<seed>/` — the matching source
  trajectories (`original_messages[0]` holds the system+question prompt
  we reuse during conversion).

---

## Outputs

- `sft/axolotl/data/train.jsonl`, `val.jsonl` — regenerable from the
  input JSONL; safe to delete.
- `sft/axolotl/prepared/` — Axolotl's tokenized dataset cache. Delete
  if you change `chat_template`, `sequence_len`, or dataset contents.
- `sft/checkpoints/axolotl-qwen3-30b-a3b/` — LoRA adapter, tokenizer,
  trainer state.
