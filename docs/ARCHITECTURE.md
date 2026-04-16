# Architecture & Data Reference

Technical reference for the data formats, directory layout, and scripts involved in agent runs. For the step-by-step operational workflow (submit, find missing, resubmit, evaluate), see [RUN_WORKFLOW.md](RUN_WORKFLOW.md).

---

## Directory Layout

```
runs/
└── {dataset}/                          # bcp, frames, musique
    └── {retriever}/                    # Qwen3-Embedding-8B, Qwen3-Embedding-0.6B
        └── {split}/                    # full (10 shards), first50 (single file)
            └── {model}/                # gpt-oss-120b, tongyi
                └── {run_name}/         # e.g. seed0, planning_v4_seed0, traj_ext_gpt-oss-120b_seed0
                    └── run_*.json      # one per query

evals/
└── (mirrors runs/ structure)
    └── {run_name}/
        ├── evaluation_summary.json
        └── detailed_results.csv

topics-qrels/
└── {dataset}/
    ├── queries.tsv                     # full query set (id \t query)
    ├── queries_first50.tsv
    └── {dataset}_10_shards/
        └── q_{0..9}.tsv                # per-shard query subsets

indexes/
└── Qwen3-Embedding-8B/
    └── corpus.shard*.pkl               # FAISS index shards

data/
├── browsecomp_plus_decrypted.jsonl     # BCP ground truth
└── frames_ground_truth.jsonl           # FRAMES ground truth
```

---

## Query Files (TSV)

Tab-separated, no header. Each line is `query_id\tquery_text`:

```
262     What is the name of the algorithm used for ...
600     Which paper first proposed ...
```

The **full** split uses 10 shard files (`q_0.tsv` through `q_9.tsv`) under `{dataset}_10_shards/`. The **first50** split uses a single `queries_first50.tsv`.

---

## Trajectory JSON (agent output)

Each query produces one `run_*.json` file. Structure:

```json
{
  "metadata": {
    "model": "openai/gpt-oss-120b",
    "reasoning": {"effort": "high", "summary": "detailed"},
    "output_dir": "/scratch/.../seed0",
    "planning": true,
    "planning_trigger": "traj_summary_ext",
    "plan_text": "...",
    "plan_text_history": [{"source": "...", "plan": "...", "iteration": 0}]
  },
  "query_id": "262",
  "tool_call_counts": {"search": 5, "get_document": 2},
  "status": "completed",
  "retrieved_docids": ["12345", "67890"],
  "result": [
    {"type": "reasoning",  "tool_name": null, "arguments": null, "output": ["thinking..."]},
    {"type": "tool_call",  "tool_name": "local_knowledge_base_retrieval",
                           "arguments": "{\"user_query\": \"...\"}",
                           "output": "[{\"docid\": \"12345\", \"score\": 0.9, \"snippet\": \"...\"}]"},
    {"type": "output_text","tool_name": null, "arguments": null,
                           "output": "Explanation: ...\nExact Answer: ...\nConfidence: 85%"}
  ]
}
```

Key fields:
- `status`: `"completed"` (model stopped naturally) or `"incomplete"` (hit `max_iterations`)
- `result`: ordered list of reasoning steps, tool calls, and final output
- `retrieved_docids`: aggregated set of all docids returned by search calls
- `metadata.plan_text_history`: snapshots of plans injected/revised during the run (when planning is enabled)

---

## Ground Truth (JSONL)

One JSON object per line in `data/browsecomp_plus_decrypted.jsonl`:

```json
{
  "query_id": "262",
  "query": "What is the name of ...",
  "answer": "The XYZ algorithm",
  "gold_docs": ["12345", "67890"],
  "evidence_docs": ["12345"]
}
```

---

## Core Scripts

### `search_agent/oss_client.py`

The main agent execution script for OpenAI-compatible models (gpt-oss-120b, gpt-oss-20b) served via vLLM. This is the entry point called by the SBATCH scripts.

**High-level flow:**

```
TSV queries
    │
    ▼
_process_tsv_dataset()
    │  - reads TSV, skips already-processed query_ids
    │  - loads plans / trajectories / summaries based on --planning-trigger
    │  - dispatches queries via ThreadPoolExecutor (--num-threads)
    │
    ▼
_handle_single_query()              (per query, possibly concurrent)
    │  - formats the prompt via prompts.py templates
    │  - injects plan/trajectory if planning is enabled
    │  - builds the initial_request dict for the Responses API
    │
    ▼
run_conversation_with_tools()       (the agent loop)
    │  - sends request to vLLM via openai.responses.create()
    │  - if model returns function_call(s): execute via SearchToolHandler
    │  - optionally injects mid-conversation plan revisions
    │  - optionally re-injects plan reminders every N iterations
    │  - loops until model produces a final text output or max_iterations
    │
    ▼
_persist_response()
    │  - normalizes messages into the result[] format
    │  - extracts retrieved docids
    │  - writes run_{timestamp}.json
```

**Key classes and functions:**

| Symbol | Purpose |
|--------|---------|
| `SearchToolHandler` | Wraps a `BaseSearcher` into tool definitions and execution. Registers `local_knowledge_base_retrieval` and optionally `get_document`. |
| `run_conversation_with_tools()` | The main agent loop. Handles tool dispatch, plan injection/reinject/revise logic. |
| `_build_trajectory_user_content()` | Builds the user prompt when using trajectory-based triggers (`traj_ext`, `traj_summary_ext`, etc.). |
| `_persist_response()` | Normalizes the raw Responses API messages into the canonical `result[]` format and writes JSON. |
| `call_planner()` / `call_planner_mid()` | Call the planner model to generate or revise a plan. |

### `search_agent/prompts.py`

All prompt templates. The main ones:

| Template | Used when |
|----------|-----------|
| `QUERY_TEMPLATE` | Default: search + get_document tools |
| `QUERY_TEMPLATE_NO_GET_DOCUMENT` | Search-only (no get_document) |
| `QUERY_TEMPLATE_GIVEN_TRAJECTORY` | `traj_ext` / `traj_orig_ext` triggers |
| `QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY` | `traj_summary_ext` / `traj_summary_orig_ext` triggers |

### `search_agent/planning_utils.py`

Shared planning helpers:

- `parse_plan_from_response()` — extracts text between `<plan>...</plan>` tags
- `inject_plan_into_messages()` — returns the assistant+user message pair that injects a plan
- `load_and_validate_plans()` — loads pre-generated plans from a JSONL file
- `run_planner_with_retries()` — retry wrapper for planner API calls

### `search_agent/trajectory_utils.py`

Shared trajectory loading, formatting, and summarization:

- `_load_and_validate_trajectories()` — loads JSON trajectory files from a directory
- `_load_trajectory_summaries()` — loads pre-computed summaries from a JSONL file
- `_format_trajectory_for_prompt()` — formats a trajectory dict into a text block for the prompt
- `_format_original_messages_for_prompt_oss()` — formats raw messages for OSS client (for `_orig_ext` variants)
- `_format_original_messages_for_prompt_tongyi()` — formats raw messages for Tongyi client (for `_orig_ext` variants)
- `call_trajectory_summarizer()` — calls the LLM to summarize a trajectory on the fly

### `search_agent/utils.py`

- `extract_retrieved_docids_from_result()` — scans `result[]` for all docids returned by search tool calls

### `searcher/searchers/`

Pluggable retrieval backends. Selected via `--searcher-type`.

| Class | CLI name | Backend |
|-------|----------|---------|
| `FaissSearcher` | `faiss` | Dense retrieval with Qwen3-Embedding models. Loads pre-built `.pkl` index shards. |
| `BM25Searcher` | `bm25` | Pyserini BM25 sparse retrieval. |
| `CustomSearcher` | `custom` | Placeholder for custom implementations. |

All implement `BaseSearcher` with `search(query, k)` and `get_document(docid)`.

---

## Run Modes (planning triggers)

The `--planning-trigger` flag controls how the agent is augmented before or during execution.

| Trigger | What happens | Required flags |
|---------|-------------|----------------|
| (none / `org`) | Baseline: no planning, plain query prompt | — |
| `start` | Call planner at start, inject plan into messages | `--planning`, `--plan-prompt-file` |
| `start_ext` | Load pre-generated plan from JSONL, inject at start | `--planning`, `--planning-file` |
| `after_steps` | Call planner after N tool calls | `--planning`, `--planning-steps` |
| `start_and_after_steps` | Both start + after_steps | `--planning`, `--plan-prompt-file`, `--planning-steps` |
| `traj_ext` | Prepend a prior trajectory to the prompt | `--planning`, `--trajectory-dir`, `--query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY` |
| `traj_orig_ext` | Like `traj_ext` but uses original (raw) messages | Same as `traj_ext` |
| `traj_summary_ext` | Summarize a prior trajectory, prepend summary | `--planning`, `--trajectory-dir` or `--trajectory-summary-file`, `--query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY` |
| `traj_summary_orig_ext` | Like `traj_summary_ext` but from original messages | Same as `traj_summary_ext` |

Additional modifiers (applied on top of any trigger):

| Modifier | Flag | Effect |
|----------|------|--------|
| Plan reinject | `--plan-reinject-every N` | Re-inject the initial plan text as a user reminder every N iterations |
| Plan revise | `--plan-revise-every N` | Call the planner to produce a revised plan every N iterations |

---

## SBATCH Templates

### `run_qwen3_planning.SBATCH` (full split)

1. Starts a vLLM server for the agent model (runs in background)
2. Waits 120s for server startup
3. Runs `search_agent/oss_client.py` (or `tongyi_client.py`) with the shard query file `q_${SLURM_ARRAY_TASK_ID}.tsv`
4. The `mode` variable is parsed into the correct `--planning-trigger`, `--trajectory-dir`, and other flags via a `case` block

### `run_qwen3_first50.SBATCH` (first50 split)

Same structure but no `--array` directive. Queries come from `queries_first50.tsv`.

### `submit_missing.py`

Not an SBATCH template itself, but a Python script that:
1. Reads a template SBATCH file
2. For each entry in `MISSING` / `MISSING_FIRST50` / etc., patches the template with the correct `--array`, `MODEL_NAME`, `mode`, `seed`, and `dataset`
3. Writes a temp file and submits it via `sbatch`

The `parse_run_name()` function maps a run name key back to `(model, mode, seed)`:
```
"gpt-oss-120b_planning_v4_seed0"  →  model="gpt-oss-120b", mode="planning_v4", seed=0
"gpt-oss-120b_traj_ext_gpt-oss-120b_seed0"  →  model="gpt-oss-120b", mode="traj_ext", seed=0
```

---

## Evaluation Scripts

### `scripts_evaluation/evaluate_run.py`

LLM-as-judge evaluation using Qwen3-32B via local vLLM. For each trajectory JSON:
1. Extracts the final answer from the `result[]` array
2. Builds a judge prompt with question, response, and correct answer
3. Batch-infers judgments via vLLM
4. Writes per-query eval JSONs and `evaluation_summary.json`

### `scripts_evaluation/aggregate_score.py`

Scans a directory tree for `evaluation_summary.json` files and produces a single CSV with accuracy, recall, and token-limit stats per run.
