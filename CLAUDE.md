# CLAUDE.md — BrowseComp-Plus

## Project Overview

**BrowseComp-Plus** is a research platform for evaluating AI agents on complex information retrieval and multi-hop reasoning tasks. It supports multiple LLM providers (OpenAI, Anthropic, Gemini, Qwen, OSS), multiple retrieval backends (BM25, FAISS, custom), and provides a full evaluation pipeline using LLM-as-judge.

---

## General Rules

- Before editing files, confirm which files are in scope. Never edit files the user hasn't mentioned or has explicitly excluded (e.g., model.py, finetuning.py).
- Prefer reusing existing functions and modifying existing files over creating new ones, unless explicitly asked to create something new.

---

## Repository Structure

```
BrowseComp-Plus/
├── search_agent/           # LLM client implementations (one per provider)
├── searcher/               # Retrieval backends and MCP server
│   └── searchers/          # BaseSearcher + BM25/FAISS/Custom implementations
├── src_utils/              # Data loading, filtering, and preprocessing utilities
├── scripts_evaluation/     # Evaluation pipeline (grading, metrics, dedup)
├── scripts_build_index/    # Index construction scripts
├── data/                   # Ground truth and decrypted run files
│   ├── browsecomp_plus_decrypted.jsonl            # BCP ground truth (answers + gold docs)
│   └── decrypted_run_files/{retriever}/{model}/   # Agent trajectories
├── indexes/                # Prebuilt BM25 and FAISS indexes
├── runs/                   # All agent trajectories: runs/{dataset}/{retriever}/{split}/{run_name}/
├── evals/                  # All evaluation results: evals/{dataset}/{retriever}/{split}/{run_name}/
├── docs/                   # Integration guides per provider
├── parametric/             # Parametric experiment configs
├── qampari_experiments/    # QAMPARI benchmark scripts
├── topics-qrels/           # Relevance judgments
├── figures/                # Generated charts
├── compute_repeats.py      # Query repetition analysis (Jaccard similarity)
├── compute_repeats_vllm.py # Query repetition analysis (LLM-based)
├── query_grader.py         # Score search queries using vLLM
├── trace_viewer.py         # Streamlit UI for browsing agent traces
├── find_missing_ids.py     # Find query IDs missing from output
├── write_sbatch.py         # Generate SLURM SBATCH scripts from templates
└── pyproject.toml          # Minimal project config (only tqdm required)
```

---

## File Operations

- When reorganizing or renaming directories, always ask the user for the explicit naming convention and full path structure (including all subdirectory levels like {run_name}) before executing bulk operations.

---

## Project Structure

- When working with this project's eval results, the directory structure is: `evals/{dataset}/{retriever}/{split}/{run_name}/`. Always verify the correct directory path with the user before scanning for results.

---

## Data Analysis

- When the user asks for analysis or summaries, do not include categories or data the user explicitly asked to exclude. Ask for clarification rather than guessing what to include.

---

## Key Architecture Concepts

### Agent Execution Flow

```
Query JSONL → Prompt Template → LLM API (with tool defs) → Tool Calls →
BaseSearcher (BM25/FAISS/Custom) → Results → LLM → ... → Final Answer →
Trajectory JSON → Evaluation (Qwen3-32B judge)
```

### Search Tool Architecture

```
LLM API Request (tool definitions: search, get_document)
    ↓
Tool Execution Handler (in each client)
    ↓
BaseSearcher.search(query, k) → [{"docid", "score", "text"}, ...]
BaseSearcher.get_document(docid) → {"docid", "text"}
```

### Output Directory Convention

All runs live under `runs/`, organized as `runs/{dataset}/{retriever}/{split}/{agent_model}/`:

| Dataset folder | Benchmark |
|----------------|-----------|
| `bcp/` | BrowseComp-Plus |
| `frames/` | FRAMES benchmark |

| Split folder | Queries |
|--------------|---------|
| `full/` | All queries |
| `first_50/` | First 50 queries |
| `first_100/` | First 100 queries |

| Agent model folder | Examples |
|--------------------|---------|
| `tongyi/` | Tongyi-DeepResearch |
| `gpt-oss-120b/` | GPT-4 OSS 120B |
| (other model names) | Named by model/run identifier |

```
runs/
└── {dataset}/              # e.g., bcp, frames
    └── {retriever}/        # e.g., Qwen3-Embedding-8B
        └── {split}/        # e.g., full, first_50, first_100
            └── {agent_model}/  # e.g., tongyi, gpt-oss-120b
                └── {run_name}/ # e.g., planning_v8_prompt_seed0
                    └── {query_id}.json

evals/
└── {dataset}/              # e.g., bcp, frames
    └── {retriever}/        # e.g., Qwen3-Embedding-8B
        └── {split}/        # e.g., full, first_50, first_100
            └── {agent_model}/  # mirrors runs/ structure
                └── {run_name}/
                    └── eval results
```

---

## Key Files

### search_agent/ — LLM Clients

| File | Provider | Notes |
|------|----------|-------|
| `openai_client.py` | OpenAI | GPT-4, O3, O4-mini; most complete reference |
| `anthropic_client.py` | Anthropic | Uses MCP server + ngrok tunnel |
| `gemini_client.py` | Google | ThreadPoolExecutor for concurrency |
| `oss_client.py` | vLLM | Open-source models via local vLLM |
| `qwen_client.py` | Alibaba | Dashscope API |
| `tongyi_client.py` | Alibaba | Tongyi-DeepResearch-30B, vLLM port 6008 |
| `search_r1_client.py` | Search-R1 | Custom streaming parser |
| `prompts.py` | — | All prompt templates |
| `utils.py` | — | docid extraction utilities |

### searcher/ — Retrieval Backends

| File | Purpose |
|------|---------|
| `searchers/base.py` | Abstract `BaseSearcher` — implement this for custom retrievers |
| `searchers/bm25_searcher.py` | Pyserini-based BM25 |
| `searchers/faiss_searcher.py` | Dense FAISS (Qwen3-Embedding 0.6B/4B/8B) |
| `searchers/custom_searcher.py` | Placeholder for custom implementations |
| `mcp_server.py` | FastMCP server exposing search/get_document tools |
| `tools.py` | MCP tool registration |

### scripts_evaluation/

| File | Purpose |
|------|---------|
| `evaluate_run.py` | Main grader — Qwen3-32B as judge, batch vLLM inference |
| `deduplicate_trajectories.py` | Remove duplicate query_id entries |
| `compute_pass_k.py` | pass@k metrics |
| `aggregate_score.py` | Aggregate scores across runs |

### src_utils/

| File | Purpose |
|------|---------|
| `load_trajectories.py` | Core JSONL loading and file discovery |
| `filter_by_query_ids.py` | Subset by query IDs |
| `add_search_counts.py` | Augment trajectories with search stats |
| `shard.py` | Partition large datasets |
| `combine_json_to_jsonl.py` | JSON → JSONL batch conversion |

---

## Data Formats

### Trajectory JSONL (agent output)

```json
{
  "query_id": "Q001",
  "query": "...",
  "search_counts": 3,
  "status": "completed",
  "result": [
    {"type": "reasoning", "output": ["thinking text..."]},
    {"type": "tool_call", "tool_name": "search",
     "arguments": {"query": "..."}, "output": [{"docid": "...", "score": 0.9, "text": "..."}]}
  ]
}
```

### Queries (topics-qrels/)

Queries are stored as TSV files (`{query_id}\t{query_text}`):

```
topics-qrels/
├── bcp/
│   ├── queries.tsv               # Full query set
│   ├── queries_first10.tsv
│   ├── queries_first50.tsv
│   ├── queries_first100.tsv
│   ├── queries_last730.tsv
│   ├── qrel_golds.txt            # Gold relevance judgments (TREC format)
│   ├── qrel_evidence.txt         # Evidence relevance judgments (TREC format)
│   └── bcp_10_shards/q_{0-9}.tsv
├── frames/
│   ├── queries.tsv               # Full query set
│   ├── queries_first50.tsv
│   └── frames_10_shards/q_{0-9}.tsv
├── fanoutqa/
├── musique/
├── nq/
├── qampari/
├── quest/
└── webqsp/
```

### Ground Truth (data/)

```
data/
├── browsecomp_plus_decrypted.jsonl  # BCP ground truth (answers + gold docs)
└── frames_ground_truth.jsonl        # FRAMES ground truth (answers + wiki links)
```

```json
{
  "query_id": "Q001",
  "query": "...",
  "answer": "...",
  "gold_docs": [...],
  "negative_docs": [...],
  "evidence_docs": [...]
}
```

---

## Development Workflows

### Run an Agent (BM25 + OpenAI)

```bash
# 1. Start MCP retrieval server
python searcher/mcp_server.py --searcher-type bm25 --index-path indexes/bm25 --port 8000

# 2. Run agent
export OPENAI_API_KEY="sk-..."
python search_agent/openai_client.py \
  --model gpt-4 \
  --output-dir runs/bcp/bm25/full/gpt4 \
  --searcher-type bm25 \
  --index-path indexes/bm25 \
  --num-threads 10

# 3. Evaluate
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/bcp/bm25/full/gpt4 \
  --output_dir evals/bcp/bm25/full/gpt4 \
  --tensor_parallel_size 1
```

### Run an Agent (Anthropic + ngrok MCP)

```bash
export NGROK_AUTHTOKEN="..."
python searcher/mcp_server.py --searcher-type bm25 --index-path indexes/bm25 --port 8080 --public
# → public URL: https://xxxx.ngrok-free.app/mcp

export ANTHROPIC_API_KEY="sk-ant-..."
python search_agent/anthropic_client.py \
  --model claude-opus-4-20250514 \
  --mcp-url https://xxxx.ngrok-free.app/mcp \
  --output-dir runs/bcp/bm25/full/claude
```

### Implement a Custom Retriever

1. Subclass `BaseSearcher` in `searcher/searchers/custom_searcher.py`
2. Implement: `parse_args()`, `search()`, `get_document()`, `search_type`
3. Run any client with `--searcher-type custom`

### HPC Cluster (SLURM)

```bash
# Generate SBATCH files
python write_sbatch.py gen_embed     # Embedding index jobs
python write_sbatch.py run_agentic   # Agent execution jobs
python write_sbatch.py run_qwen3     # Qwen3 jobs

# Submit
sbatch run.SBATCH
sbatch run_web.SBATCH
```

Typical resources: 2× A100 GPU, 10 CPU, 300 GB RAM, 12–48 h.

### Analyze Query Repetition

```bash
# String-based (fast)
python compute_repeats.py --input-dir runs/bcp/bm25/full/gpt4

# LLM-based (semantic, requires vLLM with Qwen3-30B)
python compute_repeats_vllm.py --input-dir runs/bcp/bm25/full/gpt4
```

### Browse Traces (Streamlit UI)

```bash
streamlit run trace_viewer.py
```

---

## Environment Variables

| Variable | Used By |
|----------|---------|
| `OPENAI_API_KEY` | openai_client.py, evaluate_with_openai.py |
| `ANTHROPIC_API_KEY` | anthropic_client.py |
| `GOOGLE_API_KEY` | gemini_client.py |
| `HF_TOKEN` | Model downloads via Hugging Face |
| `NGROK_AUTHTOKEN` | mcp_server.py --public flag |

---

## Common CLI Arguments (across clients)

| Argument | Purpose |
|----------|---------|
| `--output-dir` | Where to save trajectory JSON files |
| `--searcher-type` | `bm25`, `faiss`, or `custom` |
| `--index-path` | Path to search index |
| `--model` | LLM model identifier |
| `--num-threads` | Concurrent query processing |
| `--max-tokens` | Max output tokens per LLM call |
| `--k` | Number of search results to return |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MCP URL unreachable | Check ngrok tunnel is active; verify `NGROK_AUTHTOKEN` |
| Search returns no results | Verify index path; try `--searcher-type bm25` as baseline |
| Evaluation parse errors | Check ground truth JSONL exists; confirm `final_answer` field present |
| OOM with FAISS | Use smaller embedding model (0.6B vs 8B); enable shard loading |
| API rate limiting | Reduce `--num-threads`; add retry delay in client |

---

## Workflow Orchestration

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules that prevent the same mistake from recurring
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for project-relevant patterns

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after any corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
