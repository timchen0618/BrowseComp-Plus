# CLAUDE.md — BrowseComp-Plus

Comprehensive guide for AI assistants working in this repository.

---

## Project Overview

**BrowseComp-Plus** is a research platform for evaluating complex search-and-reasoning tasks. It benchmarks multiple LLM-based "deep research" agents against a difficult question-answering dataset, measuring accuracy, retrieval recall, and calibration.

The system consists of two loosely-coupled halves:
1. **Retrieval server** (`searcher/`) — a FastMCP server exposing `search` and `get_document` tools over BM25 or dense (FAISS) indexes.
2. **LLM agent clients** (`search_agent/`) — thin wrappers around various provider APIs (Anthropic, OpenAI, Gemini, Qwen, OSS via vLLM, etc.) that call the retrieval server as a tool.

Evaluation, data prep, and utility scripts live in dedicated top-level directories.

---

## Repository Structure

```
BrowseComp-Plus/
├── searcher/                    # MCP retrieval server
│   ├── mcp_server.py            # Entry point; launches FastMCP server
│   ├── tools.py                 # Tool registration (search, get_document)
│   ├── search_r1_server.py      # Variant server for Search-R1 model
│   └── searchers/
│       ├── base.py              # Abstract BaseSearcher interface
│       ├── __init__.py          # SearcherType enum + factory
│       ├── bm25_searcher.py     # Pyserini / Lucene BM25
│       ├── faiss_searcher.py    # Dense vector (FAISS + Qwen3-Embedding)
│       └── custom_searcher.py   # Template for custom retrievers
│
├── search_agent/                # LLM agent clients
│   ├── prompts.py               # All prompt templates + format_query()
│   ├── utils.py                 # extract_retrieved_docids_from_result()
│   ├── anthropic_client.py      # Anthropic API + MCP beta
│   ├── openai_client.py         # OpenAI API (local or cloud)
│   ├── openai_client_with_mcp.py
│   ├── gemini_client.py         # Async Gemini client
│   ├── gemini_back.py
│   ├── qwen_client.py           # Qwen Agent framework
│   ├── oss_client.py            # vLLM-based OSS models
│   ├── search_r1_client.py
│   ├── tongyi_client.py         # Tongyi DeepResearch
│   └── tongyi_utils/            # ReAct agent + prompts for Tongyi
│
├── scripts_evaluation/          # Evaluation pipeline
│   ├── evaluate_run.py          # Main eval (vLLM Qwen3-32B judge)
│   ├── evaluate_with_openai.py  # OpenAI-based eval alternative
│   ├── compute_pass_k.py        # Pass@K metric
│   ├── deduplicate_trajectories.py
│   └── aggregate_score.py
│
├── scripts_build_index/         # Index creation utilities
│   ├── decrypt_dataset.py
│   └── download_and_decrypt_run.py
│
├── qampari_experiments/         # QAMPARI multi-answer ranking experiments
│   ├── generate_qampari_results.py
│   └── download.py
│
├── parametric/                  # Parametric knowledge generation
│   ├── generator.py             # vLLM-based doc generation
│   └── grade_generated_doc.py
│
├── src_utils/                   # General-purpose utilities
│   ├── load_trajectories.py     # Load + parse trajectory JSONL files
│   ├── filter_by_query_ids.py
│   ├── add_search_counts.py
│   ├── shard.py
│   ├── parse_output.py
│   ├── combine_json_to_jsonl.py
│   ├── remove_exceed_llm_calls.py
│   ├── create_corpus_bcp.py
│   ├── average_time_taken.py
│   └── move_webexplorer_runs.py
│
├── docs/                        # Per-provider setup documentation
│   ├── README.md                # Index
│   ├── anthropic.md
│   ├── openai.md
│   ├── gemini.md
│   ├── qwen.md
│   ├── oss.md
│   ├── search-r1.md
│   ├── tongyi.md
│   ├── custom_retriever.md
│   ├── netmind_mcp.md
│   └── llm_as_judge.md
│
├── data/                        # Gitignored — decrypted dataset files
├── topics-qrels/                # qrel files for citation evaluation
├── trace_viewer.py              # Streamlit UI for inspecting trajectories
├── query_grader.py              # vLLM-based query quality grader
├── compute_repeats.py           # Detect repeated documents in results
├── compute_repeats_vllm.py
├── find_missing_ids.py
├── write_sbatch.py              # Generate SLURM batch scripts
├── pyproject.toml               # Poetry project (Python >=3.10)
└── .python-version              # Python 3.10
```

---

## Key Abstractions

### BaseSearcher (`searcher/searchers/base.py`)

All search backends implement this ABC:

```python
class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Returns: [{"docid": str, "score": float, "snippet": str}, ...]

    @abstractmethod
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        # Returns: {"docid": str, "snippet": str} | None

    @property
    @abstractmethod
    def search_type(self) -> str: ...

    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None: ...
```

To add a new retrieval backend, subclass `BaseSearcher`, implement all abstract members, and register the class in `searcher/searchers/__init__.py` under the `SearcherType` enum.

### MCP Server (`searcher/mcp_server.py`)

Launched with `python -m searcher.mcp_server` (or directly). Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--searcher-type` | required | `bm25`, `faiss`, `custom`, … |
| `--transport` | `sse` | `stdio`, `streamable-http`, `sse` |
| `--port` | `8000` | HTTP port |
| `--k` | `5` | Results per search call |
| `--snippet-max-tokens` | `512` | Token truncation via Qwen3-0.6B tokenizer |
| `--get-document` | off | Also expose `get_document` tool |
| `--public` | off | Open ngrok tunnel, print public URL |

### Prompt Templates (`search_agent/prompts.py`)

| Constant | Purpose |
|----------|---------|
| `QUERY_TEMPLATE` | Standard deep-research agent (search + get_document, with citations) |
| `QUERY_TEMPLATE_NO_GET_DOCUMENT` | Search-only variant |
| `QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION` | No citations variant |
| `QUERY_TEMPLATE_ORACLE` | Oracle: documents are provided directly |
| `GRADER_TEMPLATE` | Judge model prompt |
| `WEBSAILOR_SYSTEM_PROMPT_MULTI` | Alternative system prompt |

`format_query(query, query_template)` maps a template name string to the actual format call.

### Agent Response Format

All agent clients persist results as individual JSON files under a `runs/` directory (gitignored). Each file has this structure:

```json
{
  "query_id": "string",
  "status": "completed | failed | token limit reached",
  "result": [
    {
      "type": "reasoning | tool_call | output_text",
      "tool_name": "search | get_document | null",
      "arguments": "json_string | null",
      "output": "string | list | dict"
    }
  ],
  "retrieved_docids": ["docid1", "docid2"],
  "tool_call_counts": {"search": 5, "get_document": 2},
  "metadata": {"model": "claude-opus-4-5", ...}
}
```

---

## Data Files

**Gitignored paths** (must be obtained separately):
- `data/browsecomp_plus_decrypted.jsonl` — ground-truth dataset (`query_id`, `query`, `answer`)
- `data/indexes/` — BM25 Lucene or FAISS index directories
- `runs/` — agent trajectory output
- `evals/` — evaluation output
- `decrypted_run_files/` — pre-existing decrypted trajectory files for the trace viewer

**Topics/qrels** (tracked):
- `topics-qrels/qrel_evidence.txt` — qrel file for citation recall evaluation (TREC format: `query_id 0 doc_id 1`)

---

## Development Workflows

### 1. Running the Retrieval Server

```bash
# BM25
python searcher/mcp_server.py \
  --searcher-type bm25 \
  --index-path /path/to/lucene-index \
  --transport sse --port 8000

# Dense (FAISS)
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-path /path/to/faiss.index \
  --transport sse --port 8000
```

### 2. Running an Agent

Each client script accepts `--mcp-url` pointing to the running MCP server. Example:

```bash
python search_agent/anthropic_client.py \
  --mcp-url http://localhost:8000/mcp \
  --model claude-opus-4-5 \
  --input data/browsecomp_plus_decrypted.jsonl \
  --output-dir runs/anthropic_test
```

Refer to `docs/<provider>.md` for provider-specific flags and environment variables.

### 3. Evaluating Results

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir runs/anthropic_test \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --eval_dir evals/anthropic_test \
  --model Qwen/Qwen3-32B \
  --tensor_parallel_size 4
```

Outputs:
- Per-query `*_eval.json` files under `evals/`
- `evaluation_summary.json` with Accuracy, Recall, Calibration Error
- `detailed_judge_results.csv`

### 4. Visualizing Traces

```bash
streamlit run trace_viewer.py
```

Reads from `decrypted_run_files/` directory. Allows browsing per-query trajectories with expandable reasoning and tool-call panels.

### 5. QAMPARI Experiments

```bash
# 1. Run qampari.SBATCH (SLURM) or equivalent
# 2. Parse last queries from trajectories
python qampari_experiments/generate_qampari_results.py \
  --trajectory-dir runs/qampari \
  --output qampari_results.jsonl
```

---

## Code Conventions

### Style
- Python 3.10+ syntax; use `str | None` union type hints (not `Optional[str]`)
- `snake_case` for functions and variables
- `CamelCase` for classes
- `UPPER_CASE` for module-level constants (especially prompt strings)
- Private helpers prefixed with `_`

### Argument Parsing
Every script uses `argparse`. Searcher-specific flags are added via `BaseSearcher.parse_args(parser)` before `parser.parse_args()` is called.

### File I/O
- Agent output: individual `.json` per query (not JSONL) under a run directory
- Dataset/trajectories: `.jsonl` (one JSON object per line, UTF-8)
- Use `pathlib.Path` consistently, not raw `os.path` strings
- Always open files with `encoding="utf-8"`

### Error Handling
- Rate limit errors: retry with exponential backoff (see `make_request_with_retry` in `anthropic_client.py`)
- Max retries: 5, initial delay: 60s for API calls
- Log warnings with `print(f"[Warning] ...")` convention; use `rich.print` (`rprint`) in agent clients for coloured output

### Environment Variables
Loaded via `python-dotenv`; both a global `.env` at the repo root and a script-local `.env` are supported (`load_dotenv` is called twice with `override=False`). Key variables used across the codebase:

| Variable | Used by |
|----------|---------|
| `ANTHROPIC_API_KEY` | anthropic_client.py |
| `OPENAI_API_KEY` | openai_client.py |
| `GOOGLE_API_KEY` | gemini_client.py |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | faiss_searcher.py, mcp_server.py |
| `HF_HOME` | mcp_server.py |
| `NGROK_AUTHTOKEN` | mcp_server.py (`--public` mode) |

### Concurrency
- Agent clients use `concurrent.futures.ThreadPoolExecutor` to parallelize queries
- Gemini client uses `asyncio`
- Evaluation uses vLLM's batched `.chat()` API (`--batch_size` flag, default 64)

---

## Adding a New Search Backend

1. Create `searcher/searchers/my_searcher.py` subclassing `BaseSearcher`.
2. Implement `search()`, `get_document()`, `search_type` property, and `parse_args()`.
3. Add an entry to the `SearcherType` enum in `searcher/searchers/__init__.py`.
4. Document setup in `docs/custom_retriever.md` or a new doc file.

## Adding a New LLM Agent Client

1. Create `search_agent/my_client.py`.
2. Accept `--mcp-url` (or equivalent), `--model`, `--input`, `--output-dir` args.
3. Use `format_query()` from `prompts.py` for consistent prompting.
4. Call `extract_retrieved_docids_from_result()` from `utils.py` when saving results.
5. Persist each query result as a JSON file matching the agent response schema above.
6. Add provider-specific docs under `docs/`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy (%)** | Fraction of queries judged correct by Qwen3-32B |
| **Retrieval Recall (%)** | Fraction of qrel positive docs retrieved across all search calls |
| **Calibration Error (%)** | RMS calibration error between agent confidence and judge correctness (requires ≥100 samples) |
| **Citation Precision/Recall** | Precision and recall of docids cited in the final answer vs qrel positives |
| **Pass@K** | Computed separately via `scripts_evaluation/compute_pass_k.py` |

The judge model uses `Qwen/Qwen3-32B` by default with `enable_thinking=False`.

---

## Dependencies

Not all dependencies are in `pyproject.toml` (only `tqdm` is listed). The actual runtime requirements vary by component:

| Component | Key Packages |
|-----------|-------------|
| BM25 server | `pyserini`, `faiss-cpu` |
| Dense server | `faiss-gpu`, `transformers`, `datasets`, `torch` |
| MCP server | `fastmcp`, `pyngrok`, `python-dotenv` |
| Anthropic client | `anthropic` |
| OpenAI client | `openai` |
| Gemini client | `google-genai` |
| Qwen client | `qwen_agent` |
| OSS / evaluation | `vllm`, `transformers`, `numpy` |
| Trace viewer | `streamlit` |
| Utilities | `tqdm`, `rich`, `python-dotenv` |

Install provider-specific dependencies as needed. Use Python 3.10 (see `.python-version`).

---

## Important Notes for AI Assistants

- **`data/`, `runs/`, `evals/`, `indexes/`, `embeddings/`** are gitignored. Never assume these exist; always check before reading.
- **Do not hardcode model names** — pass them via CLI args. The default judge is `Qwen/Qwen3-32B`.
- **The `result` list in trajectory JSON** ends with `{"type": "output_text"}` when the agent completed successfully. Earlier entries are `"reasoning"` or `"tool_call"` steps.
- **Citation extraction** supports both `[docid]` (standard) and `【docid】` (fullwidth bracket, used by some OSS fine-tunes).
- **The evaluation script mirrors directory structure**: `runs/foo/bar/` → `evals/foo/bar/`. Do not flatten.
- **`status` field** can be `"completed"`, `"failed"`, or contain `"token limit reached"` as a substring; check with `in` not `==` for the token-limit case.
- **Snippet truncation** in the MCP server uses the `Qwen/Qwen3-0.6B` tokenizer regardless of the embedding model used for retrieval.

---

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project context

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
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
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
