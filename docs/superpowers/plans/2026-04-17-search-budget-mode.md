# Search Budget Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `budget` mode that caps both the OSS and Tongyi agents to k search turns, via a budget-aware prompt and a hard iteration limit.

**Architecture:**
- **OSS (`oss_client.py`)**: Add `QUERY_TEMPLATE_NO_GET_DOCUMENT_BUDGET` user-message template (one sentence added to existing template) and a `--search-budget k` CLI arg that uses it and overrides `max_iterations`.
- **Tongyi (`tongyi_client.py` + `react_agent.py`)**: Add `--search-budget k` CLI arg; inject a budget note into the system prompt (same injection pattern as `traj_note`) and override `num_llm_calls_available` in the agent loop.
- **SBATCH**: Add a `budget` mode case to both `run_qwen3_test150.SBATCH` and `run_qwen3_train680.SBATCH` that passes `--search-budget ${SEARCH_BUDGET}` (which enforces the limit internally, so `--max-iterations` need not be passed separately for OSS).

**Tech Stack:** Python (argparse, string template), Bash (SBATCH)

---

### Task 1: Add the budget prompt template and format function to `search_agent/prompts.py`

**Files:**
- Modify: `search_agent/prompts.py`

- [ ] **Step 1: Add the template constant after `QUERY_TEMPLATE_NO_GET_DOCUMENT` (after line 21)**

Insert immediately after the `""".strip()` on line 21:

```python
QUERY_TEMPLATE_NO_GET_DOCUMENT_BUDGET = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You have a budget of {SearchBudget} search turns — use them efficiently.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()
```

- [ ] **Step 2: Add `format_query_with_budget` function after `format_query_with_traj_summary` (after line ~272)**

```python
def format_query_with_budget(query: str, budget: int) -> str:
    return QUERY_TEMPLATE_NO_GET_DOCUMENT_BUDGET.format(Question=query, SearchBudget=budget)
```

---

### Task 2: Add `--search-budget` CLI arg and wire it in `search_agent/oss_client.py`

**Files:**
- Modify: `search_agent/oss_client.py`

- [ ] **Step 1: Add `format_query_with_budget` to the import block (lines 14–18)**

```python
from prompts import (
    format_query,
    format_query_with_trajectory,
    format_query_with_traj_summary,
    format_query_with_budget,
)
```

- [ ] **Step 2: Add the `--search-budget` argument to the `ArgumentParser` block (after the `--k` arg, around line 667)**

```python
    parser.add_argument(
        "--search-budget",
        type=int,
        default=None,
        help="Max search turns; uses budget-aware prompt and overrides max_iterations",
    )
```

- [ ] **Step 3: Wire the budget prompt in `_handle_single_query` (around lines 501–506)**

Replace:
```python
        if args.planning_trigger in TRAJ_TRIGGERS:
            user_content = _build_trajectory_user_content(
                qid, qtext, args, client, trajectories_by_id, summaries_by_id
            )
        else:
            user_content = format_query(qtext, args.query_template)
```

With:
```python
        if args.planning_trigger in TRAJ_TRIGGERS:
            user_content = _build_trajectory_user_content(
                qid, qtext, args, client, trajectories_by_id, summaries_by_id
            )
        elif args.search_budget is not None:
            user_content = format_query_with_budget(qtext, args.search_budget)
        else:
            user_content = format_query(qtext, args.query_template)
```

- [ ] **Step 4: Override `max_iterations` when budget is set, in the `run_conversation_with_tools` call (around line 519)**

Replace:
```python
            messages, tool_usage, status = run_conversation_with_tools(
                client,
                initial_request,
                tool_handler,
                args.max_iterations,
                args.verbose,
            )
```

With:
```python
            effective_max_iter = args.search_budget if args.search_budget is not None else args.max_iterations
            messages, tool_usage, status = run_conversation_with_tools(
                client,
                initial_request,
                tool_handler,
                effective_max_iter,
                args.verbose,
            )
```

---

### Task 3: Add `--search-budget` CLI arg to `search_agent/tongyi_client.py`

**Files:**
- Modify: `search_agent/tongyi_client.py`

- [ ] **Step 1: Add the `--search-budget` argument (after the `--k` arg, around line 234)**

```python
    parser.add_argument(
        "--search-budget",
        type=int,
        default=None,
        help="Max search turns; injects budget note into system prompt and caps LLM calls",
    )
```

- [ ] **Step 2: Pass `search_budget` when constructing `MultiTurnReactAgent` (around line 286)**

Replace:
```python
    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["search"],
        search_tool_handler=search_tool_handler,
        multi_answer=args.multi_answer,
        planning_trigger=args.planning_trigger,
        max_chars=args.max_chars,
        reasoning_max_chars=args.reasoning_max_chars,
        tool_output_max_chars=args.tool_output_max_chars,
    )
```

With:
```python
    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["search"],
        search_tool_handler=search_tool_handler,
        multi_answer=args.multi_answer,
        planning_trigger=args.planning_trigger,
        max_chars=args.max_chars,
        reasoning_max_chars=args.reasoning_max_chars,
        tool_output_max_chars=args.tool_output_max_chars,
        search_budget=args.search_budget,
    )
```

---

### Task 4: Wire budget into `search_agent/tongyi_utils/react_agent.py`

**Files:**
- Modify: `search_agent/tongyi_utils/react_agent.py`

- [ ] **Step 1: Accept `search_budget` in `MultiTurnReactAgent.__init__` (around line 68)**

Add after `self.tool_output_max_chars = kwargs.pop("tool_output_max_chars", 5000)`:
```python
        self.search_budget = kwargs.pop("search_budget", None)
```

- [ ] **Step 2: Inject budget note into the system prompt and cap LLM calls in `_run` (around lines 207–228)**

Replace the existing `traj_note` injection block:
```python
        traj_note = ""
        if self.planning_trigger in ("traj_ext", "traj_orig_ext"):
            traj_note = TRAJECTORY_SYSTEM_NOTE
        elif self.planning_trigger in ("traj_summary_ext", "traj_summary_orig_ext"):
            traj_note = TRAJ_SUMMARY_SYSTEM_NOTE

        if traj_note:
            idx = system_prompt.rfind("Current date:")
            if idx >= 0:
                head = system_prompt[:idx].rstrip()
                system_prompt = head + "\n\n" + traj_note + "\n\nCurrent date: " + str(cur_date)
            else:
                system_prompt = system_prompt + "\n\n" + traj_note + "\n\n" + str(cur_date)
        else:
            system_prompt = system_prompt + str(cur_date)
```

With:
```python
        traj_note = ""
        if self.planning_trigger in ("traj_ext", "traj_orig_ext"):
            traj_note = TRAJECTORY_SYSTEM_NOTE
        elif self.planning_trigger in ("traj_summary_ext", "traj_summary_orig_ext"):
            traj_note = TRAJ_SUMMARY_SYSTEM_NOTE

        budget_note = ""
        if self.search_budget is not None:
            budget_note = f"You have a budget of {self.search_budget} search turns — use them efficiently."

        combined_note = "\n\n".join(n for n in [traj_note, budget_note] if n)

        if combined_note:
            idx = system_prompt.rfind("Current date:")
            if idx >= 0:
                head = system_prompt[:idx].rstrip()
                system_prompt = head + "\n\n" + combined_note + "\n\nCurrent date: " + str(cur_date)
            else:
                system_prompt = system_prompt + "\n\n" + combined_note + "\n\n" + str(cur_date)
        else:
            system_prompt = system_prompt + str(cur_date)
```

- [ ] **Step 3: Override `num_llm_calls_available` when budget is set (line 228)**

Replace:
```python
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
```

With:
```python
        num_llm_calls_available = self.search_budget if self.search_budget is not None else MAX_LLM_CALL_PER_RUN
```

---

### Task 5: Add `budget` mode to `run_qwen3_test150.SBATCH`

**Files:**
- Modify: `run_qwen3_test150.SBATCH`

- [ ] **Step 1: Add `SEARCH_BUDGET` variable after `seed=0` (around line 29)**

```bash
SEARCH_BUDGET=5
```

- [ ] **Step 2: Add `budget` case to the `case "$inner_mode"` block (before the `*)` catch-all)**

```bash
    budget)
        output_dir="${OUT_BASE}/budget${SEARCH_BUDGET}_seed${seed}"
        extra_args="--search-budget ${SEARCH_BUDGET}"
        ;;
```

---

### Task 6: Add `budget` mode to `run_qwen3_train680.SBATCH`

**Files:**
- Modify: `run_qwen3_train680.SBATCH`

- [ ] **Step 1: Add `SEARCH_BUDGET` variable after `seed=0` (around line 29)**

```bash
SEARCH_BUDGET=5
```

- [ ] **Step 2: Add `budget` case to the `case "$inner_mode"` block (before the `*)` catch-all)**

```bash
    budget)
        output_dir="${OUT_BASE}/budget${SEARCH_BUDGET}_seed${seed}"
        extra_args="--search-budget ${SEARCH_BUDGET}"
        ;;
```

---

## Self-Review

**Spec coverage:**
- [x] New budget-aware user-message template for OSS (minimal change — one sentence added)
- [x] Budget injected into Tongyi system prompt (same pattern as `traj_note`, one sentence added)
- [x] `k` specified via `--search-budget` CLI arg on both clients
- [x] OSS hard-stops at k iterations via `effective_max_iter`
- [x] Tongyi hard-stops at k LLM calls via `num_llm_calls_available`
- [x] Both test150 and train680 SBATCH files updated with `budget` mode case

**Placeholder scan:** None found.

**Type consistency:** `format_query_with_budget(query: str, budget: int) -> str` defined in Task 1, imported and called in Task 2. `search_budget: int | None` flows from `tongyi_client.py` → `MultiTurnReactAgent.__init__` → `_run` consistently.
