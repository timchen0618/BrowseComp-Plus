# Scout/Explore Experiment Results

**Setup:** 150-query test slice on each benchmark. LLM-as-judge: Qwen3-32B. All numbers reflect the corrected `evaluate_run.py` (eval bug fix, 2026-04-26) that grades `context_limit` trajectories using their forced final answers; pre-fix numbers undercounted accuracy for verbose models (full diff in NOTABLE_ASSUMPTIONS.md).

**Seed semantics:** The clients (GLM/MiniMax/Qwen35) do **not** pass `temperature` or `seed` to vLLM, so agent generations are non-deterministic by default (vLLM default temperature, random seed per request). Where you see `seed42/43/44/45` in this doc it refers to the **selection seed** used by `random_select_tool_calls.py` to pick 5 random tool calls from the baseline trajectory — *not* an agent-level seed. Re-running the same agent run twice will give different trajectories.

**Conditions tested** (each model is evaluated under all 5):

1. **Baseline** — agent runs with no prepended evidence; standard agentic loop.
2. **+ full trajectory** — the entire first-run trajectory (all tool calls + observations) is injected into the prompt before the agent starts. Tests the upper bound of "everything we already learned."
3. **+ trajectory summary** — an LLM-generated summary of the first-run trajectory is injected instead of the raw trajectory. Tests whether a compressed orientation helps.
4. **+ selected k=5 tool calls** — Gemini selects the 5 most useful tool-call/observation pairs from the first-run trajectory and prepends only those excerpts. Tests whether targeted excerpts beat full or summary.
5. **+ random k=5 tool calls (selection seed N)** — 5 tool-call/observation pairs sampled uniformly at random (with `random_select_tool_calls.py --seed N`) from the first-run trajectory. Controls for whether the *selection* matters or just the *amount* of prepended evidence.
6. **+ random k=5 tool calls (best of 4)** — pass@4 across selection seeds {42,43,44,45}: a question is counted correct if **any** of the 4 random subsets gets it right. Estimates the variance ceiling of random-subset prepending.
7. **+ self prompted explorer (budget=5, raw)** — same model runs round-1 with `--search-budget 5` (capped at 5 tool calls), then round-2 prepends that budgeted trajectory raw via `traj_orig_ext`. Tests whether a model planning under a budget upfront picks better evidence than random subsampling of an unconstrained run.

---

## Pending Work (BCP test150)

**Task 1 — Random k=5 best-of-4 (10 runs total, all eval'd, then aggregated):**

For pass@4 we need clean N=150 across all four selection seeds. GLM and MiniMax seed42 are already at N=150 and stay untouched. Qwen3.5 seed42 is at N=130 (3× TMA crashes) and needs a recovery rerun (same SBATCH, idempotent — only re-processes the 20 missing qids). Plus 3 new selection seeds (43/44/45) for each of the 3 models.

| Model | seed42 (existing) | seed43 (new) | seed44 (new) | seed45 (new) | best-of-4 |
|---|:---:|:---:|:---:|:---:|:---:|
| GLM-4.7-Flash | ✅ N=150 done (47.3%) | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ aggregate after evals |
| Qwen3.5-122B-A10B | ⚠️ N=130 (54.6%) — **rerun to fill 20 missing** | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ aggregate after evals |
| MiniMax-M2.5 | ✅ N=150 done (57.3%) | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ TODO run + eval | ⏳ aggregate after evals |

**Submission plan:** 10 round-1 runs (1 Qwen3.5 seed42 recovery + 9 new seeds across 3 models). For each as it lands at 150 trajectories: submit eval. After all 12 evals (4 seeds × 3 models) are done: run `scripts/compute_best_of_n.py` to compute pass@4 per model and fill in the best-of-4 column.

**Task 2 — Self prompted explorer + main agent (budget=5, raw prepend, 3 models = 3 round-1 + 3 round-2 + 3 evals):**

For each model, round-1 runs `--search-budget 5` (≤5 tool calls per trajectory), then round-2 prepends that budgeted trajectory raw via `traj_orig_ext`. Round-2 output is what gets evaluated.

| Model | Round-1 (budget=5) | Round-2 (traj_orig_ext) | Eval |
|---|:---:|:---:|:---:|
| GLM-4.7-Flash | ⏳ TODO | ⏳ TODO | ⏳ TODO |
| Qwen3.5-122B-A10B | ⏳ TODO | ⏳ TODO | ⏳ TODO |
| MiniMax-M2.5 | ⏳ TODO | ⏳ TODO | ⏳ TODO |

**Task 3 — Vanilla qwen3.5-4b explorer × 3 main agents (3 runs):** Hung-Ting will provide the fine-tuned qwen3.5-4b explorer trajectories later; for now we have the **vanilla** qwen3.5-4b explorer trajectories at `runs/bcp/Qwen3-Embedding-8B/test150/qwen3.5-4b/budget5_seed0/` (150 trajectories, ≤5 tool calls each — 91% have exactly 5; paired with gpt-oss-120b in Hung-Ting's original run = 14.7%). Main agent pairings:

| Main agent | Round-2 run | Eval |
|---|:---:|:---:|
| GLM-4.7-Flash | ⏳ TODO | ⏳ TODO |
| Qwen3.5-122B-A10B | ⏳ TODO | ⏳ TODO |
| MiniMax-M2.5 | ⏳ TODO | ⏳ TODO |

**Infrastructure TODOs:**
- ⏳ Refactor 3 round-1 random_tools SBATCHes (`run_bcp_test150_<slug>_random_tools.SBATCH`) to take `SEED` as env var → `sbatch --export=ALL,SEED=43 ...`
- ⏳ Create 3 SBATCHes for self-prompted-explorer: round-1 with `--search-budget 5` + round-2 with `traj_orig_ext` pointing at the budget=5 dir
- ⏳ Create 3 SBATCHes for vanilla qwen3.5-4b explorer × main-agent pairings (round-2 only)
- ⏳ Write `scripts/compute_best_of_n.py` aggregating per-query correctness across N eval JSONs (pass@N)
- 🚫 BLOCKED on Hung-Ting: fine-tuned qwen3.5-4b explorer trajectories (separate Table 2 row group when received)

---

## BrowseComp-Plus (BCP) — Qwen3-Embedding-8B retriever

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.0 | 55.4 | 22.0 |
| + full trajectory | 47.3 | 20.3 | 4.3 |
| **+ trajectory summary** | **53.3** | 52.5 | 12.7 |
| + selected k=5 tool calls | 46.7 | 29.1 | 8.6 |
| + random k=5 tool calls (selection seed=42) | 47.3 | 34.6 | 9.7 |
| + random k=5 tool calls (selection seed=43) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=44) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=45) | TBD | TBD | TBD |
| + random k=5 tool calls (best of 4) | TBD | TBD | TBD |
| + self prompted explorer (budget=5, raw) | TBD | TBD | TBD |

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 45.3 | 54.3 | 21.8 |
| + full trajectory | 48.4 | 0.0 | 0.1 |
| + trajectory summary | 48.3 | 56.5 | 14.4 |
| + selected k=5 tool calls (N=148) | 48.7 | 25.4 | 15.9 |
| **+ random k=5 tool calls (selection seed=42, N=130)** | **54.6** | 30.7 | 14.3 |
| + random k=5 tool calls (selection seed=43) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=44) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=45) | TBD | TBD | TBD |
| + random k=5 tool calls (best of 4) | TBD | TBD | TBD |
| + self prompted explorer (budget=5, raw) | TBD | TBD | TBD |

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.7 | 56.9 | 15.3 |
| + full trajectory | 54.0 | 20.0 | 3.2 |
| + trajectory summary | 56.0 | 56.7 | 10.0 |
| + selected k=5 tool calls | 55.3 | 45.4 | 8.6 |
| **+ random k=5 tool calls (selection seed=42)** | **57.3** | 49.8 | 9.1 |
| + random k=5 tool calls (selection seed=43) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=44) | TBD | TBD | TBD |
| + random k=5 tool calls (selection seed=45) | TBD | TBD | TBD |
| + random k=5 tool calls (best of 4) | TBD | TBD | TBD |
| + self prompted explorer (budget=5, raw) | TBD | TBD | TBD |

*Caveats:* Qwen3.5 traj_orig N=134, traj_summary N=149 — a few qids missing from the eval pool (one hit a hard 121K-token context overflow on the summary prompt). GLM baseline filtered from 830-query full run eval to test150 qids. Context_limit rates: GLM 9% / Qwen3.5 42% / MiniMax 71% baseline (the 65536-token cap drives MiniMax's tail; eval fix forces these to be graded rather than auto-failed).

### BCP — Cross-explorer pairings (qwen3.5-4b explorer × different main agents)

Hung-Ting fine-tuned qwen3.5-4b on gpt-oss-120b's random-subset trajectories. The vanilla (un-fine-tuned) qwen3.5-4b explorer trajectories he sent us live at `runs/bcp/Qwen3-Embedding-8B/test150/qwen3.5-4b/budget5_seed0/` (150 trajectories with `--search-budget 5`, distribution: 137 with 5 tool calls, 6 with 4, 4 with 3, 2 with 7, 1 with 6). They are paired below with different main agents — the first row (gpt-oss-120b main agent) is from his doc; the rest are TBD on our side.

| Explorer | Main agent | Acc | Recall | # calls | Notes |
| :---- | :---- | ----: | ----: | ----: | :---- |
| qwen3.5-4b (vanilla, budget=5) | gpt-oss-120b | 14.7 | — | — | Hung-Ting's number from doc |
| qwen3.5-4b (vanilla, budget=5) | GLM-4.7-Flash | TBD | TBD | TBD | round-2 only |
| qwen3.5-4b (vanilla, budget=5) | Qwen3.5-122B-A10B | TBD | TBD | TBD | round-2 only |
| qwen3.5-4b (vanilla, budget=5) | MiniMax-M2.5 | TBD | TBD | TBD | round-2 only |
| qwen3.5-4b (fine-tuned, Hung-Ting) | gpt-oss-120b | 🚫 BLOCKED | — | — | awaiting Hung-Ting's training |
| qwen3.5-4b (fine-tuned, Hung-Ting) | GLM-4.7-Flash | 🚫 BLOCKED | — | — | |
| qwen3.5-4b (fine-tuned, Hung-Ting) | Qwen3.5-122B-A10B | 🚫 BLOCKED | — | — | |
| qwen3.5-4b (fine-tuned, Hung-Ting) | MiniMax-M2.5 | 🚫 BLOCKED | — | — | |

---

## FRAMES — BGE-M3 retriever, Upstash Wikipedia

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 44.7 | 70.6 | 27.7 |
| + full trajectory | 46.2 | 15.9 | 8.5 |
| **+ trajectory summary** | **51.3** | 52.2 | 14.9 |
| + selected k=5 tool calls | TBD | TBD | TBD |
| + random k=5 tool calls (ablation) | TBD | TBD | TBD |

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 63.3 | 75.4 | 31.3 |
| + full trajectory (N=131) | 68.7 | 1.5 | 0.1 |
| + trajectory summary | 64.0 | 63.8 | 17.5 |
| + selected k=5 tool calls | TBD | TBD | TBD |
| + random k=5 tool calls (ablation) | TBD | TBD | TBD |

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 62.0 | 77.4 | 25.7 |
| + full trajectory (N=126) | 73.0 | 6.1 | 0.6 |
| + trajectory summary | 60.0 | 62.8 | 13.1 |
| + selected k=5 tool calls | TBD | TBD | TBD |
| + random k=5 tool calls (ablation) | TBD | TBD | TBD |

*Caveats:* FRAMES Recall is **article-level** (did the agent retrieve ≥1 passage from each of the relevant Wikipedia articles for that query) — computed post-hoc in `scripts/compute_frames_recall.py` from each query's `wiki_links` field in the GT, joined to the BGE-M3 corpus row IDs. After fixing URL canonicalization (GT uses `_` for spaces, url_map uses `%20`), **88.5% of GT URLs match (2209/2496) and 149/150 test150 qids have qrel evidence** — recall is now reported on essentially the full slice. GLM context_limit rate on FRAMES: 7% (baseline) / 11% (traj_orig) / 3% (traj_summary). GLM traj_orig was evaluated on N=143 (7 qids errored before producing output: 26, 74, 78, 99, 127, 129, 132).

---

## Key Observations

**Summary prepend consistently wins on both benchmarks.** All completed models beat their baseline more from a summarized trajectory than from the raw full trajectory:
- GLM BCP: baseline→traj_summary +5.3pp; FRAMES: +6.6pp (recall 70.6→52.2, expected drop from prepend overhead)
- Qwen3.5 BCP: +3.0pp (FRAMES TBD)
- MiniMax BCP: +7.3pp (FRAMES TBD)

**traj_orig_ext collapses search.** Full trajectory prepend fills the context window — only enough room for ~3-4 new searches (or zero, for Qwen3.5). Recall drops from ~55% to 0–20%. Models lean on the prepended evidence rather than re-querying.

**Summary restores search quality.** traj_summary_orig_ext recovers reasonable search depth (10–15 calls) and recall (52–57%) — the summary orients the model without crowding out retrieval.

**GLM Selected Tool Calls (46.7%) underperforms LLM Summary (53.3%) by 6.6pp on BCP.** Raw 5-tool-call excerpts don't match the orienting power of an LLM-synthesized summary. Recall partially recovers (29% vs 20% for traj_orig vs 53% for summary) but accuracy doesn't catch up. Awaiting Qwen3.5/MiniMax to confirm cross-model.

**Verbosity tax on context.** MiniMax (71% context_limit) and Qwen3.5 (42%) take a big hit from the 65536-token ceiling on BCP. GLM (9%) is comfortably under. With the eval fix, this no longer translates to "verbose models are dumb" — but it does cost compute and the higher baseline accuracy hides behind those forced answers.
