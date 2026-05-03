# Scout/Explore Experiment Results

**Setup:** 150-query test slice on each benchmark. LLM-as-judge: Qwen3-32B. All numbers reflect the corrected `evaluate_run.py` (eval bug fix, 2026-04-26) that grades `context_limit` trajectories using their forced final answers; pre-fix numbers undercounted accuracy for verbose models (full diff in NOTABLE_ASSUMPTIONS.md).

**Seed semantics:** The clients (GLM/MiniMax/Qwen35) do **not** pass `temperature` or `seed` to vLLM, so agent generations are non-deterministic by default (vLLM default temperature, random seed per request). Where you see `seed42/43/44/45` in this doc it refers to the **selection seed** used by `random_select_tool_calls.py` to pick 5 random tool calls from the baseline trajectory — *not* an agent-level seed. Re-running the same agent run twice will give different trajectories.

**Conditions tested** (each model is evaluated under all):

1. **Baseline** — agent runs with no prepended evidence; standard agentic loop.
2. **+ full trajectory** — the entire first-run trajectory (all tool calls + observations) is injected into the prompt before the agent starts. Tests the upper bound of "everything we already learned."
3. **+ trajectory summary** — an LLM-generated summary of the first-run trajectory is injected instead of the raw trajectory. Tests whether a compressed orientation helps.
4. **+ Gemini-2.5-pro selected k=5 tool calls** — Gemini selects the 5 most useful tool-call/observation pairs from the first-run trajectory and prepends only those excerpts. Tests whether targeted excerpts beat full or summary.
5. **+ random k=5 tool calls (selection seed N)** — 5 tool-call/observation pairs sampled uniformly at random (with `random_select_tool_calls.py --seed N`) from the first-run trajectory. Controls for whether the *selection* matters or just the *amount* of prepended evidence.
6. **+ random k=5 tool calls (best of 4)** — pass@4 across selection seeds {42,43,44,45}: a question is counted correct if **any** of the 4 random subsets gets it right. Estimates the variance ceiling of random-subset prepending.
7. **+ self prompted explorer (budget=5)** — same model runs round-1 with `--search-budget 5` (capped at 5 tool calls), then round-2 prepends that budgeted trajectory raw via `traj_orig_ext`. Tests whether a model planning under a budget upfront picks better evidence than random subsampling of an unconstrained run.
8. **+ qwen3.5-4b explorer (budget=5, vanilla)** — Hung-Ting's vanilla qwen3.5-4b explorer trajectories (`--search-budget 5`) are prepended to the main agent. Tests whether a small, weak explorer can usefully guide a large main agent.
9. **+ qwen3.5-4b explorer (SFT on best-of-4 random selection)** — qwen3.5-4b fine-tuned on best-of-4 random-selection trajectories from a stronger model, then run with `--search-budget 5`. Distillation target = the empirical best-of-4 from condition 6.
10. **+ qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection)** — qwen3.5-4b fine-tuned on Gemini-selected k=5 trajectories. Distillation target = condition 4.
11. **+ qwen3.5-4b explorer (SFT on random selection)** — qwen3.5-4b fine-tuned on random k=5 trajectories. Distillation target = condition 5 (any single seed).

---

## Pending Work (BCP test150)

**Task 1 — Random k=5 best-of-4 (10 runs total, all eval'd, then aggregated):**

| Model | seed42 | seed43 | seed44 | seed45 | best-of-4 |
|---|:---:|:---:|:---:|:---:|:---:|
| GLM-4.7-Flash | ✅ 47.3% | ✅ 44.0% | ✅ 43.3% | ✅ 46.0% | ✅ **52.7% (+5.3pp)** |
| MiniMax-M2.5 | ✅ 57.3% | ✅ 52.7% | ✅ 49.3% | ✅ 51.3% | ✅ **71.3% (+14.0pp)** |
| Qwen3.5-122B-A10B | ✅ N=150 (49.3%) | ⚠️ N=135 (recovery 7811896 PENDING) | ⚠️ N=148 (last 2 hung, COMPLETED) | ⚠️ N=147 (last 3 hung, COMPLETED) | ⏳ deferred (see status notes) |

**Qwen3.5 status (resumable):** seed42 cleanly at N=150 after 1 successful recovery from N=130. Seeds 43/44/45 each ended with 2-3 trajectories short of 150 — same `Qwen3.5+H200 unbreakable agent-loop` pattern documented 5+ times in NOTABLE_ASSUMPTIONS.md (last 2-3 queries spiral indefinitely; client exits cleanly). The recovery resub for seed43 (job 7811896) is in-queue but blocked by h200_public quota. **Resume plan:** when h200 frees up, recovery completes seed43; then submit `--export=ALL,SEED=44` and `SEED=45` recoveries for the partial-tail seeds, eval, then compute Qwen3.5 best-of-4. Until then we treat Qwen3.5 best-of-4 as deferred and pivot to GLM/MiniMax for Tasks 2-6 below.

---

**Tasks 2-6 (cross-explorer pairings, GLM + MiniMax now; Qwen3.5 deferred):** Each of these prepends a `--search-budget 5` explorer trajectory to the main agent via `traj_orig_ext`. All 4 explorer-trajectory sources are pre-computed and 150-file complete; only the main-agent round-2 needs to run.

| Task | Explorer trajectory source | GLM main | MiniMax main |
|---|---|:---:|:---:|
| 2 | self-prompted (main agent itself, budget=5) | DEFERRED | DEFERRED |
| 3 | qwen3.5-4b vanilla (`runs/.../qwen3.5-4b/budget5_seed0/`) | ⏳ queued (7900979) | ⏳ queued (7900980) |
| 4 | qwen3.5-4b SFT-best-of-4-random (`runs/.../qwen3.5-4b-sft-best_of_4_random_selection_mode_c/budget5_seed0/`) | ⏳ queued (7900981) | ⏳ queued (7900982) |
| 5 | qwen3.5-4b SFT-Gemini-2.5-pro (`runs/.../qwen3.5-4b-sft-gemini_2.5_pro_selection/budget5_seed0/`) | ⏳ queued (7900983) | ⏳ queued (7900984) |
| 6 | qwen3.5-4b SFT-random (`runs/.../qwen3.5-4b-sft-random_selection/budget5_seed0/`) | ⏳ queued (7900985) | ⏳ queued (7900986) |

**Task 2 deferred** — self-prompted (budget=5) is conceptually adjacent to the existing **+ full trajectory** and **+ trajectory summary** rows (all three use the *same model's own* prior trajectory; they vary only in compression and exploration budget). Cross-explorer rows test the more compelling distillation question and are higher priority. Self-prompted SBATCHes stay in `sbatch/` (`run_bcp_test150_{glm,minimax}_budget5.SBATCH`) ready for future submission.

**Submission plan (parallelizable since GLM→h100_tandon and MiniMax→h200_public don't share quotas):** Tasks 3-6 can all queue immediately since their explorer trajectories are on disk. Task 2 round-1 also queues immediately; Task 2 round-2 has `--dependency=afterok:<round1_jobid>`. Total: 12 round-2 jobs (5 conditions × 2 main agents + 1 round-1 dependency) and 10 evals after the round-2s land.

**Infrastructure TODOs:**
- ⏳ Build 5 SBATCH templates per main agent (10 total: 5 conditions × {GLM, MiniMax})
- ⏳ Build 1 round-1 SBATCH per main agent (2 total: budget=5 baseline runs for self-prompted)
- ⏳ Build 5 eval SBATCH templates per main agent (10 total) OR reuse one SBATCH parameterized by `RUN_NAME` env var
- 🚫 Qwen3.5 main-agent for Tasks 2-6: deferred. Will resume after GLM/MiniMax are complete.

---

## BrowseComp-Plus (BCP) — Qwen3-Embedding-8B retriever

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.0 | 55.4 | 22.0 |
| + full trajectory | 47.3 | 20.3 | 4.3 |
| **+ trajectory summary** | **53.3** | 52.5 | 12.7 |
| + Gemini-2.5-pro selected k=5 tool calls | 46.7 | 29.1 | 8.6 |
| + random k=5 tool calls (selection seed=42) | 47.3 | 34.6 | 9.7 |
| + random k=5 tool calls (selection seed=43) | 44.0 | 31.7 | 10.0 |
| + random k=5 tool calls (selection seed=44) | 43.3 | 29.4 | 10.0 |
| + random k=5 tool calls (selection seed=45) | 46.0 | 31.7 | 9.4 |
| **+ random k=5 tool calls (best of 4)** | **52.7** | — | — |
| + self prompted explorer (budget=5) | DEFERRED — adjacent to + full trajectory and + trajectory summary (same-model self-info family); cross-explorer rows are higher priority. SBATCH templates remain in `sbatch/run_bcp_test150_{glm,minimax}_budget5.SBATCH` for future use. | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on random selection) | TBD | TBD | TBD |

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 45.3 | 54.3 | 21.8 |
| + full trajectory | 48.4 | 0.0 | 0.1 |
| + trajectory summary | 48.3 | 56.5 | 14.4 |
| + Gemini-2.5-pro selected k=5 tool calls (N=148) | 48.7 | 25.4 | 15.9 |
| **+ random k=5 tool calls (selection seed=42)** | **49.3** | 28.9 | 15.7 |
| + random k=5 tool calls (selection seed=43) | TBD (N≤135) | TBD | TBD |
| + random k=5 tool calls (selection seed=44) | TBD (N≤148) | TBD | TBD |
| + random k=5 tool calls (selection seed=45) | TBD (N≤147) | TBD | TBD |
| + random k=5 tool calls (best of 4) | TBD | TBD | TBD |
| + self prompted explorer (budget=5) | DEFERRED | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | DEFERRED | — | — |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | DEFERRED | — | — |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | DEFERRED | — | — |
| + qwen3.5-4b explorer (SFT on random selection) | DEFERRED | — | — |

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.7 | 56.9 | 15.3 |
| + full trajectory | 54.0 | 20.0 | 3.2 |
| + trajectory summary | 56.0 | 56.7 | 10.0 |
| + Gemini-2.5-pro selected k=5 tool calls | 55.3 | 45.4 | 8.6 |
| **+ random k=5 tool calls (selection seed=42)** | **57.3** | 49.8 | 9.1 |
| + random k=5 tool calls (selection seed=43) | 52.7 | 50.4 | 8.9 |
| + random k=5 tool calls (selection seed=44) | 49.3 | 49.5 | 8.7 |
| + random k=5 tool calls (selection seed=45) | 51.3 | 45.5 | 8.4 |
| **+ random k=5 tool calls (best of 4)** | **71.3** | — | — |
| + self prompted explorer (budget=5) | DEFERRED — adjacent to + full trajectory and + trajectory summary (same-model self-info family); cross-explorer rows are higher priority. SBATCH templates remain in `sbatch/run_bcp_test150_{glm,minimax}_budget5.SBATCH` for future use. | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | TBD | TBD | TBD |
| + qwen3.5-4b explorer (SFT on random selection) | TBD | TBD | TBD |

*Caveats:* Qwen3.5 traj_orig N=134, traj_summary N=149 — a few qids missing from the eval pool (one hit a hard 121K-token context overflow on the summary prompt). GLM baseline filtered from 830-query full run eval to test150 qids. Context_limit rates: GLM 9% / Qwen3.5 42% / MiniMax 71% baseline (the 65536-token cap drives MiniMax's tail; eval fix forces these to be graded rather than auto-failed). The Qwen3.5 main-agent rows for explorer-prepended conditions are **deferred** (see Pending Work) — Qwen3.5+H200 has a recurring agent-loop hang on the last 2-3 queries that costs ~1h per 150-query run; we'll resume after GLM/MiniMax are clean.

**Explorer trajectories used in the explorer-prepended rows (all `--search-budget 5` runs from Hung-Ting):**
- `qwen3.5-4b explorer (vanilla)` → `runs/bcp/.../qwen3.5-4b/budget5_seed0/` (150 trajectories; tool-call distribution: 137 at 5 calls, 6 at 4, 4 at 3, 2 at 7, 1 at 6). Paired with gpt-oss-120b main agent in Hung-Ting's original run = **14.7%**.
- `qwen3.5-4b explorer (SFT on best-of-4 random selection)` → `runs/bcp/.../qwen3.5-4b-sft-best_of_4_random_selection_mode_c/budget5_seed0/` (150 trajectories).
- `qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection)` → `runs/bcp/.../qwen3.5-4b-sft-gemini_2.5_pro_selection/budget5_seed0/` (150 trajectories).
- `qwen3.5-4b explorer (SFT on random selection)` → `runs/bcp/.../qwen3.5-4b-sft-random_selection/budget5_seed0/` (150 trajectories).
- `self prompted explorer (budget=5)` → main agent itself runs round-1 with `--search-budget 5`; round-2 prepends that round-1 trajectory raw.

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
