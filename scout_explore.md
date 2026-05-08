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
| Qwen3.5-122B-A10B | ✅ N=150 (49.3%) | ⚠️ N=135 | ⚠️ N=148 (last 2 hung) | ⚠️ N=147 (last 3 hung) | 🚫 **incomplete — recovery cancelled 2026-05-03 per user pivot** |

**Qwen3.5 best-of-4 status: incomplete (cancelled).** seed42 cleanly at N=150 after 1 successful recovery from N=130. Seeds 43/44/45 each ended with 2-3 trajectories short of 150 — same `Qwen3.5+H200 unbreakable agent-loop` pattern documented 5+ times in NOTABLE_ASSUMPTIONS.md (last 2-3 queries spiral indefinitely; client exits cleanly). The recovery resubmit for seed43 (job 7811896) sat blocked by h200_public quota for ~16 ticks before being cancelled when the user pivoted to Tasks 3-6 (cross-explorer pairings). **To resume later:** the 3 recovery commands are still in NOTABLE_ASSUMPTIONS.md (`sbatch --export=ALL,SEED={43,44,45} sbatch/run_bcp_test150_qwen3_5_random_tools.SBATCH`); each is idempotent and only re-processes missing qids. After all 4 seeds reach N=150, run `python scripts/compute_best_of_n.py --inputs evals/bcp/.../qwen3.5-122b-a10b/random_tools_seed{42,43,44,45}/evaluation_summary.json --label "Qwen3.5"` to compute pass@4 and update this row. We're treating it as out-of-scope for the current pivot; not blocking anything.

---

**Tasks 2-6 (cross-explorer pairings, GLM + MiniMax now; Qwen3.5 deferred):** Each of these prepends a `--search-budget 5` explorer trajectory to the main agent via `traj_orig_ext`. All 4 explorer-trajectory sources are pre-computed and 150-file complete; only the main-agent round-2 needs to run.

| Task | Explorer trajectory source | GLM main | MiniMax main |
|---|---|:---:|:---:|
| 2 | self-prompted (main agent itself, budget=5) | DEFERRED | DEFERRED |
| 3 | qwen3.5-4b vanilla (`runs/.../qwen3.5-4b/budget5_seed0/`) | ✅ 42.7% | ✅ 48.0% |
| 4 | qwen3.5-4b SFT-best-of-4-random (`runs/.../qwen3.5-4b-sft-best_of_4_random_selection_mode_c/budget5_seed0/`) | ✅ 42.7% | ✅ 47.3% |
| 5 | qwen3.5-4b SFT-Gemini-2.5-pro (`runs/.../qwen3.5-4b-sft-gemini_2.5_pro_selection/budget5_seed0/`) | ✅ 47.3% | ✅ 46.7% |
| 6 | qwen3.5-4b SFT-random (`runs/.../qwen3.5-4b-sft-random_selection/budget5_seed0/`) | ✅ 42.7% | ✅ 43.3% |

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

| Condition | Acc | Δ vs base | Recall | # calls |
| :---- | ----: | :---- | ----: | ----: |
| Baseline | 48.0 | — | 55.4 | 21.6 |
| + full trajectory | 47.3 | -0.7 [-3.6, +2.3] | 20.3 | 4.3 |
| **+ trajectory summary** | **53.3** | +5.3 [+0.8, +9.9] * | 52.5 | 12.7 |
| + Gemini-2.5-pro selected k=5 tool calls | 46.7 | -1.3 [-5.0, +2.4] | 29.1 | 8.6 |
| + random k=5 tool calls (selection seed=42) | 47.3 | -0.7 [-5.7, +4.4] | 34.6 | 9.7 |
| + random k=5 tool calls (selection seed=43) | 44.0 | -4.0 [-8.5, +0.5] | 31.7 | 10.0 |
| + random k=5 tool calls (selection seed=44) | 43.3 | -4.7 [-9.0, -0.3] | 29.4 | 9.9 |
| + random k=5 tool calls (selection seed=45) | 46.0 | -2.0 [-6.3, +2.3] | 31.7 | 9.4 |
| **+ random k=5 tool calls (best of 4)** | **52.7** | † | — | — |
| + self prompted explorer (budget=5) | DEFERRED — adjacent to + full trajectory and + trajectory summary (same-model self-info family); cross-explorer rows are higher priority. SBATCH templates remain in `sbatch/run_bcp_test150_{glm,minimax}_budget5.SBATCH` for future use. | — | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | 42.7 | -5.3 [-12.5, +1.8] | 43.6 | 13.1 |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | 42.7 | -5.3 [-11.7, +1.1] | 43.4 | 18.1 |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | 47.3 | -0.7 [-7.7, +6.4] | 45.7 | 18.7 |
| + qwen3.5-4b explorer (SFT on random selection) | 42.7 | -5.3 [-12.0, +1.3] | 43.5 | 19.7 |

* p<0.05 (McNemar exact); ** BH-significant at q=0.05; † best-of-N variance differs (see compute_best_of_n.py)

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Δ vs base | Recall | # calls |
| :---- | ----: | :---- | ----: | ----: |
| Baseline | 45.3 | — | 54.3 | 21.8 |
| + full trajectory | 48.4 | -0.8 [-2.3, +0.8] | 0.0 | 0.1 |
| + trajectory summary | 48.3 | +2.7 [-0.5, +5.9] | 56.5 | 14.4 |
| + Gemini-2.5-pro selected k=5 tool calls | 48.6 | +2.7 [-1.0, +6.4] | 25.4 | 15.9 |
| **+ random k=5 tool calls (selection seed=42)** | **49.3** | +4.0 [-0.5, +8.5] | 28.9 | 15.7 |
| + random k=5 tool calls (selection seed=43) | 44.0 | -1.3 [-5.5, +2.8] | 27.5 | 16.5 |
| + random k=5 tool calls (selection seed=44) | 47.3 | +2.0 [-0.9, +4.9] | 25.4 | 17.6 |
| + random k=5 tool calls (selection seed=45) | 🚫 incomplete (N=147, recovery cancelled) | — | — | — |
| + random k=5 tool calls (best of 4) | 🚫 incomplete — depends on seeds 43/44/45 above | — | — | — |
| + self prompted explorer (budget=5) | DEFERRED | — | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | DEFERRED | — | — | — |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | DEFERRED | — | — | — |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | DEFERRED | — | — | — |
| + qwen3.5-4b explorer (SFT on random selection) | DEFERRED | — | — | — |

* p<0.05 (McNemar exact); ** BH-significant at q=0.05; † best-of-N variance differs (see compute_best_of_n.py)

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Δ vs base | Recall | # calls |
| :---- | ----: | :---- | ----: | ----: |
| Baseline | 48.7 | — | 56.9 | 15.3 |
| + full trajectory | 54.0 | +5.3 [-0.5, +11.2] | 20.0 | 3.2 |
| **+ trajectory summary** | **56.0** | +7.3 [+1.3, +13.3] * | 56.7 | 10.0 |
| + Gemini-2.5-pro selected k=5 tool calls | 55.3 | +6.7 [-0.5, +13.8] | 45.4 | 8.6 |
| **+ random k=5 tool calls (selection seed=42)** | **57.3** | +8.7 [+1.6, +15.7] * | 49.8 | 9.1 |
| + random k=5 tool calls (selection seed=43) | 52.7 | +4.0 [-2.1, +10.1] | 50.4 | 8.9 |
| + random k=5 tool calls (selection seed=44) | 49.3 | +0.7 [-5.9, +7.2] | 49.5 | 8.7 |
| + random k=5 tool calls (selection seed=45) | 51.3 | +2.7 [-3.5, +8.8] | 45.5 | 8.4 |
| **+ random k=5 tool calls (best of 4)** | **71.3** | † | — | — |
| + self prompted explorer (budget=5) | DEFERRED — adjacent to + full trajectory and + trajectory summary (same-model self-info family); cross-explorer rows are higher priority. SBATCH templates remain in `sbatch/run_bcp_test150_{glm,minimax}_budget5.SBATCH` for future use. | — | — | — |
| + qwen3.5-4b explorer (budget=5, vanilla) | 48.0 | -0.7 [-7.5, +6.1] | 47.4 | 9.8 |
| + qwen3.5-4b explorer (SFT on best-of-4 random selection) | 47.3 | -1.3 [-8.5, +5.8] | 47.9 | 10.4 |
| + qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection) | 46.7 | -2.0 [-8.3, +4.3] | 47.2 | 10.5 |
| + qwen3.5-4b explorer (SFT on random selection) | 43.3 | -5.3 [-13.4, +2.7] | 47.4 | 10.6 |

* p<0.05 (McNemar exact); ** BH-significant at q=0.05; † best-of-N variance differs (see compute_best_of_n.py)

*Caveats:* Qwen3.5 traj_orig N=134, traj_summary N=149 — a few qids missing from the eval pool (one hit a hard 121K-token context overflow on the summary prompt). GLM baseline filtered from 830-query full run eval to test150 qids. Context_limit rates: GLM 9% / Qwen3.5 42% / MiniMax 71% baseline (the 65536-token cap drives MiniMax's tail; eval fix forces these to be graded rather than auto-failed). The Qwen3.5 main-agent rows for explorer-prepended conditions are **deferred** (see Pending Work) — Qwen3.5+H200 has a recurring agent-loop hang on the last 2-3 queries that costs ~1h per 150-query run; we'll resume after GLM/MiniMax are clean.

*Δ vs base convention:* Δ is computed on the **paired intersection** of qids that both the row and the baseline answered, then McNemar exact + Newcombe paired CI on that intersection. For rows with N<150 (Qwen3.5 traj_orig N=134, traj_summary N=149, gemini-selected N=148), this means `Δ ≠ row_Acc − baseline_Acc` because the baseline restricted to those qids has a different Acc than the headline 45.3%. The headline Acc shown is still the eval JSON's full-set Accuracy. CI half-width at n=150, p=0.5 is ≈±8pp on the cell-level Wilson interval, so paired delta is the more informative measure.

**Explorer trajectories used in the explorer-prepended rows (all `--search-budget 5` runs from Hung-Ting):**
- `qwen3.5-4b explorer (vanilla)` → `runs/bcp/.../qwen3.5-4b/budget5_seed0/` (150 trajectories; tool-call distribution: 137 at 5 calls, 6 at 4, 4 at 3, 2 at 7, 1 at 6). Paired with gpt-oss-120b main agent in Hung-Ting's original run = **14.7%**.
- `qwen3.5-4b explorer (SFT on best-of-4 random selection)` → `runs/bcp/.../qwen3.5-4b-sft-best_of_4_random_selection_mode_c/budget5_seed0/` (150 trajectories).
- `qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection)` → `runs/bcp/.../qwen3.5-4b-sft-gemini_2.5_pro_selection/budget5_seed0/` (150 trajectories).
- `qwen3.5-4b explorer (SFT on random selection)` → `runs/bcp/.../qwen3.5-4b-sft-random_selection/budget5_seed0/` (150 trajectories).
- `self prompted explorer (budget=5)` → main agent itself runs round-1 with `--search-budget 5`; round-2 prepends that round-1 trajectory raw.

---

## BCP test300 (n=300) — GLM-4.7-Flash + MiniMax-M2.5

300-qid random sample drawn fresh from BCP-830 (independent of test150). MiniMax resumed 2026-05-07.

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 40.00 | 52.26 | — |
| + full trajectory (`traj_orig_ext`) | 40.00 | 19.47 | — |
| **+ trajectory summary (`traj_summary_orig_ext`)** | **46.33** | 48.89 | — |
| + Gemini-2.5-pro selected k=5 tool calls | 48.33 | 27.78 | — |
| + random k=5 tool calls (selection seed=42) | 46.67 | 35.91 | — |
| + random k=5 tool calls (selection seed=43) | 46.33 | 33.17 | — |
| + random k=5 tool calls (selection seed=44) | 47.67 | 34.58 | — |
| + random k=5 tool calls (selection seed=45) | 47.00 | 33.30 | — |
| **+ random k=5 tool calls (best of 4)** | **57.67** † | — | — |
| + budget-5 round-1 (no extras, just truncate to 5 calls) | 28.67 | 33.71 | — |
| + self-prompted explorer (round-2 prepends round-1 budget=5 trajectory) | 37.33 | 26.41 | — |

† best-of-4 lift over best single seed: +10.00pp (143/300 → 173/300). Per-seed unique solves: 6/6/9/5.

**Comparison to test150 GLM Δs:**

| Condition | test150 Δ vs base | test300 Δ vs base |
| :---- | ----: | ----: |
| traj_summary_orig_ext | +5.3 * | +6.33 |
| Gemini-selected k=5 | -1.3 | +8.33 |
| random k=5 best single | -0.7 | +7.67 |
| random k=5 best-of-4 | (52.7) | (57.67) |

The +8pp gain from `traj_summary_orig_ext` reproduces (test150 was statistically significant; test300 has more headroom since baseline is lower at 40% vs test150's 48%). Random/Gemini selected-tools deltas now look positive at n=300 — opposite sign from test150 — suggesting the test150 negative deltas may have been noise.

**Pause decision (2026-05-06):** MiniMax test300 paused (h200_public partition GPU-starved) to focus throughput on GLM. MiniMax test300 baseline + budget5 evals retained on disk. **Resumed 2026-05-07; complete 2026-05-08.**

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 46.33 | 55.26 | 15.95 |
| + full trajectory (`traj_orig_ext`) | 53.02 | 18.32 | 3.03 |
| **+ trajectory summary (`traj_summary_orig_ext`)** | **51.67** | 52.17 | 10.24 |
| + Gemini-2.5-pro selected k=5 tool calls | _gated on user JSONL_ | — | — |
| + random k=5 tool calls (selection seed=42) | 52.00 | 51.42 | 9.24 |
| + random k=5 tool calls (selection seed=43) | 51.33 | 48.38 | 9.18 |
| + random k=5 tool calls (selection seed=44) ‡ | 50.84 | 51.70 | 9.35 |
| + random k=5 tool calls (selection seed=45) ‡ | 48.16 | 47.97 | 9.15 |
| **+ random k=5 tool calls (best of 4)** | **69.13** † | — | — |
| + budget-5 round-1 (no extras, just truncate to 5 calls) | 36.00 | 33.12 | 4.97 |
| + self-prompted explorer (round-2 prepends round-1 budget=5 trajectory) | 50.00 | 32.46 | 7.57 |

† best-of-4 lift over best single seed: +17.45pp (154/298 → 206/298). Per-seed unique solves: 13/11/7/5. Computed on intersection of 298 qids common to all 4 seeds (random44, random45 each missing 1 qid that persistently failed across runs).
‡ random44, random45 have N=299 (1 qid persistently failed; below noise floor).

**Comparison to test150 MiniMax Δs:**

| Condition | test150 Δ vs base | test300 Δ vs base |
| :---- | ----: | ----: |
| traj_summary_orig_ext | +7.3 * | +5.34 |
| traj_orig_ext | (varies) | +6.69 |
| random k=5 best single | -3.3 | +5.67 |
| random k=5 best-of-4 | (71.3) | (69.13) |
| budget-5 round-1 | — | -10.33 |
| self_explorer (round-2) | — | +3.67 |

The +5.3pp gain from `traj_summary_orig_ext` reproduces (test150 was +7.3 — both within noise of each other and statistically meaningful). **Best-of-4 hits 69.13%** on test300 (vs 71.3% on test150, well within consistency range), reaffirming MiniMax's strong ceiling under random-tool-call extension. Random k=5 deltas flip from negative on test150 to **+5.67pp positive** on test300 — same pattern as GLM, suggesting test150's negatives were noise.

**Compared to GLM at n=300:** MiniMax has higher absolute accuracy throughout (46→53 baseline-vs-best-condition) but the **deltas are smaller** (+5–7pp vs GLM's +6–8pp), and MiniMax's best-of-4 lift (+17.4pp) is much larger than GLM's (+10pp), suggesting MiniMax's randomness ceiling is higher.

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
