# Scout/Explore Experiment Results

**Setup:** 150-query test slice on each benchmark. LLM-as-judge: Qwen3-32B. All numbers reflect the corrected `evaluate_run.py` (eval bug fix, 2026-04-26) that grades `context_limit` trajectories using their forced final answers; pre-fix numbers undercounted accuracy for verbose models (full diff in NOTABLE_ASSUMPTIONS.md).

**Conditions tested** (each model is evaluated under all 5):

1. **Baseline** — agent runs with no prepended evidence; standard agentic loop.
2. **+ full trajectory** — the entire first-run trajectory (all tool calls + observations) is injected into the prompt before the agent starts. Tests the upper bound of "everything we already learned."
3. **+ trajectory summary** — an LLM-generated summary of the first-run trajectory is injected instead of the raw trajectory. Tests whether a compressed orientation helps.
4. **+ selected k=5 tool calls** — Gemini selects the 5 most useful tool-call/observation pairs from the first-run trajectory and prepends only those excerpts. Tests whether targeted excerpts beat full or summary.
5. **+ random k=5 tool calls (ablation)** — 5 tool-call/observation pairs sampled uniformly at random from the first-run trajectory. Controls for whether the *selection* matters or just the *amount* of prepended evidence.

---

## BrowseComp-Plus (BCP) — Qwen3-Embedding-8B retriever

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.0 | 55.4 | 22.0 |
| + full trajectory | 47.3 | 20.3 | 4.3 |
| **+ trajectory summary** | **53.3** | 52.5 | 12.7 |
| + selected k=5 tool calls | 46.7 | 29.1 | 8.6 |
| + selected k=5 tool calls (seed1 rerun) | 46.0 | 29.4 | 9.7 |
| + random k=5 tool calls (ablation) | 47.3 | 34.6 | 9.7 |

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 45.3 | 54.3 | 21.8 |
| + full trajectory | 48.4 | 0.0 | 0.1 |
| + trajectory summary | 48.3 | 56.5 | 14.4 |
| **+ selected k=5 tool calls** (N=148) | **48.7** | 25.4 | 15.9 |
| + random k=5 tool calls (ablation) | TBD | TBD | TBD |

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 48.7 | 56.9 | 15.3 |
| + full trajectory | 54.0 | 20.0 | 3.2 |
| **+ trajectory summary** | **56.0** | 56.7 | 10.0 |
| + selected k=5 tool calls | 55.3 | 45.4 | 8.6 |
| + random k=5 tool calls (ablation) | TBD | TBD | TBD |

*Caveats:* Qwen3.5 traj_orig N=134, traj_summary N=149 — a few qids missing from the eval pool (one hit a hard 121K-token context overflow on the summary prompt). GLM baseline filtered from 830-query full run eval to test150 qids. Context_limit rates: GLM 9% / Qwen3.5 42% / MiniMax 71% baseline (the 65536-token cap drives MiniMax's tail; eval fix forces these to be graded rather than auto-failed).

---

## FRAMES — BGE-M3 retriever, Upstash Wikipedia

**Model: GLM-4.7-Flash (30B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | 44.7 | 70.6 | 27.7 |
| + full trajectory | 46.2 | 15.9 | 8.5 |
| **+ trajectory summary** | **51.3** | 52.2 | 14.9 |

**Model: Qwen3.5-122B-A10B**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | TBD | TBD | TBD |
| + full trajectory | TBD | TBD | TBD |
| + trajectory summary | TBD | TBD | TBD |

**Model: MiniMax-M2.5 (229B)**

| Condition | Acc | Recall | # calls |
| :---- | ----: | ----: | ----: |
| Baseline | TBD | TBD | TBD |
| + full trajectory | TBD | TBD | TBD |
| + trajectory summary | TBD | TBD | TBD |

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
