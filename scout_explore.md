# Scout/Explore Experiment Results

**Setup:** 150-query test slice on each benchmark. LLM-as-judge: Qwen3-32B. All numbers reflect the corrected `evaluate_run.py` (eval bug fix, 2026-04-26) that grades `context_limit` trajectories using their forced final answers; pre-fix numbers undercounted accuracy for verbose models (full diff in NOTABLE_ASSUMPTIONS.md).

---

## BrowseComp-Plus (BCP) — Qwen3-Embedding-8B retriever

| Model | Condition | Acc | Recall | # calls |
| :---- | :---- | ----: | ----: | ----: |
| GLM-4.7-Flash (30B) | Base (No Plan) — base agent | 48.0 | 55.4 | 22.0 |
| GLM-4.7-Flash (30B) | Base (No Plan) — full first-run trajectory injected | 47.3 | 20.3 | 4.3 |
| GLM-4.7-Flash (30B) | Base (Summary) — first-run trajectory summary injected | 53.3 | 52.5 | 12.7 |
| GLM-4.7-Flash (30B) | Base (Selected Tool Calls) — Gemini-selected k=5 excerpts | 46.7 | 29.1 | 8.6 |
| GLM-4.7-Flash (30B) | Base (Random Tool Calls) — random k=5 excerpts (ablation) | 47.3 | 34.6 | 9.7 |
| Qwen3.5-122B-A10B | Base (No Plan) — base agent | 45.3 | 54.3 | 21.8 |
| Qwen3.5-122B-A10B | Base (No Plan) — full first-run trajectory injected | 48.4 | 0.0 | 0.1 |
| Qwen3.5-122B-A10B | Base (Summary) — first-run trajectory summary injected | 48.3 | 56.5 | 14.4 |
| **TODO: Qwen3.5 selected tool calls (running, h200 PENDING)** |  |  |  |  |
| **TODO: Qwen3.5 random tool calls (chained after selected)** |  |  |  |  |
| MiniMax-M2.5 (229B) | Base (No Plan) — base agent | 48.7 | 56.9 | 15.3 |
| MiniMax-M2.5 (229B) | Base (No Plan) — full first-run trajectory injected | 54.0 | 20.0 | 3.2 |
| MiniMax-M2.5 (229B) | Base (Summary) — first-run trajectory summary injected | 56.0 | 56.7 | 10.0 |
| **TODO: MiniMax selected tool calls (running, h200 PENDING)** |  |  |  |  |
| **TODO: MiniMax random tool calls (chained after selected)** |  |  |  |  |

*Caveats:* Qwen3.5 traj_orig N=134, traj_summary N=149 — a few qids missing from the eval pool (one hit a hard 121K-token context overflow on the summary prompt). GLM baseline filtered from 830-query full run eval to test150 qids. Context_limit rates: GLM 9% / Qwen3.5 42% / MiniMax 71% baseline (the 65536-token cap drives MiniMax's tail; eval fix forces these to be graded rather than auto-failed).

---

## FRAMES — BGE-M3 retriever, Upstash Wikipedia

| Model | Condition | Acc | Recall | # calls |
| :---- | :---- | ----: | ----: | ----: |
| GLM-4.7-Flash (30B) | Base (No Plan) — base agent | 44.7 | 70.6 | 27.7 |
| GLM-4.7-Flash (30B) | Base (No Plan) — full first-run trajectory injected | 46.2 | 15.9 | 8.5 |
| GLM-4.7-Flash (30B) | Base (Summary) — first-run trajectory summary injected | 51.3 | 52.2 | 14.9 |
| **TODO: Qwen3.5 baseline (cancelled to free h200_public for BCP, will resubmit)** |  |  |  |  |
| **TODO: Qwen3.5 traj_orig_ext** |  |  |  |  |
| **TODO: Qwen3.5 traj_summary_orig_ext** |  |  |  |  |
| **TODO: MiniMax baseline (cancelled to free h200_public for BCP, will resubmit)** |  |  |  |  |
| **TODO: MiniMax traj_orig_ext** |  |  |  |  |
| **TODO: MiniMax traj_summary_orig_ext** |  |  |  |  |

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
