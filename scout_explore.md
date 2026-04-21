# Scout/Explore Experiment Results — BrowseComp-Plus test150

**Setup:** 150-query test slice, FAISS retriever (Qwen3-Embedding-8B), LLM-as-judge (Qwen3-32B, TP=2).

---

|  | GLM-4.7-Flash \+ Qwen3-Embedding-8B | Inject at Start |  |  |
| :---- | :---- | ----- | ----- | ----- |
|  |  | Acc | Recall | \# calls |
| Base (No Plan) | Base agent | 44.0 | 55.4 | 21.6 |
| **Original Messages** |  |  |  |  |
| Base (No Plan) | Second run takes the output from first run, full trajectory | 44.0 | 20.3 | 4.3 |
| **Base (Summary)** | **Second run takes the summary of first run trajectory and do the task again** | **53.3** | **52.5** | **12.7** |

---

|  | Qwen3.5-122B-A10B \+ Qwen3-Embedding-8B | Inject at Start |  |  |
| :---- | :---- | ----- | ----- | ----- |
|  |  | Acc | Recall | \# calls |
| Base (No Plan) | Base agent | 42.7 | 54.3 | 21.8 |
| **Original Messages** |  |  |  |  |
| Base (No Plan) | Second run takes the output from first run, full trajectory | 46.3 | 0.9 | 1.1 |
| **Base (Summary)** | **Second run takes the summary of first run trajectory and do the task again** | **48.3** | **56.5** | **14.4** |

*Qwen traj_summary N=149: qid 1193 hard context overflow (121K token summary prompt).*
*GLM baseline filtered from 830-query full run eval to test150 qids.*

---

## Key Observations

**Summary prepend consistently wins.** Both models improve more from summarized trajectory than full trajectory:
- GLM: baseline→traj_summary +9.3pp; baseline→traj_orig 0pp
- Qwen: baseline→traj_summary +5.7pp; baseline→traj_orig +3.6pp

**traj_orig_ext collapses search steps.** Full trajectory prepend fills the context window, leaving little room:
- GLM traj_orig: 4.3 avg steps (vs 21.6 baseline), recall drops 55% → 20%
- Qwen traj_orig: 1.1 avg steps (vs 21.8 baseline), recall collapses to 0.9% — Qwen's stronger reasoning extracts answer directly from prepended evidence without new searches

**Qwen baseline has high context_limit rate (42%).** 63/150 queries hit context_limit on the standard baseline run. Qwen3.5-122B generates longer reasoning chains than GLM (9% context_limit rate).

**Summary restores search quality.** traj_summary_orig_ext recovers reasonable search depth (12.7/14.4 steps) and recall (52/57%) while still improving accuracy, suggesting the summary provides useful orientation without crowding out retrieval.

---

## Run Metadata

| Model | Condition | Jobs | Notes |
|-------|-----------|------|-------|
| GLM-4.7-Flash | baseline | 6686507+6686508 (full), fillin 6708885 | 830-query full run; test150 filtered post-hoc |
| GLM-4.7-Flash | traj_orig_ext | 6713447, eval 6744057 | |
| GLM-4.7-Flash | traj_summary_orig_ext | summarize 6713445, run 6713449, eval 6736427 | |
| Qwen3.5-122B-A10B | baseline | 6744333 (3rd attempt), fillin 6761848 | 2 OOM crashes (6712988, 6736408); fix: `CUDA_VISIBLE_DEVICES=-1` |
| Qwen3.5-122B-A10B | traj_orig_ext | 6761849, eval 6765565 | |
| Qwen3.5-122B-A10B | traj_summary_orig_ext | summarize 6765564, run 6765969, eval 6767388 | 2 summarize failures before fix (Mamba cache + missing --queries-tsv) |
| Qwen3.5-122B-A10B | baseline eval | 6767389 | |
