# Notable Assumptions

Running log of non-obvious choices made during autonomous agent work. Each entry includes the decision, the reasoning, and the revert path.

---

## 2026-04-20 — Qwen3.5-122B-A10B full-run SBATCH generation

**Context:** User instructed autonomous submission of Qwen full run once smoke 6697628 passes. Then clarified they only have access to 2 H200 GPUs at a time on `h200_public`, so sharding into two concurrent jobs offers no wall-clock benefit. Reworked to a single 830-query job.

**Current plan:** `sbatch/run_bcp_full_qwen3_5.SBATCH` — single job, all 830 queries via `topics-qrels/bcp/queries.tsv`. Assumptions baked in:

1. **48h walltime** (double the GLM shard walltime). Qwen smoke took ~6h for 10 queries including ~5min cold load → ~35min/query raw is an overestimate from cold-start overhead. More realistic steady-state: ~30-90s/query at `--max-num-seqs 16`. 830 queries ≈ 7-21h. 48h gives headroom + resume buffer.
2. **Partition `h200_public`, 2 GPUs** (same as Qwen smoke). GLM uses `h100_tandon`; Qwen3.5-122B FP8 weights (~122GB) won't fit on 2×H100 80GB.
3. **`--max-num-seqs 32` + `--num-threads 32`** (up from smoke's 16/10). Real throughput lever is server-side `max-num-seqs` — client threads past that just queue. On 2×H200 (282GB - 122GB weights - 15GB overhead ≈ 145GB KV budget) batch 32 fits comfortably at typical trajectory lengths (30-80K tokens, not 131K). Expect ~1.5-1.8× throughput vs batch 16 for this MoE decode. If we see preemption/recompute stalls (KV pressure → vLLM preempts+restarts sequences, which slows things down net), drop to 24.
4. **Context-overflow fix already in client** (`context_threshold=0.75` default + `BadRequestError` handler in `qwen35_client.py`). Smoke pass gates this — if smoke still hits hard-fail sys.exit at 0.75, lower threshold further before full submission.
5. **Auto-submit gate = `sacct` State `COMPLETED`** on smoke 6697628 (no manual validation of trajectory quality beyond the smoke script's exit code + spot-check that 10 trajectories landed). If smoke exits `FAILED/TIMEOUT/CANCELLED`, DO NOT submit full — surface the log and wait for user.
6. **Output dir `runs/bcp/Qwen3-Embedding-8B/full/qwen3.5-122b-a10b/seed0/`** (parallel to GLM path).
7. **Resume-on-restart:** if job hits 48h walltime with queries remaining, resubmit the same SBATCH. Client skips query_ids already present in output dir. No cleanup required.

**Superseded:** earlier `run_bcp_full_qwen3_5_shard_{a,b}.SBATCH` files were deleted after user clarified the 2-GPU constraint. Single-job version is now canonical.

**Revert path:** `scancel <jobid>` and `rm sbatch/run_bcp_full_qwen3_5.SBATCH`. Trajectory files under `seed0/` are resumable — no cleanup needed unless user wants a fresh run.

---

## 2026-04-20 — GLM shard_a missed qid 41 (pre-fix version)

**Context:** GLM shard_a (6686507) completed with 414/415 trajectories. Missing qid 41 hit `BadRequestError: maximum context length 131072` at iteration 48 → 3 consecutive API errors → client `sys.exit(1)` for that query. This is the SAME bug I fixed for qwen smoke AFTER shard_a was already submitted. shard_a ran with pre-fix `glm_client.py`.

**shard_b (6686508)** just started now — fresh Python import picks up current `glm_client.py` which HAS the `BadRequestError` handler + 0.75 context threshold. shard_b's overflow queries should land as `status="context_limit"` gracefully, no hard-fail.

**Plan for qid 41:** after shard_b completes, submit a fill-in job (resume-safe client will skip the 414/415 already done in shard_a's range, and shard_b's output). `sbatch sbatch/run_bcp_full_glm_shard_a.SBATCH` would work since client skips existing — but that's a 5min cold-start for one query. Alternative: build a one-query TSV and a slim SBATCH. Either is acceptable; deferring the choice until shard_b finishes in case other qids also go missing and we want to batch fills together.

**Distribution (shard_a final):** 384 completed, 30 context_limit (graceful — these happened to hit the proactive 0.85 threshold before the client would have crashed; they were not affected by the BadRequestError bug), 1 crashed (qid 41).

---

## 2026-04-20 — GLM full run final state + fill-in submission (job 6708885)

**Context:** shard_a (6686507) COMPLETED 414/415. shard_b (6686508) COMPLETED 414/415. Aggregate: 828/830 trajectories in `runs/bcp/Qwen3-Embedding-8B/full/glm-4.7-flash/seed0/`. Missing: qid 41 (shard_a pre-fix BadRequestError crash) + qid 1219 (shard_b "Request timed out" — transient).

**Status distribution (shard_b complete):** 757 completed, 71 context_limit (graceful, up from shard_a's 30 because shard_b had the `BadRequestError` handler catching overflows the proactive 0.75 threshold missed), 2 missing.

**Fill-in job:** `sbatch/run_bcp_fillin_glm.SBATCH` submitted as job 6708885. Single vLLM cold-start (~5 min) to process just qids 41 + 1219 via `topics-qrels/bcp/queries_fillin_glm.tsv`. 3h walltime (overkill for 2 queries but cheap). `--num-threads 2` since only 2 queries. Writes to same `seed0/` output dir — client resume-safe, won't touch existing 828.

**Expected outcome:** qid 41 will either complete or (more likely) hit context_limit gracefully with the BadRequestError handler now in place; qid 1219 should complete on retry since timeout was transient.

**Revert path:** `scancel 6708885` and `rm sbatch/run_bcp_fillin_glm.SBATCH topics-qrels/bcp/queries_fillin_glm.tsv`. No cleanup needed for trajectories since fill-in just adds to the 828.

---

## 2026-04-20 — Policy shift: all models run on 150-query test slice, not full 830

**Context:** User directive: *"Rather than the smoke test can we just queue up the full qwen test. But only run it on the -query topics-qrels/bcp/queries_test150.tsv test queries. For all models we only need to run it on those 150 test queries."* Qwen smoke 6697628 cancelled mid-queue; no further smoke tests planned.

**New convention:** split folder = `test150` (matches existing `first10`/`full` no-underscore pattern). All future model runs go to `runs/bcp/Qwen3-Embedding-8B/test150/{model}/seed0/` and `evals/bcp/Qwen3-Embedding-8B/test150/{model}/seed0/`.

**Qwen3.5-122B-A10B submission:** `sbatch/run_bcp_test150_qwen3_5.SBATCH` submitted as job 6712988. Differences from deleted full variant:
- Walltime 12h (scaled from 48h at 150/830 ≈ 18% + safety margin)
- `--query topics-qrels/bcp/queries_test150.tsv`
- Split `test150` in output path
- All other flags identical (TP=2, MAX_NUM_SEQS=32, --num-threads 32, qwen3 parsers, vLLM nightly container)

**Open question on GLM eval 6712301:** currently evaluating all 830 GLM trajectories. Since test150 is (presumably) a subset of the 830, the 830 eval's results are a superset and can be filtered post-hoc to the 150 test qids for cross-model comparison. Letting it run rather than cancelling — cheaper to filter than to rerun. Flag for user if they want to cancel.

**Revert path:** `scancel 6712988`, `rm sbatch/run_bcp_test150_qwen3_5.SBATCH`. Trajectories under `runs/bcp/Qwen3-Embedding-8B/test150/qwen3.5-122b-a10b/seed0/` are resumable.

---

## 2026-04-20 — GLM round-2 pipeline on test150 (self-summarization, original_messages only)

**Context:** User directive for two round-2 modes on the 150 test qids:
1. `traj_orig_ext` — full first-run trajectory prepended (raw `original_messages`)
2. `traj_summary_orig_ext` — pre-computed summary prepended (from `original_messages`)

Explicitly NOT using `traj_ext` / `traj_summary_ext` (template-tag variants) — only the `_orig` versions that emit raw JSON-dump of `original_messages`.

**Pipeline:**
1. **Summarize** (job 6713445, h100_tandon, 3h walltime, TP=2): `summarize_trajectories.py` self-summarizes GLM-4.7-Flash first-run trajectories for the 150 test qids. Source = `runs/bcp/Qwen3-Embedding-8B/full/glm-4.7-flash/seed0/` (the closed 830-run), filtered via `--queries-tsv queries_test150.tsv`. Output = `summaries/glm-4.7-flash_test150.jsonl`.
2. **traj_orig_ext** (job 6713447, h100_tandon, 12h, TP=2): runs in parallel with summarize. Prepends full first-run messages (truncation defaults: reasoning 3000, tool output 5000, overall 500000 chars — hung-ting's defaults made explicit).
3. **traj_summary_orig_ext** (job 6713449, h100_tandon, 12h, TP=2): submitted with `--dependency=afterok:6713445`. Uses summary JSONL from step 1.

**Decisions confirmed by user:**
- **Summarizer model** = `zai-org/GLM-4.7-Flash` itself (self-summarization, matches hung-ting's gpt-oss-120b pattern where agent summarizes its own trajectory)
- **Output layout** = `runs/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/{traj_orig_ext_seed0,traj_summary_orig_ext_seed0}/` (all GLM conditions under one agent-model folder; run_name disambiguates the mode)
- **Source trajectories** = the existing 830-trajectory first run (no regeneration); loader filters to test150 qids at load time

**Expected outcome:** round-2 accuracy should improve over round-1's 41.33% if the prior-trajectory signal actually helps. `traj_orig_ext` may have more `context_limit` graceful degrades than round-1 because the prepended messages alone can be ~60-80K tokens. `traj_summary_orig_ext` should not (summaries capped at 8192 tokens by summarizer's `--max-tokens`).

**Revert path:** `scancel 6713445 6713447 6713449` and `rm sbatch/run_bcp_summarize_glm_test150.SBATCH sbatch/run_bcp_test150_glm_traj_orig.SBATCH sbatch/run_bcp_test150_glm_traj_summary_orig.SBATCH`. Trajectory/summary outputs are resumable.

---

## 2026-04-20 — Qwen3.5-122B OOM fix: CPU embedding (CUDA_VISIBLE_DEVICES=-1)

**Context:** Jobs 6712988 (batch=32) and 6736408 (batch=24) both OOM'd at ~9-25 min, producing only 16/150 trajectories each. Root cause: `Qwen3-Embedding-8B` loaded via `sentence-transformers` in the host Python client grabbed ~16 GiB on GPU 0 AFTER vLLM had already profiled KV cache. vLLM's 20GB EMBED_RESERVE_GB reservation only reduced its utilization budget — it didn't prevent the client from physically allocating into that reserved space. Result: ~2 GiB KV cache available → saturated at first batch → MoE FusedMoeRunner workspace OOM.

**Fix:** `export CUDA_VISIBLE_DEVICES=-1` before `python qwen35_client.py` forces CPU-only for the host client. vLLM in Singularity is unaffected (already running). `EMBED_RESERVE_GB` lowered to 5 (no GPU for embedding, just CUDA overhead).

**GLM comparison:** GLM at 7B/H100 has 80GB - 7GB weights = 73GB KV budget — embedding fits in the 25GB reserve easily. For MoE ≥100B on H200, always use CPU embedding.

**Affected SBATCHes:** `run_bcp_test150_qwen3_5.SBATCH` (resubmitted as 6744333), `run_bcp_test150_qwen3_5_traj_orig.SBATCH`, `run_bcp_test150_qwen3_5_traj_summary_orig.SBATCH`.

**GLM round-2 results (2026-04-20):** traj_orig_ext 44.00%, traj_summary_orig_ext 53.33% (vs round-1 41.33%). Summary variant +12pp. Full-trajectory variant marginally better (+2.7pp) despite higher context_limit rate (9 vs 5).

---

## 2026-04-21 — Qwen3.5-122B baseline partial (145/150) + fill-in job 6761848

**Context:** Baseline run 6744333 COMPLETED after 6:42 but produced only 145/150 trajectories. Five qids failed with "3 consecutive API errors: Request timed out" — qids 3, 311, 516, 527, 943. Root cause: transient vLLM server timeouts during deep multi-hop queries (iters 24-30), same pattern as GLM qid 1219 in the full run.

**Fill-in:** `sbatch/run_bcp_fillin_qwen3_5.SBATCH` submitted as job 6761848. Targets only the 5 missing qids via `topics-qrels/bcp/queries_fillin_qwen3_5.tsv`. `--max-num-seqs 5`, 3h walltime, same h200_public 2×H200 config with `CUDA_VISIBLE_DEVICES=-1`.

**Round-2 dependency chain:**
- `run_bcp_test150_qwen3_5_traj_orig.SBATCH` → job 6761849, `--dependency=afterok:6761848`
- `run_bcp_summarize_qwen3_5_test150.SBATCH` → job 6761850, `--dependency=afterok:6761848`
- `run_bcp_test150_qwen3_5_traj_summary_orig.SBATCH` → submit after 6761850 completes

**Revert path:** `scancel 6761848 6761849 6761850 6765380`, `rm sbatch/run_bcp_fillin_qwen3_5.SBATCH topics-qrels/bcp/queries_fillin_qwen3_5.tsv`. No cleanup needed for trajectories.

**Fix — summarize job 6761850 FAILED (Mamba cache init):** vLLM default `max_num_seqs=1024` exceeded available Mamba cache blocks (575) for Qwen3.5-122B on 2×H200. Error: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (575)`. Fixed `run_bcp_summarize_qwen3_5_test150.SBATCH` by adding `MAX_NUM_SEQS=512` + `--max-num-seqs $MAX_NUM_SEQS` to vLLM serve. Resubmitted as **6765380**. Root cause: Qwen3.5 uses hybrid attention with Mamba-like state caches; summarize SBATCH omitted `--max-num-seqs` since the other SBATCHes inherited it from the OOM-fix iteration.

**Fix — summarize job 6765380 FAILED (missing --queries-tsv):** vLLM started correctly with `--max-num-seqs 512` fix, but `summarize_trajectories.py` requires `--queries-tsv` (mandatory arg). The Qwen summarize SBATCH was created without this arg (GLM summarize SBATCH had it). Added `--queries-tsv "topics-qrels/${dataset}/queries_test150.tsv"` to the python call. Resubmitted as **6765564**. Also submitted traj_orig_ext eval as **6765565** (`run_bcp_eval_qwen3_5_test150_traj_orig.SBATCH`).

---

## 2026-04-21 — MiniMax-M2.5 TP=2 smoke test on 2×H200 (job 6789048)

**Context:** User requested MiniMax-M2.5 smoke tests + base run + traj_orig/summary pipeline after Qwen finishes. Qwen is fully complete. MiniMax-M2.5 official docs recommend TP=4 (4×96GB GPU), but the user only has 2×H200 (282GB total VRAM) via `h200_public`.

**Feasibility analysis:** Model weights = 215GB on disk (FP8). 282GB - 215GB = 67GB remaining for KV cache at `gpu_memory_utilization≈0.97`. With `max_model_len=65536` (vs 196608 max) and `max_num_seqs=8` (conservative), the KV footprint is small enough that 67GB is plausible. Architecture is TP=2 compatible: 48 attention heads and 256 experts both divisible by 2.

**Smoke test submitted:** `sbatch/run_bcp_first10_minimax.SBATCH` → job **6789048**. 10 queries on `queries_first10.tsv`, h200_public, 2×H200, 6h walltime.

**Key assumptions:**
1. **TP=2 will not OOM** during model loading. If it does, the fix is requesting `h200_tandon` access for 4-GPU partition via HPC portal.
2. **vLLM nightly has `minimax_m2_append_think` parser** — model card says commit `cf3eacfe58fa9e745c2854782ada884a9f992cf7` required. Container built ~2026-04-08; if parser is missing, vLLM will error at startup with unknown parser name.
3. **`minimax_client.py` based on `qwen35_client.py`** — removed `enable_thinking` chat_template_kwargs (MiniMax emits reasoning by default, no kwarg needed); changed tool-call leak detection to `<minimax:tool_call>`.
4. **No `original_messages` round-2 until smoke passes** — traj_orig and summarize SBATCHes queued after smoke COMPLETED.

**Next steps if smoke passes:** submit `run_bcp_test150_minimax.SBATCH` (already written), then eval, summarize, traj_orig, traj_summary.

**Revert path:** `scancel 6789048`, `rm sbatch/run_bcp_first10_minimax.SBATCH sbatch/run_bcp_test150_minimax.SBATCH search_agent/minimax_client.py`. No trajectory cleanup needed.
