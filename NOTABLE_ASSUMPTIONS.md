# Notable Assumptions

Running log of non-obvious choices made during autonomous agent work. Each entry includes the decision, the reasoning, and the revert path.

---

## 2026-05-03 — Qwen3.5 random_tools seed45 ended at 147/150 (last 3 hung)

**Context:** Job 7741702 (Qwen3.5-122B-A10B random_tools_seed45) ran 06:52:38 and SLURM marked COMPLETED — log ended `=== Done ===`. 147/150 trajectories landed; last 3 queries hung in agent loop ~80 minutes before client exit. Same exact pattern as seed44 (148/150) and the 2026-04-28 selected_tools incident — recurring Qwen3.5 issue where a few queries enter unbreakable agent-loop spirals.

**Action:** No autonomous resubmit yet — same posture as seed44. Eval can run at N=147 once user decides on N=147/148 vs chase-all-150. seed43 recovery (7811896) is next in queue.

**Revert path:** `sbatch --export=ALL,SEED=45 sbatch/run_bcp_test150_qwen3_5_random_tools.SBATCH` to recover the missing 3.

---

## 2026-05-02 — Qwen3.5 random_tools seed44 ended at 148/150 (last 2 hung)

**Context:** Job 7741701 (Qwen3.5-122B-A10B random_tools_seed44) ran 06:05:29 and SLURM marked COMPLETED — log ended with `=== Done ===` cleanly, so the client exited naturally. But only 148/150 trajectories landed; final 2 queries hung in the agent loop for ~70 minutes before the iter cap (or some other terminal state) released them without writing trajectories. Same exact pattern as the 2026-04-28 Qwen3.5 selected_tools incident at 148/150.

**Action:** No autonomous resubmit yet — eval can run at N=148 (state machine condition 1 says "150 trajectories"; this is 148, so technically not yet eligible). Flag to user for decision: (a) eval at N=148 like the prior incident, or (b) resubmit again to chase the last 2.

**Revert path:** `sbatch --export=ALL,SEED=44 sbatch/run_bcp_test150_qwen3_5_random_tools.SBATCH` to recover the missing 2 (idempotent client picks them up).

---

## 2026-05-02 — Qwen3.5 random_tools seed43 vLLM crash at 135/150, resubmitted

**Context:** Job 7741700 (Qwen3.5-122B-A10B random_tools_seed43) ran for 03:21:50 and SLURM marked it COMPLETED, but only 135/150 trajectories landed. Tail of `/scratch/afw8937/logs/qwen3.5-122b-a10b_test150_random_7741700.out` shows 15 qids (275, 149, 81, 747, 309, 962, 1107, 600, 174, 389, 1077, 427, 132, +2 more) all failed with `[iter N] chat.completions error (3 consecutive): Connection error.` — vLLM died mid-run, classic Qwen3.5+H200 instability (same family as the recurring TMA crashes).

**Action:** `sbatch --export=ALL,SEED=43 sbatch/run_bcp_test150_qwen3_5_random_tools.SBATCH` → job 7811896. Idempotent client will pick up the 15 missing qids.

**Revert path:** N/A — recovery only adds files; existing 135 trajectories untouched.

---

## 2026-04-28 — Qwen3.5 selected_tools cancelled at 148/150 (last 2 queries hung)

**Context:** Job 7372606 (Qwen3.5 BCP selected_tools resume after the earlier vLLM TMA crash) ran for 5h14m. After hitting 148/150 trajectories, the last 2 queries cycled in a long agent loop for ~1.5h with no progress. vLLM stayed healthy (240 tok/s gen, 11% KV cache), so the issue was the agent client's iteration loop — likely repetitive search-failure cycles that don't trip the iter cap of 100.

**Action:** `scancel 7372606`. Submitted eval on the 148 trajectories we have (`sbatch run_bcp_eval_qwen3_5_test150_selected_tools.SBATCH` → 7391578). Submitted Qwen3.5 random_tools (7391579) to continue the experiment matrix.

**Caveat:** Qwen3.5 selected_tools result will be on N=148 instead of 150. Note in scout_explore.md row.

**Revert path:** If the 2 missing trajectories are critical, resubmit `run_bcp_test150_qwen3_5_selected_tools.SBATCH` later — idempotent client picks up just those 2.

---

## 2026-04-28 — eval `max_model_len` 4096 → 16384 (silent prompt drop bug)

**Context:** GLM seed1 selected_tools eval (job 7373745) reported 37.21% acc on **only 86/150 trajectories**. The eval log showed:

```
Error running vLLM batch 64//64: The decoder prompt (length 6772) is longer than the maximum model length of 4096.
```

vLLM throws this as a non-fatal batch error — the batch is dropped from results but eval continues. With `max_model_len=4096` (`evaluate_run.py:460`) and seed1 having 10.5 search calls/query (vs seed0's 8.6), longer trajectory_history representations in the judge prompt exceeded the limit on 64/150 queries. Seed0 (8.6 searches/query) happened to fit and got 150/150 evaluated cleanly at 46.7%, so this bug was latent.

**Fix:** Bumped `max_model_len` to 16384 in `scripts_evaluation/evaluate_run.py:460`. Qwen3-32B judge can handle that easily on 2×H100 with `gpu_memory_utilization=0.85`.

**Resubmitted GLM seed1 eval as 7374231** after deleting stale eval JSONs.

**Risk:** All historical eval numbers in scout_explore.md were produced under max_model_len=4096. If any past run had >4096-token judge prompts, those queries were silently dropped. For models running shorter trajectories (low search count) this likely didn't bite, but verbose runs (Qwen3.5 + MiniMax with 21+ search calls on baseline) could've silently lost prompts. **Worth a re-eval pass on all completed conditions** with the new ceiling — but that's user-call. For now flag in scout_explore.md once seed1 number lands.

**Revert path:** `git diff` evaluate_run.py shows the one-line change. Revert if 16K causes OOM on the 2×H100 judge.

---

## 2026-04-28 — Qwen3.5 random_tools accepted at N=130 after 3 vLLM crashes

**Context:** Three vLLM crashes in a row on Qwen3.5 random_tools, all `torch.AcceleratorError: CUDA error: misaligned address`:
- 7391579: ran ~40min → 24/150 trajectories
- 7408688: ran 3h → 128/150 trajectories
- 7422866: ran 19min → 130/150 trajectories

After 3 attempts, the trajectory file count is plateauing in the 128-130 range. The remaining ~20 queries appear to consistently trigger the cutlass MoE GEMM kernel issue early in each restart. Continuing retries is unlikely to fully complete the 150.

**Action:** Submitted eval (7424615) on N=130 trajectories. Result will be reported in scout_explore.md with N=130 caveat (similar to Qwen3.5 selected_tools at N=148).

**Pattern (confirmed across multiple Qwen3.5 runs on h200_public + vLLM nightly):**
- selected_tools: 21 → 148/150 (cancelled after 1.5h hang on last 2)
- random_tools: 24 → 128 → 130/150 (3 crashes)

**Suggested fix for future runs (not done now):** try `--enforce-eager` to disable cuda graphs (likely the trigger), or pin to a stable vLLM release instead of nightly. Accept ~30% throughput cost.

---

## 2026-04-28 — Qwen3.5 random_tools resubmit after vLLM CUDA misaligned address crash (job 7391579)

**Context:** Qwen3.5 random_tools (job 7391579) ran 39:31, exited COMPLETED 0:0, but only produced 24/150 trajectories. Log shows the same `torch.AcceleratorError: CUDA error: misaligned address` pattern as the earlier Qwen3.5 selected_tools crash. vLLM internal MoE GEMM kernel error on H200 — appears to be a recurring Qwen3.5 + nightly vLLM + h200 issue.

**Action:** Resubmitted as 7408688. Idempotent client will pick up remaining 126 queries.

**Pattern observed across runs:**
- Qwen3.5 selected_tools (original 7369110): vLLM TMA crash at ~30min → 21/150
- Qwen3.5 selected_tools (resume 7372606): completed 148/150 (last 2 hung at iter cap)
- Qwen3.5 random_tools (7391579): vLLM CUDA misaligned crash at ~30min → 24/150
- → suggests vLLM stack hits this crash early in any Qwen3.5 run on h200; resume from the earlier output dir is the right strategy.

**Revert path:** `scancel 7408688`. Existing 24 trajectories are usable.

---

## 2026-04-28 — Qwen3.5 BCP selected_tools resubmit after silent vLLM TMA crash (job 7369110)

**Context:** Job 7369110 (Qwen3.5 BCP selected_tools, h200_public 2 GPU) ran for 38min and exited cleanly (`COMPLETED 0:0`), but only produced 21/150 trajectories. Tail of log shows 129 queries failing with `Connection error` after vLLM internal crash at 01:23:01:

```
Error: Failed to initialize the TMA descriptor 716
RuntimeError: [TensorRT-LLM][ERROR] Failed to initialize cutlass TMA WS grouped gemm.
torch.AcceleratorError: CUDA error: misaligned address
```

vLLM `multiproc_executor.py` workers died on cutlass MoE GEMM kernel for sm90 (H200). The API server stayed up but couldn't service requests, so the client got 3 consecutive connection errors per query and aborted each (after 3 retries, marks the qid failed and moves on, never writes JSON). Slurm sees client exit 0.

**Resubmitted** as 7372606. `qwen35_client.py:665-679` is idempotent — scans output dir, skips qids already in `run_*.json` files. Will pick up remaining 129.

**Why not a different mitigation:** Qwen3.5 baseline + traj_orig + traj_summary all completed 150 trajectories on this same h200/vLLM stack, so the TMA crash is not deterministic for this model — likely transient (specific input shape + cuda graph state). Plain resubmission is the right first try. If 7372606 also crashes mid-run, consider `--enforce-eager` to disable cuda graphs, accepting ~30% throughput loss.

**Revert path:** `scancel 7372606`. Existing 21 trajectories are usable; a fresh run would just resume from there.

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

**Smoke result (job 6789048, COMPLETED, 16 min):** TP=2 on 2×H200 confirmed feasible. All 10/10 trajectories ran with 15-19 searches each. All 10 hit `context_limit` (not `completed`) because `max_model_len=65536` fills at ~49K tokens with heavy search. KV cache peaked at 69% — healthy headroom. Model loaded and served correctly; `minimax_m2_append_think` and `minimax_m2` parsers both present in container.

**Context-limit note:** All smoke trajectories show `status=context_limit`. This is expected given the 65536 token ceiling. The client's `_force_final_answer` handler fires and the model still produces a final answer. Accepted for now; test150 baseline uses the same 65536 limit. If accuracy suffers vs GLM/Qwen3.5 due to truncation, a re-run at 98304 may be warranted.

**test150 baseline submitted:** job **6798680** (`run_bcp_test150_minimax.SBATCH`). 150 queries, h200_public, 2×H200, 24h walltime, MAX_NUM_SEQS=16.

**test150 baseline COMPLETED** (1h35m): 150/150 trajectories. Status: 44 `completed` + 106 `context_limit`. Avg 15.3 searches/query (min=5, max=20). Context-limit rate (71%) consistent with smoke test — 65536 token ceiling fills quickly under heavy search. No missing qids; no fill-in needed.

**Round-2 jobs submitted in parallel:**
- traj_orig_ext → job **6811307** (`run_bcp_test150_minimax_traj_orig.SBATCH`)
- summarize → job **6811308** (`run_bcp_summarize_minimax_test150.SBATCH`)
- traj_summary_orig_ext → will submit once 6811308 completes and summaries/minimax-m2.5_test150.jsonl exists

**traj_orig_ext COMPLETED** (job 6811307, 44:25): 150/150 trajectories. Status: 93 `completed` + 57 `context_limit`. Avg search calls: **0.0** — MiniMax (229B) reads the prepended full trajectory and answers directly without issuing any new searches, identical collapse behavior to Qwen3.5-122B in traj_orig (which had 1.1 avg steps). The prepended `original_messages` context alone fills context enough to trigger 57 context_limits even without searches.

**summarize COMPLETED** (job 6811308, 18:43): 150/150 summaries written to `summaries/minimax-m2.5_test150.jsonl`.

**traj_summary_orig_ext submitted:** job **6825882** (`run_bcp_test150_minimax_traj_summary_orig.SBATCH`), h200_public, 2×H200, 24h walltime.

**traj_summary_orig_ext job 6825882 CANCELLED** — port conflict on gh116 (port 8000 occupied by co-tenant job); job also hung 20+ min past expected exit (likely overlay unmount deadlock after trap kill). **Fix applied:** `sbatch/run_bcp_test150_minimax_traj_summary_orig.SBATCH` now computes `PORT=$((20000 + SLURM_JOB_ID % 10000))` so each job binds a unique port. **Resubmitted:** job **6836427**, COMPLETED on gh114 (1:02:49).

**traj_summary_orig_ext COMPLETED** (job 6836427, 1:02:49): 150/150 trajectories. Status: 108 `completed` + 42 `context_limit`. Avg search calls: **0.0** — same zero-search behavior as traj_orig. MiniMax (229B) reads prepended summary and answers directly without searching. Notably better `completed` rate than both baseline (44) and traj_orig (93), as the shorter summary leaves more context budget for reasoning.

**MiniMax-M2.5 pipeline COMPLETE.** All three test150 round-2 runs finished:
| Condition | Status | completed | context_limit | avg searches |
|-----------|--------|-----------|---------------|--------------|
| baseline (seed0) | DONE | 44 | 106 | 15.3 |
| traj_orig_ext (6811307) | DONE | 93 | 57 | 0.0 |
| traj_summary_orig_ext (6836427) | DONE | 108 | 42 | 0.0 |

**Eval jobs submitted (2026-04-26):**
- baseline eval → job **7232739** (`run_bcp_eval_minimax_test150_baseline.SBATCH`)
- traj_orig_ext eval → job **7232740** (`run_bcp_eval_minimax_test150_traj_orig.SBATCH`)
- traj_summary_orig_ext eval → job **7232741** (`run_bcp_eval_minimax_test150_traj_summary_orig.SBATCH`)
All three on h100_tandon, 2×H100, 2h walltime, Qwen3-32B judge.

**Eval results (2026-04-26):**
| Condition | Acc | Recall | avg searches |
|-----------|-----|--------|--------------|
| baseline (seed0) | 27.3% | 56.9% | 15.3 |
| traj_orig_ext_seed0 | 48.0% | 20.0% | 3.2 |
| traj_summary_orig_ext_seed0 | 52.0% | 56.7% | 10.0 |

Key findings: MiniMax baseline (27.3%) is lower than GLM (44%) and Qwen3.5 (42.7%). traj_orig gives +20.7pp accuracy gain but recall collapses (57%→20%). traj_summary achieves +24.7pp accuracy gain while maintaining recall (56.7% ≈ baseline). scout_explore.md updated.

---

## 2026-04-26 — Bug fix: evaluate_run.py drops forced final answers

**Bug:** `scripts_evaluation/evaluate_run.py:546` had `if response == "" or not is_completed:` which auto-failed any trajectory with `status != "completed"` even when a non-empty `output_text` response was present. The clients deliberately inject *"You have now reached the maximum context length... provide your final answer now"* on context_limit, and the model produces a real answer that gets stored as the last `output_text` entry — but the eval threw it away.

**Impact (MiniMax baseline example):** 106/150 trajectories were `context_limit`; all auto-failed. On the 44 that completed, accuracy was 93.2%, but headline number reported as 27.3%. Bias scaled with verbosity: GLM (9% ctx_limit) lightly affected, Qwen3.5 (42%) moderately, MiniMax (71%) severely.

**Fix:** Removed the `or not is_completed` clause. The `response == ""` check still skips trajectories with no extractable answer.

**Re-eval submitted (2026-04-26, jobs 7242846–7242854):**
| Job | Model | Condition |
|-----|-------|-----------|
| 7242846 | glm-4.7-flash | full (830) baseline |
| 7242847 | glm-4.7-flash | test150 traj_orig |
| 7242848 | glm-4.7-flash | test150 traj_summary |
| 7242849 | qwen3.5-122b-a10b | test150 baseline |
| 7242850 | qwen3.5-122b-a10b | test150 traj_orig |
| 7242851 | qwen3.5-122b-a10b | test150 traj_summary |
| 7242852 | minimax-m2.5 | test150 baseline |
| 7242853 | minimax-m2.5 | test150 traj_orig |
| 7242854 | minimax-m2.5 | test150 traj_summary |

Pre-fix evals backed up to `evals.backup_pre_force_answer_fix_2026-04-26/`.

---

## 2026-04-26 — GLM selected_tool_calls round-2 condition submitted

GLM-4.7-Flash test150 round-2 "selected_tool_calls" run submitted as job **7244602**. Uses Gemini-selected tool-call excerpts from `selected_tool_calls/selected_tool_calls_glm_use_original_messages.jsonl` (830 entries, `excerpt` field) as the trajectory context, plugged into the existing `--trajectory-summary-file` path with `--planning-trigger traj_summary_orig_ext` and `--query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY` — the loader at `search_agent/trajectory_utils.py:_load_trajectory_summaries` accepts both `summary` and `excerpt` fields. SBATCH: `sbatch/run_bcp_test150_glm_selected_tools.SBATCH`. Output dir: `runs/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/selected_tools_seed0/`.


---

## 2026-04-26 — BCP re-eval (round 1) was a no-op due to caching; resubmitted with --force

**Bug-on-bug:** the first re-eval batch (jobs 7242846–7242854) ran the bug-fixed `evaluate_run.py`, but `scripts_evaluation/evaluate_run.py:486` short-circuits with `if eval_path.exists() and not args.force:` — so the per-query `*_eval.json` files (computed under the OLD pre-fix logic and never deleted) were just reloaded. The "new" `evaluation_summary.json` was rebuilt from those stale per-query files, so MiniMax baseline still read 27.3% etc.

**Fix:** added `--force` to all 9 eval SBATCHes; deleted every `*_eval.json` / `evaluation_summary.json` / `detailed_judge_results.csv` under `evals/bcp/` (originals safe at `evals.backup_pre_force_answer_fix_2026-04-26/`); cancelled the still-pending 7242854 and resubmitted all 9 as **7244953–7244961**.

**Lesson:** when changing `evaluate_run.py` grading logic, always pair it with `--force` OR an explicit cache wipe. The caching means stale per-query files survive any code change to the grading branch.

---

## 2026-04-26 — FRAMES smoke #1 (job 7244214) failed: search_description signature mismatch

**Symptom:** vLLM came up cleanly, BgeM3Searcher loaded the 47M-passage index on cuda:0, but `glm_client.py:94` crashed with `TypeError: BgeM3Searcher.search_description() takes 1 positional argument but 2 were given`. BCP's `BaseSearcher.search_description(self, k: int = 10)` passes `self.k` to format the tool description; parent's `BgeM3Searcher.search_description(self)` ignores k.

**Fix:** override `search_description(self, k=10)` in the BCP adapter (`searcher/searchers/bge_m3_searcher.py`). Parent project untouched.

**Resubmitted:** job **7245695**.

---

## 2026-04-26 — MiniMax baseline eval (7244959) failed transiently; resubmitted as 7246960

vLLM worker subprocess died silently during `determine_available_memory` after Qwen3-32B weights loaded fine (30.59 GiB/shard). Worker error string was empty. Same SBATCH succeeded for GLM/Qwen3.5 sibling jobs in the same batch. No code change; resubmitted unchanged.

---

## 2026-04-26 — BCP scout/explore corrected results

All 9 BCP re-evals (jobs 7244953-7244961, plus resubmit 7246960 for transient MiniMax baseline failure) completed with the bug-fixed eval + `--force` flag. scout_explore.md tables updated.

| Model | Cond | Pre-fix Acc | Post-fix Acc | Δ |
|---|---|---|---|---|
| GLM | baseline | 44.0 | 48.0 | +4.0 |
| GLM | traj_orig | 44.0 | 47.3 | +3.3 |
| GLM | traj_summary | 53.3 | 53.3 | 0 |
| Qwen3.5 | baseline | 42.7 | 45.3 | +2.6 |
| Qwen3.5 | traj_orig | 46.3 | 48.4 | +2.1 |
| Qwen3.5 | traj_summary | 48.3 | 48.3 | 0 |
| MiniMax | baseline | 27.3 | **48.7** | **+21.4** |
| MiniMax | traj_orig | 48.0 | 54.0 | +6.0 |
| MiniMax | traj_summary | 52.0 | 56.0 | +4.0 |

**Headline:** MiniMax baseline correction (+21.4pp) is the most consequential — it had 71% context_limit rate, all of which were previously auto-failed. Pattern-wise, the summary-prepend advantage still holds for all three models. MiniMax now leads on every condition.

---

## 2026-04-26 — FRAMES Stage F1 (smoke) PASS → Stage F2 baselines submitted

**Smoke test (job 7245695):** 10/10 trajectories landed in `runs/frames/BGE-M3/first10/glm-4.7-flash/seed0/`. Status: 8 completed + 2 incomplete. All have tool calls + original_messages. Spot-checked first trajectory (qid 2): BGE-M3 returned real Wikipedia content (Groundhog Day / Punxsutawney Phil snippet matching the query). End-to-end integration validated.

**Stage F2 baselines submitted:**
- GLM:     job **7248509** (`run_frames_test150_glm.SBATCH`)
- Qwen3.5: job **7248510** (`run_frames_test150_qwen3_5.SBATCH`)
- MiniMax: job **7248511** (`run_frames_test150_minimax.SBATCH`)

Same architectural choices as BCP test150 SBATCHes, with these FRAMES-specific deltas:
- `--searcher-type bge_m3` + `--bge-index-path/--bge-texts-path/--bge-device cuda:0/--bge-nprobe 128` (BGE-M3 on GPU per user directive)
- Output dir under `runs/frames/BGE-M3/test150/<model>/seed0/` (BGE-M3 in path to disambiguate retriever from BCP's Qwen3-Embedding-8B)
- `EMBED_RESERVE_GB` bumped from 5 → 10 for Qwen3.5 + MiniMax (BGE-M3 needs ~3 GB on GPU; 10 GB leaves headroom)
- `EMBED_RESERVE_GB` stays at 25 for GLM (h100_tandon has plenty of headroom on 2×H100 80GB)
- Dynamic vLLM port for h200_public co-tenancy (`PORT=$((20000 + SLURM_JOB_ID % 10000))`)

---

## 2026-04-26 — GLM selected_tool_calls run COMPLETED → eval submitted

Job 7244602 finished in 49:30. 150/150 trajectories in `runs/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/selected_tools_seed0/`. Eval submitted as job **7249478** (`sbatch/run_bcp_eval_glm_test150_selected_tools.SBATCH`). Once it completes, append a 4th GLM row to scout_explore.md.

---

## 2026-04-26 — GLM selected_tool_calls eval result

Job 7249478 completed in 5:28.

**GLM selected_tools (n=150):** Acc=46.7%, Recall=29.1%, avg_searches=8.6  
- by status: completed=66/141 (46.8%), context_limit=4/9 (44.4%)

GLM round-2 picture:
| condition | Acc | Recall | searches |
|---|---|---|---|
| baseline | 48.0 | 55.4 | 22.0 |
| traj_orig_ext | 47.3 | 20.3 | 4.3 |
| traj_summary_orig_ext | **53.3** | 52.5 | 12.7 |
| selected_tools | 46.7 | 29.1 | 8.6 |

LLM-summary still wins for GLM. Selected-tool-calls underperforms: less helpful than the LLM summary (-6.6pp) and even slightly below baseline. Hypothesis: 5 raw excerpts ≈ 8.6 fresh searches' worth of context but lack the orienting synthesis the LLM summary provides; recall jumps vs traj_orig (20→29%) because the model still searches modestly, but accuracy doesn't catch up.

scout_explore.md updated with the 4th GLM row.

---

## 2026-04-26 — FRAMES Qwen3.5 baseline (7248510) OOM with --bge-device cuda:0; falling back to CPU for both 2×H200 jobs

**Symptom:** Qwen3.5 vLLM died with `EngineDeadError` after ~12 minutes; only 1/150 trajectory landed before crash. Root cause: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 139.80 GiB of which 11.44 MiB is free. Process 1434939 has 132.95 GiB memory in use.` vLLM consumed ~133 GiB of the 140 GB H200, leaving ~7 GiB. BGE-M3 (~2.3 GiB model) + FAISS shards (3 procs × ~1.7 GiB = ~5 GiB) + fragmentation pushed past the budget.

**Fix:** the plan's documented fallback — drop BGE-M3 to CPU for both 2×H200 SBATCHes (Qwen3.5 + MiniMax). GLM stays on GPU since h100_tandon has plenty of headroom. Specific changes per SBATCH:
- `--bge-device cuda:0` → `cpu`
- `EMBED_RESERVE_GB=10` → `5` (no GPU need for BGE)
- Add `export CUDA_VISIBLE_DEVICES=-1` before client (redundant given --bge-device cpu, but matches BCP convention)
- Bump Qwen3.5 walltime 12h → 24h (CPU encoding adds ~1-2s/query on 16-thread parallel)

**Resubmitted:** Qwen3.5 → job **7254737**, MiniMax → job **7254738**. Cancelled the dead 7248510 (still allocating gh109 with crashed engine) and the pending 7248511.

---

## 2026-04-26 — Symmetric force-final-answer at iter cap (all 3 clients)

**Asymmetry found:** the agent loop's `context_limit` exit calls `_force_final_answer(...)` so the trajectory has a gradable final `output_text`. The iter-cap exit (`max_iterations=100` reached without the model returning a final answer) just returns `"incomplete"` with no final-answer call → eval skips those trajectories entirely (response = "" → grading short-circuits).

**Fix:** patched all 3 clients (`glm_client.py:417`, `qwen35_client.py:416`, `minimax_client.py:416`) to call `_force_final_answer(...)` before the `"incomplete"` return. Symmetric with the `context_limit` branch.

**FRAMES GLM fill-in:** the FRAMES test150 GLM baseline (job 7248509) had 13 trajectories at iter cap. Deleted those JSONs (resume-safe client → only re-runs missing qids) and resubmitted as job **7260922** with the patched client. The 13 should come back with status="incomplete" but a forced final answer the eval can grade.

**Per-baseline auto-eval:** updated cron job (now `b759969b`) to submit `sbatch/run_frames_eval.SBATCH` (parameterized by MODEL+CONDITION env vars) the moment each FRAMES baseline reaches 150 trajectories — no longer waits for all 3. New eval SBATCH at `sbatch/run_frames_eval.SBATCH` uses BCP's `evaluate_run.py` against `data/frames_ground_truth.jsonl`.

---

## 2026-04-27 — FRAMES GLM baseline (post-fill-in) DONE → eval submitted

GLM FRAMES test150 baseline now at 150/150 after fill-in (7260922) re-ran the 13 iter-cap qids with the patched client. New status distribution: **138 completed + 10 context_limit + 2 incomplete** (was 131 + 6 + 13 pre-fill-in). The 13 fill-in qids resolved as 7 → completed, 4 → context_limit, 2 → still incomplete (but now with forced final answers thanks to the iter-cap patch — eval will grade them).

Eval submitted: job **7266999** (`sbatch --export=ALL,MODEL=glm-4.7-flash,CONDITION=seed0 sbatch/run_frames_eval.SBATCH`). When complete, append GLM row to scout_explore.md FRAMES section.

## 2026-04-27 — GLM FRAMES eval result (job 7266999)

**GLM FRAMES test150 baseline:** Acc=44.7%, Recall=n/a (FRAMES has no evidence qrels), avg_searches=27.7
- by status: completed=64/138 (46.4%), context_limit=3/10 (30.0%), incomplete=0/2 (0%)

GLM FRAMES (44.7%) is comparable to GLM BCP (48.0%). Avg search count noticeably higher on FRAMES (27.7 vs 22.0) — multi-hop reasoning queries push the agent to explore more before converging. Iter-cap patch worked: only 2 trajectories hit it this time, both with forced final answers (graded as 0/2 — bad luck on those two queries).

scout_explore.md FRAMES section created with the GLM row; round-2 cells left blank pending Qwen3.5/MiniMax baselines + round-2 runs.

---

## 2026-04-27 — Per-model independent staging (cron `26f63469`)

Old cron `b759969b` deleted; new cron `26f63469` does per-model independent staging instead of waiting for all 3 baselines before submitting any round-2 work. As soon as a model's baseline hits 150 trajectories, its traj_orig + summarize fire (in parallel); when summarize finishes, its traj_summary fires; each round-2 condition gets auto-evaluated when its trajectories complete.

**GLM round-2 submitted now** (GLM baseline already done):
- traj_orig → job **7281082** (`run_frames_test150_glm_traj_orig.SBATCH`)
- summarize → job **7281083** (`run_frames_summarize_glm_test150.SBATCH`)

**Created 9 round-2 SBATCHes** under `sbatch/`:
- `run_frames_test150_{glm,qwen3_5,minimax}_traj_orig.SBATCH`
- `run_frames_summarize_{glm,qwen3_5,minimax}_test150.SBATCH`
- `run_frames_test150_{glm,qwen3_5,minimax}_traj_summary_orig.SBATCH`

GLM uses `--bge-device cuda:0` (h100_tandon has plenty of VRAM headroom). Qwen3.5 + MiniMax use `--bge-device cpu` + `CUDA_VISIBLE_DEVICES=-1` for the client (h200_public's vLLM consumes ~133 GiB, leaves no GPU headroom for BGE-M3 → OOM history).

---

## 2026-04-27 — All 4 pending FRAMES jobs CANCELLED

Jobs **7281082** (GLM FRAMES traj_orig), **7281083** (GLM FRAMES summarize), **7254737** (Qwen3.5 FRAMES baseline), **7254738** (MiniMax FRAMES baseline) all hit `CANCELLED+` state with `0:00:00` elapsed — i.e. cancelled while still pending, never started running. Likely user-initiated `scancel`. Cron will NOT autonomously resubmit (per the "never resubmit without diagnosing" rule); waiting for user direction.

If the cancel was unintentional, resubmit with:
```
sbatch sbatch/run_frames_test150_qwen3_5.SBATCH
sbatch sbatch/run_frames_test150_minimax.SBATCH
sbatch sbatch/run_frames_test150_glm_traj_orig.SBATCH
sbatch sbatch/run_frames_summarize_glm_test150.SBATCH
```

---

## 2026-04-27 — Mass cancellation diagnosis: `CANCELLED by 0` (root)

The 4 cancelled jobs (7254737, 7254738, 7281082, 7281083) all show `CANCELLED by 0` in `sacct -j <id> -o State%30 -X` — UID 0 is **root**. Cluster admin (or root-owned cleanup process) issued mass scancel at 2026-04-27T09:46:59. Likely cause: long-pending Priority queue on shared `h200_public` (the 2 baselines had been pending many hours; admins occasionally clear stale pendings, OR a maintenance reservation conflicted with the slots).

**Resubmitted (with user authorization to auto-resubmit on root cancellations going forward):**
- 7299058: Qwen3.5 baseline (resubmit of 7254737)
- 7299059: MiniMax baseline (resubmit of 7254738)
- 7299060: GLM traj_orig (resubmit of 7281082)
- 7299061: GLM summarize (resubmit of 7281083)

**Cron updated to `0ab38444`** with new rule: auto-resubmit on `CANCELLED by 0`, but still require diagnosis for FAILED/TIMEOUT/NODE_FAIL/`CANCELLED by <other_uid>`.

---

## 2026-04-27 — BCP selected_tools for Qwen3.5 + MiniMax (HIGH PRIORITY)

User dropped 2 new selected_tool_calls JSONL files into `selected_tool_calls/` (150 entries each, already filtered to test150):
- `selected_tool_calls_qwen3.5-122b-a10b_use_original_messages.jsonl`
- `selected_tool_calls_minimax_use_original_messages.jsonl`

Same `excerpt` field format as the GLM file → existing summary loader handles them natively. Created 2 SBATCHes (mirror of `run_bcp_test150_glm_selected_tools.SBATCH`):
- `sbatch/run_bcp_test150_qwen3_5_selected_tools.SBATCH` → submitted as job **7311137**
- `sbatch/run_bcp_test150_minimax_selected_tools.SBATCH` → submitted as job **7311140**

**Cancelled to free h200_public queue priority**: FRAMES Qwen3.5 baseline (7299058, was at 1/150 — no progress lost) and FRAMES MiniMax baseline (7299059, 0/150). They will be resubmitted by the new cron `dfa110ab` after both selected_tools jobs (7311137, 7311140) complete.

GLM FRAMES round-2 jobs (7299060 traj_orig, 7299061 summarize) remain in flight on h100_tandon — not affected.

## 2026-04-27 — GLM FRAMES summarize COMPLETE → submitted traj_summary

GLM FRAMES summarize (job 7299061) finished; `summaries/glm-4.7-flash_frames_test150.jsonl` has 150 lines. Submitted GLM traj_summary as job **7320902** (`run_frames_test150_glm_traj_summary_orig.SBATCH`). GLM FRAMES traj_orig (7299060) currently at 126/150, ~83% done.

## 2026-04-27 — GLM FRAMES traj_orig done at 143/150 → eval submitted

GLM FRAMES traj_orig (job 7299060) COMPLETED with 143/150 trajectories. Status distribution: 124 completed + 17 context_limit + 1 incomplete + 1 broken_tool_call. **7 qids missing entirely** (no output file): 26, 74, 78, 99, 127, 129, 132 — these errored before producing any trajectory file. Could fill in with a small resubmit (resume-safe client skips existing); deferring unless results look biased.

Submitted eval on the 143 as job **7322121**. scout_explore.md will note partial coverage when the row is appended.

## 2026-04-27 — GLM FRAMES traj_orig eval result + traj_summary eval submitted

**GLM FRAMES traj_orig (job 7322121)** completed in 5:42. Result: Acc=46.2% (+1.5pp vs baseline 44.7%), avg_searches=8.5 (down from 27.7 — prepended trajectory makes the model search less). Per status: completed=62/124, context_limit=4/17, incomplete/broken_tool_call=0/1 each. n=143 due to 7 missing qids. scout_explore.md updated.

**GLM FRAMES traj_summary** at 150/150 trajectories. Submitted eval as job **7324985**.

## 2026-04-27 — GLM FRAMES traj_summary eval result — GLM FRAMES sweep COMPLETE

**GLM FRAMES traj_summary (job 7324985)** completed in 4:12. Result: Acc=51.3% (+6.6pp vs baseline 44.7%, +5.1pp vs traj_orig 46.2%), avg_searches=14.9. Per status: completed=76/143 (53.1%), context_limit=1/5, incomplete=0/2.

**GLM FRAMES sweep complete** — all 3 conditions populated:
| condition | Acc | searches |
|---|---|---|
| baseline | 44.7 | 27.7 |
| traj_orig_ext | 46.2 | 8.5 |
| **traj_summary_orig_ext** | **51.3** | **14.9** |

Same pattern as BCP: summary-prepend wins (+6.6pp on FRAMES; +5.3pp on BCP). traj_orig collapses search count without much accuracy gain.

## 2026-04-27 — Random tools ablation submitted

User dropped 3 random ablation files into `selected_tool_calls/`:
- `glm_random_tools_calls.jsonl` (830 lines)
- `qwen3.5-122b-a10b_random_tools_calls.jsonl` (150 lines)
- `minimax_random_tools_calls.jsonl` (150 lines)

Same `excerpt` field schema as the gemini-selected versions — plug-and-play.

**Created 3 run SBATCHes + 4 eval SBATCHes** (mirrored from selected_tools templates):
- `run_bcp_test150_{glm,qwen3_5,minimax}_random_tools.SBATCH`
- `run_bcp_eval_{glm,qwen3_5,minimax}_test150_random_tools.SBATCH` + previously-missing eval SBATCHes for qwen3_5/minimax selected_tools

**GLM random_tools submitted now** as job **7332862** (h100_tandon, parallel to FRAMES selected_tools). Qwen3.5 + MiniMax random_tools queued for cron auto-submission after their selected_tools complete (h200_public is bottleneck — submitting both would just fight for the same slot).

**Cron updated to `b601b394`** with new state machine:
- Auto-eval each (model, condition) when 150 trajectories land
- Auto-submit Qwen3.5/MiniMax random_tools when their selected_tools finish
- Resubmit FRAMES baselines after all BCP work clears

---

## 2026-04-27 — Storage relocation: home → scratch (file-count quota was at 93%)

**Problem:** `/home/afw8937` hit 27,991/30,000 inodes (93%). The bulk of inodes lived in `evals/` (~2,626 per-query JSON files) and an `evals.backup_pre_force_answer_fix_2026-04-26/` directory (~2,000 more files from a safety snapshot earlier in the session). Mid-day, several running BCP jobs failed because the user moved `runs/` and `data/` out of `/home` to `/scratch/afw8937/browsecomp-plus/` to avoid the inode wall:

- 7311137 (Qwen3.5 selected_tools): every query failed with "Connection error" — vLLM-side died likely because the relative path `runs/...` disappeared mid-write
- 7311140 (MiniMax selected_tools): exited 13 in 0:02 during setup, same root cause
- 7332862 (GLM random_tools): "COMPLETED" but only 81/150 trajectories landed before output dir vanished

**Fix applied (2026-04-27):**
1. Deleted `evals.backup_pre_force_answer_fix_2026-04-26/` — eval bug-fix already validated by the corrected scout_explore.md numbers; backup no longer needed (~2,000 inodes recovered)
2. Moved `evals/` → `/scratch/afw8937/browsecomp-plus/evals/` and symlinked
3. Moved `summaries/` → `/scratch/afw8937/browsecomp-plus/summaries/` and symlinked
4. Created symlinks for the user's pre-existing moves: `runs/` → `/scratch/.../runs/`, `data/` → `/scratch/.../data/`
5. Resubmitted the 2 BCP selected_tools jobs (7369110, 7369111) and the GLM random_tools eval on the partial 81 (7369183)

**Where outputs go now:**

| Path in repo | Resolves to | Notes |
|---|---|---|
| `runs/*` | `/scratch/afw8937/browsecomp-plus/runs/*` | Trajectory JSON outputs |
| `evals/*` | `/scratch/afw8937/browsecomp-plus/evals/*` | Per-query + summary eval JSONs |
| `summaries/*` | `/scratch/afw8937/browsecomp-plus/summaries/*` | Trajectory summaries |
| `data/*` | `/scratch/afw8937/browsecomp-plus/data/*` | Ground-truth + input JSONLs |
| Logs (`#SBATCH --output=`) | `/scratch/afw8937/logs/` directly | Already on scratch — no change |
| `selected_tool_calls/*` | `/home/...` (still home) | Read-only inputs, only 6 files — not a quota concern |
| HF cache (`HF_HOME`) | `/scratch/afw8937/.huggingface/` | Already on scratch via singularity bind |
| vLLM container (SIF + overlay) | `/scratch/afw8937/singularity_images/` | Already on scratch |

All SBATCHes use relative paths (`runs/...`, `evals/...`, `summaries/...`) so they transparently resolve through the symlinks — **no SBATCH edits required**.

**Quota after cleanup:** /home 25,172/30,000 inodes (83%). /scratch 1.8 TB / 5 TB (36%) — plenty of headroom.

**Going forward:** any new SBATCHes should use the same relative-path pattern. If the user ever rebuilds the project on a fresh checkout, they'll need to recreate these 4 symlinks (or `mkdir -p` real dirs if they want home-resident).

## 2026-04-28 — GLM BCP random_tools eval result (partial n=81)

Job 7369183 completed; ablation on the 81 trajectories that landed before the dir-move interruption (job 7332862 only got 81/150 of `random_tools_seed0/` written).

**GLM random_tools (n=81):** Acc=49.4%, Recall=36.5%, avg_searches=9.1

**Surprising on partial:** Random tool calls (49.4%) currently *out-performs* Gemini-selected (46.7%) on GLM. Heavy caveat — different N (81 vs 150), so the random subset may be enriched for easier queries (the 81 that completed before the disruption are the queries the agent dispatched first). Need full 150-query random_tools run to confirm. If the gap holds with full data, it suggests Gemini's selection of "useful" tool calls is no better than random for downstream answer accuracy in this regime.

scout_explore.md updated with the partial row.

## 2026-04-28 — GLM random_tools fillin submitted

Resubmitted `run_bcp_test150_glm_random_tools.SBATCH` as job **7370023**. Client is resume-safe — will skip the 81 already-completed qids and process the missing 69. Once all 150 land, re-eval needs to overwrite the partial-81 result; the eval SBATCH already has `--force` so just resubmitting the eval after will do the right thing. The cron will auto-re-eval when 150 trajectories are present (its guard checks `evaluation_summary.json` exists, so we'd need to delete that first to trigger re-eval — manual: `rm evals/bcp/Qwen3-Embedding-8B/test150/glm-4.7-flash/random_tools_seed0/*` after the run completes).

## 2026-04-28 — GLM random_tools 150/150 → full eval submitted

GLM random_tools fillin (job 7370023) brought trajectory count from 81 → 150. Full eval submitted as job **7371029**. Will overwrite the partial-81 row (49.4%) in scout_explore.md with the corrected full-150 numbers.

## 2026-04-28 — FRAMES recall computed post-hoc (article-level, n=39 subsample)

Built `topics-qrels/frames/qrel_evidence.txt` from joining `frames-all-gt.jsonl` `wiki_links` with `frames_wiki_url_to_rowindex.json`. Two issues encountered:

1. **URL match rate: 21%** — only 234/2496 unique GT URLs found in the BGE-M3 corpus URL map. Causes likely: redirects, anchor fragments, non-English variants, or articles outside the indexed Upstash subset. After matching, 172/824 qids (21%) have any evidence; on test150 this is 39/150 qids.
2. **URL → list of row_ids, not single row_id** — each Wikipedia article is split into many passages. Bug fixed in `scripts/compute_frames_recall.py` by expanding the list into multiple qrel lines.

**Article-level recall** (more meaningful than passage-level — agent only needs to retrieve ≥1 passage per article) computed by grouping consecutive row_ids per qid into "articles". The script lives at `scripts/compute_frames_recall.py` — set-intersection only, no LLM judge needed.

**Results — GLM FRAMES, n=39 with evidence:**
| condition | article-recall |
|---|---|
| baseline | 62.8% |
| traj_orig_ext | 16.9% |
| traj_summary_orig_ext | 41.2% |

Pattern matches BCP recall: traj_orig collapses; traj_summary partially recovers; baseline is highest.

## 2026-04-29: Qwen3.5 FRAMES baseline TMA/MoE GEMM crash (#4)
- Job 7431649 (Qwen3.5 FRAMES test150 baseline) exited "COMPLETED 0:0" after 3:02 hours but vLLM crashed mid-run with the recurring `[TensorRT-LLM][ERROR] Failed to initialize cutlass TMA WS grouped gemm` / `torch.AcceleratorError: CUDA error: misaligned address` error on H200 at iter ~1:20:26
- 117/149 trajectories saved; remaining 32 queries returned "3 consecutive API errors; last: Connection error" because vLLM was dead. No JSON written for those qids.
- Resubmitted as 7455790 — qwen35_client idempotent skip logic will only process the 32 missing qids.
- Same kernel crash also hit Qwen3.5 BCP random_tools 3 times before we accepted partial N=130; this is now the 4th time on H200 + Qwen3.5.

## 2026-04-29: MiniMax FRAMES traj_orig N=126 (24 context overflows)
- Job 7468481 (MiniMax FRAMES test150 traj_orig_ext) completed in 20:35 min, but only 126/150 trajectories saved.
- 24 queries hit a 65536-token max-context-length error at the FIRST chat completion call: prepended FRAMES baseline trajectory (~55537 input tokens) + 10000 reserved output tokens exceeded the limit. No JSON was written for those qids since the agent loop never started.
- Same pattern as MiniMax BCP traj_orig (verbose baseline trajectories overflow on prepend). Eval submitted on N=126; results comparable to BCP MiniMax traj_orig where some trajectories were also dropped.

## 2026-04-29: Qwen3.5 FRAMES traj_orig N=131 (19 context overflows)
- Job 7472945 (Qwen3.5 FRAMES test150 traj_orig_ext) completed in 14:11 min, 131/150 trajectories saved.
- 19 queries hit a 131072-token max-context-length error at the FIRST chat completion call: prepended FRAMES baseline trajectory (~121073 input tokens) + 10000 reserved output tokens exceeded the limit.
- Qwen3.5's 131K context is much larger than MiniMax's 65K, but Qwen3.5 baseline trajectories are also much longer (verbose). Net result: similar drop rate (19 vs 24).

## 2026-04-29: Qwen3.5 FRAMES traj_summary TMA crash (#5)
- Job 7484898 (Qwen3.5 FRAMES test150 traj_summary_orig) exited "COMPLETED 0:0" after 59:09 min, but vLLM crashed mid-run with the same TMA/cutlass MoE GEMM kernel error on H200. 88/150 trajectories saved.
- Resubmitting; qwen35_client idempotent skip will only process the 62 missing qids.
- Cumulative count of this exact crash on Qwen3.5 + H200: 5 (3× BCP random_tools, 1× FRAMES baseline, 1× FRAMES traj_summary).

## 2026-04-30: Truncation budget bug — `--max-chars 500000` is window-blind

The `--max-chars 500000` flag in every traj_orig/traj_summary/random_tools SBATCH is a **global per-trajectory char cap**, not derived from `MAX_MODEL_LEN`. At ~4 chars/token, 500K chars ≈ 125K tokens — which is **2× too large for MiniMax's 65K window** and within ~5K tokens of Qwen3.5/GLM's 131K window. This is *the* reason FRAMES traj_orig dropped 24 (MiniMax) and 19 (Qwen3.5) trajectories: per-block truncation (`--reasoning-max-chars 3000`, `--tool-output-max-chars 5000`) was active and capping individual blocks, but the global cap of 500K never kicked in below the model window, so 25–30 tool calls × 8K chars/block = 200–240K chars passed the global cap unchanged.

Recommended values, derived from `MAX_MODEL_LEN` minus a 15–25K-token reservation for query + new searches + reserved output:

| Model | `MAX_MODEL_LEN` | Reserve | Available for prepend (tokens) | Recommended `--max-chars` |
|---|---|---|---|---|
| MiniMax-M2.5 | 65,536 | ~15,000 | ~50,000 | **200,000** |
| GLM-4.7-Flash | 131,072 | ~25,000 | ~106,000 | **400,000** |
| Qwen3.5-122B-A10B | 131,072 | ~25,000 | ~106,000 | **400,000** |

**Why MiniMax is at 65K and not 245760 (its real max):** advertised context is 245760 tokens. We capped at 65K in `vllm serve` to leave KV-cache headroom on 2×H200 (model weights ~215 GB; ~67 GB left for KV across all concurrent seqs). Going to 130K+ would require dropping `--max-num-seqs` from 16 to ~8, hurting throughput. For BCP that tradeoff is reasonable; for FRAMES traj_orig it bites because Wikipedia tool outputs are 5–10× larger than BCP's docids.

**Action items deferred until BCP focus is done:**
- Patch all FRAMES SBATCHes to use window-aware `--max-chars` per the table above
- Re-run MiniMax + Qwen3.5 FRAMES traj_orig with the corrected cap (would recover the 24 + 19 dropped trajectories)
- Optionally bump MiniMax `MAX_MODEL_LEN` 65K→100K with `--max-num-seqs 8` if memory permits

## 2026-04-30: Seed semantics clarification (no agent seed, no temperature override)

The BCP/FRAMES clients (`glm_client.py`, `minimax_client.py`, `qwen35_client.py`) **do not pass `temperature` or `seed` to vLLM** — only `gemini_client.py` and `openai_client.py` do. Consequences:

- **Temperature**: vLLM's OpenAI-server default applies (likely 1.0 or whatever the model's chat-template config sets). Generations are *not* greedy.
- **Seed**: not specified, vLLM picks one randomly per request. Two runs of the same agent on the same input produce **different trajectories**.
- The `seed0` label in run directory names (e.g. `random_tools_seed0/`) was always a labeling convention — there was no actual seed=0 being passed anywhere.
- Renamed existing `random_tools_seed0/` → `random_tools_seed42/` (BCP test150, all 3 models, in `runs/` and `evals/`) to match the **selection seed** used by `random_select_tool_calls.py`. The "seed42" now refers unambiguously to the random-subset selection seed, which IS reproducible.

**Implication for Task 1 (best-of-4 random selection seeds):** what we vary is *which 5 tool calls get prepended*, not agent-level RNG. The agent's downstream sampling on top of those 5 prepended calls is non-deterministic and contributes its own variance — so two runs with the same selection seed would also produce different accuracy numbers. Best-of-4 is therefore confounded with run-to-run variance, but since the variance is shared across all 5 prepend strategies, intra-model comparison is still fair.

## 2026-04-30: Qwen3.5 selected_tools N=148 (cancelled, not crashed)

Distinct from the TMA crashes: **Qwen3.5 BCP selected_tools** finished N=148/150 — the run got stuck on the last 2 queries (`completed=148/150`) for ~1.5 hours with vLLM still alive but the agent loop not progressing. Manually `scancel`'d and ran eval on N=148.

Pattern observed 2-3× now with Qwen3.5: a long generation that never hits a stop token. Likely fix would be a per-query timeout in `qwen35_client.py`; not patched yet.

## 2026-04-30: Where FRAMES infrastructure actually lives

| Component | Location |
|---|---|
| Adapter glue (BCP-side `BgeM3Searcher`) | `searcher/searchers/bge_m3_searcher.py` (in `afw-scout-first10` branch) |
| Actual retriever implementation | `~/projects/efficient-search-agents/frames/retrieval/bge_m3_backend.py` (parent project, **not in this repo**) |
| FRAMES query subsets | `topics-qrels/frames/queries_{first10,test150}.tsv` (in branch) |
| FAISS index + Wikipedia texts | `/scratch/afw8937/efficient-search-agents/frames/index/` (~47 GB, scratch only) |
| FRAMES ground truth | `/scratch/afw8937/efficient-search-agents/frames/data/frames-all-gt.jsonl` |
| URL→row_id map | `/scratch/afw8937/efficient-search-agents/frames/index/frames_wiki_url_to_rowindex.json` |
| FRAMES SBATCH templates | `sbatch/run_frames_test150_*.SBATCH` (in branch) |
| Article-level recall computation | `scripts/compute_frames_recall.py` (in branch) |
| Trajectory output | `runs/frames/...` (symlink → `/scratch/afw8937/browsecomp-plus/runs/frames/`) |

**Implication:** if a coworker checks out `afw-scout-first10` *standalone* without the parent `efficient-search-agents` directory, the FRAMES retriever import will fail at the `sys.path.insert(0, parents[4]/"frames"/"retrieval")` line in `bge_m3_searcher.py`. Either bundle `bge_m3_backend.py` into the BCP repo or document the parent-project dependency in the README.

## 2026-04-30: Best-of-N over partial-N seeds

Two options for computing best-of-4 across {seed42, 43, 44, 45} when individual seeds may be partial-N (e.g., Qwen3.5 random_tools_seed42 = N=130):

1. **Force-resubmit until all 150 land** per seed. Probably needs 3+ retries on Qwen3.5 due to TMA crash. Risky and expensive.
2. **Compute best-of-4 only on the qid intersection** that succeeded across all 4 seeds. Cleaner, smaller N, but apples-to-apples comparison.

Choosing **option 2** for write-up purity: "best-of-4 over the N qids where all 4 random selection seeds completed." The aggregation script (`scripts/compute_best_of_n.py`, TBD) should output:
- N_intersection (qids in all 4 evals)
- pass@4 accuracy on intersection
- per-seed accuracy on intersection (for honest comparison)

## 2026-05-01: scratch reorganization — temp-upload → browsecomp-plus

Renamed `/scratch/afw8937/temp-upload/` → `/scratch/afw8937/browsecomp-plus/` to give the BCP/scout-explore data tree a project-descriptive name instead of "temp-upload" (which was historically meaningful as the rsync staging path but became the canonical home for runs/evals/summaries/data).

Layout under the new root:
- `/scratch/afw8937/browsecomp-plus/runs/`     ← agent trajectories  (was `/scratch/afw8937/temp-upload/runs/`)
- `/scratch/afw8937/browsecomp-plus/evals/`    ← eval results        (was `/scratch/afw8937/temp-upload/evals/`)
- `/scratch/afw8937/browsecomp-plus/summaries/`← trajectory summaries (was `/scratch/afw8937/temp-upload/summaries/`)
- `/scratch/afw8937/browsecomp-plus/data/`     ← BCP ground truth     (was `/scratch/afw8937/temp-upload/data/`)
- `/scratch/afw8937/browsecomp-plus/logs/`     ← SLURM job logs       (NEW — formerly mixed into `/scratch/afw8937/logs/`)

Updates applied:
- 4 in-repo symlinks (`runs/`, `evals/`, `summaries/`, `data/`) point to the new path
- New `logs/` symlink in repo root for convenience
- 56 SBATCH templates: `#SBATCH --output=/scratch/afw8937/browsecomp-plus/logs/...`
- Historical logs at `/scratch/afw8937/logs/` left in place (shared with other projects); only future BCP/FRAMES SLURM logs land under the new path

The 10 currently-queued random_tools jobs (job IDs 7741696–7741705) were submitted with the OLD `--output=/scratch/afw8937/logs/...` baked in, so their logs land at the old path. All future submissions land at the new path.

## 2026-05-01: 3 eval SBATCHes refactored for SEED env var

`run_bcp_eval_{glm,minimax,qwen3_5}_test150_random_tools.SBATCH` previously hardcoded `random_tools_seed0` in `INPUT_DIR`. Refactored to take `SEED` env var, defaulting to 42:

```
INPUT_DIR="runs/bcp/Qwen3-Embedding-8B/test150/${MODEL_NAME}/random_tools_seed${SEED}"
```

Submission for any seed:
```
sbatch --export=ALL,SEED=43 sbatch/run_bcp_eval_glm_test150_random_tools.SBATCH
```

Eval output dir mirrors the run dir (eval pipeline writes to `evals/bcp/.../random_tools_seed${SEED}/`).
