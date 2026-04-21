# Auto-Pipeline for `submit_missing.py` Runs — Design

**Date:** 2026-04-20
**Author:** hc3337 (via Claude Code brainstorming)
**Status:** Draft — awaiting user review

## Goal

Automate the submit → monitor → resubmit-missing → detect-empty → eval loop for SLURM array jobs driven by `submit_missing.py`. A single command kicks off the pipeline; it runs unattended until all configured runs are complete and evaluated, or halts with a diagnostic report when it detects a stuck run.

## Scope

**In scope:**
- Pre-flight validation of generated SBATCH content before any job hits the queue
- Initial submission of missing shards only (using `shard_monitor` as ground truth, not the hardcoded `MISSING*` dict)
- Periodic monitoring until all targets complete
- Empty-run detection (via existing `src_utils/filter_empty_runs.py`) + resubmit
- Stuck-run detection (same missing-query-ID set across N cycles) → halt with diagnostic
- Eval SBATCH generation + submission + result parsing
- Markdown summary (`pipeline_summary.md`)

**Out of scope:**
- Rewriting `submit_missing.py` — it remains the source of truth for which runs are in scope
- Rewriting the hand-curated `eval.SBATCH` — pipeline generates a separate `eval_auto.SBATCH`
- Auto-retrying transient errors with code fixes (we diagnose and stop, not patch)
- Email/slack notification (Markdown file only)

## Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| Q1 | Entry point | Single `python auto_pipeline.py --submit` command; reads `MISSING*` dicts from `submit_missing.py` as the scope of targets |
| Q2 | Where the loop runs | Background process on login node via `nohup` |
| Q3 | Pre-flight | Lightweight: `sbatch --test-only` + path existence checks (index, ground truth, template files) |
| Q4 | Empty-run criteria | Use `src_utils/filter_empty_runs.py` as-is (no status/heuristic additions) |
| Q5 | Polling interval + auto-debug | Configurable interval via `--interval-seconds` (default 7200 = 2h); diagnose stuck runs and exit with structured `pipeline_stopped.md` |
| Q6 | Eval + summary | Generate fresh `eval_auto.SBATCH` (never mutate `eval.SBATCH`); write `pipeline_summary.md` via `src_utils/parse_eval_out.py` |

## Architecture

One new script at project root: `auto_pipeline.py`. One thin wrapper: `auto_pipeline.sh` for `nohup`-friendly invocation. Both live alongside `submit_missing.py` and `monitor_and_eval.sh`.

The pipeline imports from existing modules — no logic duplication:

- `submit_missing` → reuses `MISSING`, `MISSING_FIRST50`, `MISSING_FRAMES_FIRST50`, `MISSING_MUSIQUE_FIRST50`, `MISSING_TEST150`, `MISSING_TRAIN680`, `parse_run_name()`, `patch_sbatch()`, and the four `TEMPLATE_PATH_*` constants
- `shard_monitor` → reused as ground-truth completeness oracle (per target)
- `src_utils/filter_empty_runs` → reused in delete mode
- `src_utils/parse_eval_out` → reused for summary generation

### Flow

```
auto_pipeline.py --submit
  │
  ├─ [phase: preflight]
  │    ├─ Collect targets from submit_missing.MISSING_* (uncommented entries only)
  │    ├─ For each target:
  │    │    ├─ Render patched SBATCH content (via patch_sbatch)
  │    │    ├─ Validate with `sbatch --test-only` on a temp file
  │    │    └─ Check referenced paths exist (index, ground truth, template)
  │    └─ Any failure → write preflight_failed.md, exit 1
  │
  ├─ [phase: submitting]
  │    └─ For each target:
  │         ├─ actual_missing = shard_monitor(target)
  │         ├─ If empty: mark target complete, keep in monitor set for eval
  │         └─ Else: sbatch patched file; record job_id in state
  │
  ├─ [phase: monitoring] (loop)
  │    Every INTERVAL seconds:
  │    ├─ Poll SLURM: any of our recorded job_ids still in queue/running?
  │    │    └─ Yes → sleep, continue
  │    ├─ All jobs finished for this cycle:
  │    │    ├─ For each target: shard_monitor → set of missing query IDs
  │    │    ├─ Compare to last_missing_qids:
  │    │    │    ├─ Shrunk (progress) → reset stuck counter
  │    │    │    └─ Unchanged → stuck_cycles += 1
  │    │    ├─ For complete targets: run filter_empty_runs (delete mode)
  │    │    │    └─ Re-check shard_monitor; if new gaps, resubmit those shards
  │    │    └─ For incomplete + not-stuck targets: resubmit missing shards
  │    ├─ If any target stuck_cycles >= STUCK_THRESHOLD (default 3):
  │    │    └─ Diagnose (scan latest sbatch_outputs/*.out + slurm-*.out for that target),
  │    │       write pipeline_stopped.md, exit 2
  │    └─ If all targets complete and clean → exit loop
  │
  ├─ [phase: eval]
  │    ├─ Build eval_auto.SBATCH with singularity-exec lines for each target run dir
  │    │   (copy header from eval.SBATCH, rewrite body)
  │    ├─ sbatch eval_auto.SBATCH
  │    └─ Poll until eval job finishes (same SLURM polling)
  │    └─ On non-zero exit: write eval_failed.md with slurm-*.out tail, exit 3
  │
  └─ [phase: done]
       └─ Parse eval outputs via parse_eval_out.py → pipeline_summary.md
```

## Components

### `auto_pipeline.py`

Single-file script. Key functions (pure helpers testable in isolation):

| Function | Purpose |
|----------|---------|
| `collect_targets() -> list[Target]` | Read `MISSING*` dicts from `submit_missing`, return flat list with `(run_name, shards, template, split, dataset)` |
| `preflight(targets) -> list[PreflightError]` | Render each SBATCH, run `sbatch --test-only`, check paths |
| `compute_actual_missing(target) -> list[int]` | Call `shard_monitor` logic → incomplete shard indices |
| `submit_target(target, shards) -> int` | Render SBATCH → `sbatch` → return job_id |
| `poll_jobs(job_ids) -> dict[int, str]` | `squeue -j ...` → `{job_id: state}` |
| `detect_stuck_targets(state) -> list[Target]` | Compare missing-qid set to previous cycle across all targets |
| `diagnose_slurm_out(target, state) -> str` | Grep recent `sbatch_outputs/*.out` + `slurm-*.out` for the target's patterns (OOM, CUDA, missing file, port conflict, traceback) → text block |
| `build_eval_sbatch(targets) -> str` | Copy `eval.SBATCH` header, rewrite body with one `singularity exec` per target |
| `write_summary(targets, state) -> None` | Call `parse_eval_out.py`, format Markdown |
| `save_state(state) / load_state() -> PipelineState` | JSON serialization |

### `auto_pipeline.sh`

```bash
#!/bin/bash
# Usage: nohup bash auto_pipeline.sh > auto_pipeline.log 2>&1 &
cd /scratch/hc3337/projects/BrowseComp-Plus
python auto_pipeline.py --submit "$@"
```

### CLI

```
auto_pipeline.py
  --submit                    Actually submit jobs. Default is dry-run (prints actions, captures sbatch commands without executing).
                              Matches submit_missing.py's semantics.
  --interval-seconds N        Polling interval (default: 7200 = 2h)
  --stuck-threshold N         Cycles of no progress before declaring stuck (default: 3)
  --skip-preflight            Skip pre-flight checks (emergency use only)
  --skip-eval                 Stop after all runs complete; don't submit eval
  --resume                    Load existing auto_pipeline_state.json and continue from recorded phase
```

## State & Persistence

**`auto_pipeline_state.json`** at project root. Overwritten every cycle:

```json
{
  "pid": 12345,
  "started_at": "2026-04-20T14:30:00",
  "last_check_at": "2026-04-20T16:30:00",
  "cycle_count": 1,
  "phase": "monitoring",
  "interval_seconds": 7200,
  "stuck_threshold": 3,
  "targets": [
    {
      "run_name": "gpt-oss-120b_traj_budget_orig_ext_qwen3.5-4b_seed0",
      "dataset": "bcp",
      "split": "test150",
      "shards_total": 3,
      "shards_done": 1,
      "missing_qids": ["q_42", "q_87"],
      "last_missing_qids": ["q_42", "q_87", "q_99"],
      "stuck_cycles": 0,
      "submitted_job_ids": [6751645, 6751680],
      "status": "submitting"
    }
  ]
}
```

`--resume` reads this file and picks up at the recorded phase, skipping preflight if it already passed.

## Error Handling

| Class | Behavior |
|-------|----------|
| Preflight failure (bad path, `sbatch --test-only` fails) | Write `preflight_failed.md`, exit 1, no jobs submitted |
| `sbatch` fails mid-loop | Retry 3× with 30s backoff, then mark target stuck |
| `shard_monitor` / `filter_empty_runs` raises | Log exception, continue with next target (don't kill whole pipeline over one target) |
| Stuck target (missing-qid set unchanged `STUCK_THRESHOLD` cycles) | Write `pipeline_stopped.md`, exit 2 (all targets halted, not just the stuck one) |
| Eval job exits non-zero | Write `eval_failed.md` with `slurm-*.out` tail, exit 3, no summary |
| SIGTERM / Ctrl-C | Save state, log "interrupted at cycle N", exit 130 |

## Output Files

| File | When written | Contents |
|------|--------------|----------|
| `auto_pipeline.log` | Throughout | Timestamped log via Python logging (stdout redirected by `nohup`) |
| `auto_pipeline_state.json` | Every cycle | State snapshot for `--resume` and human inspection |
| `preflight_failed.md` | Preflight phase | Per-target error: template path, reason (bad path / sbatch stderr), fix suggestion |
| `pipeline_stopped.md` | On stuck detection | Stuck target name, missing shards/qids, grepped error excerpts from `slurm-*.out`, fix suggestions |
| `eval_auto.SBATCH` | Eval phase | Generated; copies `eval.SBATCH` header, one `singularity exec` line per target run dir |
| `eval_failed.md` | Eval non-zero exit | Target eval was running, SLURM output tail |
| `pipeline_summary.md` | Done phase | Per-target accuracy / recall / search counts from `parse_eval_out.py`, plus total wallclock, cycle count, resubmit count |

## Testing Plan

- **Unit tests** (`test_auto_pipeline.py`): test pure helpers with synthetic inputs:
  - `diagnose_slurm_out` against hand-crafted `slurm-*.out` fixtures
  - `detect_stuck_targets` with stacked state snapshots
  - `build_eval_sbatch` against known target lists
  - `collect_targets` against an importable `submit_missing` with a known `MISSING` dict
- **Integration smoke test** (default dry-run, i.e. run without `--submit`): single fake target `MISSING = {"fake_seed0": [0]}` in a temp project dir; all `sbatch` calls captured instead of executed; assert expected command sequence and output files.
- **Manual end-to-end**: uncomment one small target in `MISSING_TEST150` (3 shards), run `nohup bash auto_pipeline.sh --interval-seconds 300 &`, verify preflight passes → submission → completion → eval → summary.

## Open Questions / Future Work

- None for v1. After usage, likely candidates: email notification on `pipeline_stopped.md` / `pipeline_summary.md`; web dashboard reading `auto_pipeline_state.json`.
