#!/usr/bin/env python3
"""
auto_pipeline.py — end-to-end automation over submit_missing.py targets.

See docs/superpowers/specs/2026-04-20-auto-pipeline-design.md for design.

Usage:
    python auto_pipeline.py          # dry-run (prints actions, no sbatch)
    python auto_pipeline.py --submit # actually submit
    nohup bash auto_pipeline.sh --submit > auto_pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
STATE_PATH = PROJECT_ROOT / "auto_pipeline_state.json"
LOG_PATH = PROJECT_ROOT / "auto_pipeline.log"

log = logging.getLogger("auto_pipeline")


@dataclass
class Target:
    run_name: str
    dataset: str                # bcp, frames, musique
    split: str                  # full, first50, test150, train680
    template_path: str          # path to the template SBATCH
    # Ground-truth expected shards for this target (may be None for first50 single-shard)
    declared_shards: list[int] | None
    # Parsed fields:
    model: str
    mode: str
    seed: int
    traj_model: str | None


@dataclass
class TargetState:
    target: Target
    missing_qids: list[str] = field(default_factory=list)
    last_missing_qids: list[str] = field(default_factory=list)
    stuck_cycles: int = 0
    submitted_job_ids: list[int] = field(default_factory=list)
    status: str = "pending"     # pending | submitting | running | complete | stuck


@dataclass
class PreflightError:
    target: Target
    reason: str


@dataclass
class PipelineState:
    pid: int = 0
    started_at: str = ""
    last_check_at: str = ""
    cycle_count: int = 0
    phase: str = "init"         # init | preflight | submitting | monitoring | stuck | eval | done
    interval_seconds: int = 7200
    stuck_threshold: int = 3
    targets: list[TargetState] = field(default_factory=list)


def _setup_logging(log_path: Path = LOG_PATH) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def collect_targets() -> list[Target]:
    """Flatten submit_missing.MISSING* dicts into a list of Target objects."""
    import submit_missing as sm

    groups = [
        (sm.MISSING,                  sm.TEMPLATE_PATH,           "full",     "bcp"),
        (sm.MISSING_FIRST50,          sm.TEMPLATE_PATH_FIRST50,   "first50",  "bcp"),
        (sm.MISSING_FRAMES_FIRST50,   sm.TEMPLATE_PATH_FIRST50,   "first50",  "frames"),
        (sm.MISSING_MUSIQUE_FIRST50,  sm.TEMPLATE_PATH_FIRST50,   "first50",  "musique"),
        (sm.MISSING_TEST150,          sm.TEMPLATE_PATH_TEST150,   "test150",  "bcp"),
        (sm.MISSING_TRAIN680,         sm.TEMPLATE_PATH_TRAIN680,  "train680", "bcp"),
    ]
    targets: list[Target] = []
    for missing_dict, template, split, dataset in groups:
        for run_name, shards in missing_dict.items():
            model, mode, seed, traj_model = sm.parse_run_name(run_name)
            declared = shards if isinstance(shards, list) else None
            targets.append(Target(
                run_name=run_name,
                dataset=dataset,
                split=split,
                template_path=template,
                declared_shards=declared,
                model=model,
                mode=mode,
                seed=seed,
                traj_model=traj_model,
            ))
    return targets


def compute_actual_missing(
    target: Target,
    retriever: str = "Qwen3-Embedding-8B",
    agent_model: str | None = None,
) -> tuple[list[int], list[str]]:
    """Return (missing_shard_indices, missing_query_ids) by scanning target's run dir."""
    import shard_monitor as sm

    agent_model = agent_model or target.model
    shard_dir = sm.find_shard_dir(target.dataset)
    if shard_dir is None:
        return ([], [])
    shards = sm.load_shard_query_ids(shard_dir)
    split_ids = sm.load_split_query_ids(target.dataset, target.split)
    if split_ids is not None:
        shards = sm.filter_shards_by_split(shards, split_ids)

    run_dir = os.path.join(
        "runs", target.dataset, retriever, target.split, agent_model, target.run_name,
    )
    completed = sm.scan_completed_ids(run_dir)
    gaps = sm.compute_shard_gaps(shards, completed)

    missing_shards = sorted(
        int(name.split("_")[-1]) for name in gaps
        if name.split("_")[-1].isdigit()
    )
    missing_qids = sorted({q for ids in gaps.values() for q in ids})
    return (missing_shards, missing_qids)


def preflight(targets: list[Target], run_sbatch_check: bool = True) -> list[PreflightError]:
    """Validate SBATCH templates and confirm `sbatch --test-only` would accept them.

    Returns a list of PreflightError (empty = all clear).
    """
    import tempfile
    import submit_missing as sm

    errors: list[PreflightError] = []
    for t in targets:
        if not os.path.isfile(t.template_path):
            errors.append(PreflightError(t, f"template not found: {t.template_path}"))
            continue
        try:
            with open(t.template_path) as f:
                template = f.read()
            patched = sm.patch_sbatch(
                template, t.run_name, t.model, t.mode, t.seed,
                shards=t.declared_shards if t.declared_shards else [0],
                dataset=t.dataset, split=t.split, traj_model=t.traj_model,
            )
        except Exception as e:
            errors.append(PreflightError(t, f"patch_sbatch failed: {e}"))
            continue

        if not run_sbatch_check:
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".SBATCH", delete=False) as tmp:
            tmp.write(patched)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["sbatch", "--test-only", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                errors.append(PreflightError(
                    t, f"sbatch --test-only failed: {result.stderr.strip()}"
                ))
        except Exception as e:
            errors.append(PreflightError(t, f"sbatch --test-only raised: {e}"))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return errors


def submit_target(target: Target, shards: list[int], submit: bool = False) -> int | None:
    """Render target's SBATCH for the given shards and submit. Returns job_id or None."""
    import submit_missing as sm
    import tempfile

    with open(target.template_path) as f:
        template = f.read()
    content = sm.patch_sbatch(
        template, target.run_name, target.model, target.mode, target.seed,
        shards=shards, dataset=target.dataset, split=target.split,
        traj_model=target.traj_model,
    )
    if not submit:
        log.info("[dry-run] would submit %s shards=%s", target.run_name, shards)
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".SBATCH", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    if result.returncode != 0:
        log.error("sbatch failed for %s: %s", target.run_name, result.stderr.strip())
        return None
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        log.error("could not parse job_id from: %s", result.stdout.strip())
        return None
    jid = int(m.group(1))
    log.info("submitted %s shards=%s job_id=%d", target.run_name, shards, jid)
    return jid


def poll_jobs(job_ids: list[int]) -> dict[int, str]:
    """Return {job_id: state}. Jobs absent from squeue are reported as 'DONE'."""
    if not job_ids:
        return {}
    jobs_arg = ",".join(str(j) for j in job_ids)
    result = subprocess.run(
        ["squeue", "-j", jobs_arg, "-h", "-o", "%i %T"],
        capture_output=True, text=True,
    )
    states: dict[int, str] = {jid: "DONE" for jid in job_ids}
    if result.returncode != 0:
        return states
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        # Array jobs appear as "12345_0"; collapse to parent job id.
        head = parts[0].split("_")[0]
        if not head.isdigit():
            continue
        jid = int(head)
        if jid in states:
            # Any still-scheduled array element overrides a stale 'DONE'.
            if states[jid] == "DONE":
                states[jid] = parts[1]
    return states


def detect_stuck_targets(
    states: list[TargetState], threshold: int,
) -> list[TargetState]:
    """Mutate stuck_cycles based on missing-qid comparison; return those at/above threshold."""
    stuck: list[TargetState] = []
    for s in states:
        cur = set(s.missing_qids)
        prev = set(s.last_missing_qids)
        if not cur:
            s.stuck_cycles = 0
            continue
        if cur == prev:
            s.stuck_cycles += 1
        else:
            s.stuck_cycles = 0
        if s.stuck_cycles >= threshold:
            stuck.append(s)
    return stuck


def save_state(state: PipelineState, path: Path | None = None) -> None:
    """Serialize PipelineState to JSON."""
    p = path or STATE_PATH
    p.write_text(json.dumps(asdict(state), indent=2))


def load_state(path: Path | None = None) -> PipelineState:
    """Load PipelineState from JSON, reconstructing nested dataclasses."""
    p = path or STATE_PATH
    data = json.loads(p.read_text())
    targets: list[TargetState] = []
    for ts in data.get("targets", []):
        t = Target(**ts["target"])
        ts_copy = dict(ts)
        ts_copy["target"] = t
        targets.append(TargetState(**ts_copy))
    data["targets"] = targets
    return PipelineState(**data)


ERROR_PATTERNS = [
    re.compile(r"CUDA out of memory", re.I),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.I),
    re.compile(r"FileNotFoundError", re.I),
    re.compile(r"(Address|port) already in use", re.I),
    re.compile(r"Authentication failed", re.I),
    re.compile(r"TimeoutError", re.I),
    re.compile(r"Killed by signal|DUE TO TIME LIMIT|OUT_OF_MEMORY", re.I),
    re.compile(r"Traceback \(most recent call last\)"),
]


def diagnose_slurm_out(
    target: Target,
    slurm_glob: str = "slurm-*.out",
    sbatch_glob: str = "sbatch_outputs/*.out",
    max_files: int = 5,
    context_lines: int = 5,
) -> str:
    """Scan recent SLURM/sbatch out files for common error patterns; return Markdown."""
    import glob as gb

    candidates = sorted(gb.glob(slurm_glob), key=os.path.getmtime, reverse=True)[:max_files]
    candidates += sorted(gb.glob(sbatch_glob), key=os.path.getmtime, reverse=True)[:max_files]
    out: list[str] = []
    for path in candidates:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        if target.run_name not in text and not any(p.search(text) for p in ERROR_PATTERNS):
            continue
        lines = text.splitlines()
        matches: list[str] = []
        for i, line in enumerate(lines):
            if any(p.search(line) for p in ERROR_PATTERNS):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "\n".join(lines[start:end])
                matches.append(f"```\n{snippet}\n```")
        if matches:
            out.append(f"### `{path}`\n\n" + "\n\n".join(matches[:3]))
    return "\n\n".join(out) if out else "_No matching error patterns found._"


def write_pipeline_stopped(
    stuck_states: list[TargetState],
    all_states: list[TargetState],
    path: Path | None = None,
) -> Path:
    path = path or (PROJECT_ROOT / "pipeline_stopped.md")
    lines = ["# Pipeline Stopped — Stuck Target(s) Detected", ""]
    for s in stuck_states:
        missing_preview = ", ".join(s.missing_qids[:30])
        if len(s.missing_qids) > 30:
            missing_preview += "..."
        lines.extend([
            f"## Stuck: {s.target.run_name}",
            f"- Dataset: {s.target.dataset} | Split: {s.target.split} | Model: {s.target.model}",
            f"- Stuck cycles: {s.stuck_cycles}",
            f"- Missing query IDs ({len(s.missing_qids)}): {missing_preview}",
            "",
            "### Diagnostic log excerpts",
            diagnose_slurm_out(s.target),
            "",
        ])
    lines.extend(["## All targets summary", ""])
    for s in all_states:
        flag = " (stuck)" if s in stuck_states else ""
        lines.append(
            f"- `{s.target.run_name}`: status={s.status}, "
            f"missing={len(s.missing_qids)}{flag}"
        )
    path.write_text("\n".join(lines))
    return path


def write_preflight_failed(errors: list[PreflightError], path: Path | None = None) -> Path:
    """Write a human-readable report of preflight failures and return the path."""
    path = path or (PROJECT_ROOT / "preflight_failed.md")
    lines = ["# Preflight Failed", ""]
    for e in errors:
        lines.extend([
            f"## {e.target.run_name}",
            f"- Dataset: {e.target.dataset} | Split: {e.target.split}",
            f"- Template: `{e.target.template_path}`",
            f"- Reason: {e.reason}",
            "",
        ])
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Actually submit (default: dry-run)")
    parser.add_argument("--interval-seconds", type=int, default=7200)
    parser.add_argument("--stuck-threshold", type=int, default=3)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _setup_logging()
    log.info("auto_pipeline.py starting (submit=%s, interval=%ds)",
             args.submit, args.interval_seconds)
    # Orchestrator wired up in later task.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
