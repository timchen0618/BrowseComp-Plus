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
    interval_seconds: int = 14400
    stuck_threshold: int = 6
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


def _resolve_run_subdir(target: Target) -> str:
    """Reproduce the SBATCH template's ``output_dir`` subdir (relative to OUT_BASE).

    Needed because ``run_name`` in ``submit_missing.py`` is a logical identifier
    that does not always match the on-disk directory name. For example,
    ``qwen3.5-4b-sft_budget_seed0`` lives on disk as ``budget5_seed0``
    (the SBATCH injects ``SEARCH_BUDGET=5`` into the dir name).
    """
    mode = target.mode
    seed = target.seed
    traj = target.traj_model

    # Read SEARCH_BUDGET / PLAN_MODEL defaults from the template when present.
    search_budget = "5"
    plan_model = "gemini_2.5_pro"
    try:
        with open(target.template_path) as f:
            tpl = f.read()
        m = re.search(r"^SEARCH_BUDGET=(\S+)", tpl, re.MULTILINE)
        if m:
            search_budget = m.group(1).strip('"')
        m = re.search(r"^PLAN_MODEL=(\S+)", tpl, re.MULTILINE)
        if m:
            plan_model = m.group(1).strip('"')
    except OSError:
        pass

    # Mirror the ``case "$inner_mode"`` block in the SBATCH templates.
    if mode == "org":
        return f"seed{seed}"
    if mode == "budget":
        return f"budget{search_budget}_seed{seed}"
    if mode in {
        "traj_ext",
        "traj_orig_ext",
        "traj_budget_orig_ext",
        "traj_summary_ext",
        "traj_summary_orig_ext",
        "traj_summary_ext_selected_tools",
        "traj_summary_orig_ext_selected_tools",
    }:
        return f"{mode}_{traj}_seed{seed}"
    # planning_v*_start_ext[_reinject_every_5] — plan_model gets injected
    m = re.match(r"^(planning_v[^_]+)_start_ext(_reinject_every_\d+)?$", mode)
    if m:
        inner = m.group(1)
        reinject = m.group(2) or ""
        return f"{inner}_start_ext_{plan_model}{reinject}_seed{seed}"
    # default
    return f"{mode}_seed{seed}"


def _find_split_shard_dir(dataset: str, split: str) -> str | None:
    """Pick the shard directory matching this dataset+split (prefer split-specific)."""
    base = os.path.join("topics-qrels", dataset)
    if not os.path.isdir(base):
        return None
    entries = sorted(os.listdir(base))
    # Split-specific: e.g. bcp_test150_3_shards, bcp_train680_8_shards
    split_prefix = f"{dataset}_{split}_"
    for e in entries:
        if e.startswith(split_prefix) and e.endswith("_shards"):
            return os.path.join(base, e)
    # Default (full split): a *_shards dir without any split tag in the name
    known_split_tokens = ("first50", "first100", "test150", "train680")
    for e in entries:
        if e.endswith("_shards") and not any(tok in e for tok in known_split_tokens):
            return os.path.join(base, e)
    return None


def compute_actual_missing(
    target: Target,
    retriever: str = "Qwen3-Embedding-8B",
    agent_model: str | None = None,
) -> tuple[list[int], list[str]]:
    """Return (missing_shard_indices, missing_query_ids) by scanning target's run dir."""
    import shard_monitor as sm

    agent_model = agent_model or target.model
    shard_dir = _find_split_shard_dir(target.dataset, target.split)
    if shard_dir is None:
        shard_dir = sm.find_shard_dir(target.dataset)
    if shard_dir is None:
        return ([], [])
    shards = sm.load_shard_query_ids(shard_dir)
    split_ids = sm.load_split_query_ids(target.dataset, target.split)
    if split_ids is not None:
        shards = sm.filter_shards_by_split(shards, split_ids)

    run_subdir = _resolve_run_subdir(target)
    run_dir = os.path.join(
        "runs", target.dataset, retriever, target.split, agent_model, run_subdir,
    )
    completed = sm.scan_completed_ids(run_dir)
    gaps = sm.compute_shard_gaps(shards, completed)

    missing_shards = sorted(
        int(name.split("_")[-1]) for name in gaps
        if name.split("_")[-1].isdigit()
    )
    missing_qids = sorted({q for ids in gaps.values() for q in ids})
    return (missing_shards, missing_qids)


# Modes whose SBATCH trigger is traj_orig_ext / traj_summary_orig_ext —
# react_agent.py requires original_messages in each upstream trajectory JSON.
_ORIG_MESSAGES_MODES = {
    "traj_orig_ext",
    "traj_budget_orig_ext",
    "traj_summary_orig_ext",
}


def _upstream_traj_dir(target: Target, retriever: str = "Qwen3-Embedding-8B") -> Path | None:
    """Return the upstream trajectory dir a traj_*_orig_ext target reads from."""
    if target.mode not in _ORIG_MESSAGES_MODES or not target.traj_model:
        return None
    if target.mode == "traj_budget_orig_ext":
        # Read SEARCH_BUDGET from the template (default 5).
        search_budget = "5"
        try:
            tpl = Path(target.template_path).read_text()
            m = re.search(r"^SEARCH_BUDGET=(\S+)", tpl, re.MULTILINE)
            if m:
                search_budget = m.group(1).strip('"')
        except OSError:
            pass
        subdir = f"budget{search_budget}_seed{target.seed}"
    else:
        subdir = f"seed{target.seed}"
    return (
        PROJECT_ROOT / "runs" / target.dataset / retriever
        / target.split / target.traj_model / subdir
    )


def _check_upstream_original_messages(
    target: Target, sample_size: int = 5,
) -> str | None:
    """Return a preflight error string if the upstream traj dir lacks original_messages."""
    up = _upstream_traj_dir(target)
    if up is None:
        return None
    if not up.is_dir():
        return f"upstream trajectory dir missing: {up}"
    json_files = sorted(up.glob("run_*.json"))
    if not json_files:
        return f"upstream trajectory dir has no run_*.json files: {up}"
    # Sample-check: spread across the list (not just the first few).
    step = max(1, len(json_files) // sample_size)
    sampled = json_files[::step][:sample_size]
    for p in sampled:
        try:
            with p.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return f"cannot read {p}: {e}"
        if not data.get("original_messages"):
            return (
                f"{p.name} in {up} lacks 'original_messages' — "
                f"regenerate upstream with --save-raw-messages before running "
                f"mode={target.mode}"
            )
    return None


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

        orig_err = _check_upstream_original_messages(t)
        if orig_err:
            errors.append(PreflightError(t, orig_err))
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


EVAL_LINE_TEMPLATE = (
    'singularity exec --nv --overlay ${{OVERLAY_FILE}}:ro $SINGULARITY_IMAGE '
    '/bin/bash -c "${{CMD_PREFIX}} '
    '--input_dir ${{BASE}}/{split}/{agent_model}/{run_name}/ '
    '--tensor_parallel_size 1 '
    '--ground_truth {ground_truth} '
    '--eval_dir evals/"'
)

GROUND_TRUTH_BY_SPLIT = {
    "full":      "data/browsecomp_plus_decrypted.jsonl",
    "first50":   "data/browsecomp_plus_decrypted_first50.jsonl",
    "first100":  "data/browsecomp_plus_decrypted_first100.jsonl",
    "test150":   "data/browsecomp_plus_decrypted_test150.jsonl",
    "train680":  "data/browsecomp_plus_decrypted_train680.jsonl",
}


def build_eval_sbatch(
    targets: list[Target],
    eval_template: Path | None = None,
    out_path: Path | None = None,
    retriever: str = "Qwen3-Embedding-8B",
) -> Path:
    """Generate eval_auto.SBATCH with one exec line per complete target."""
    eval_template = eval_template or (PROJECT_ROOT / "eval.SBATCH")
    out_path = out_path or (PROJECT_ROOT / "eval_auto.SBATCH")

    header_text = eval_template.read_text()
    header_lines: list[str] = []
    for line in header_text.splitlines():
        header_lines.append(line)
        if line.strip().startswith("CMD_PREFIX="):
            break

    for i, l in enumerate(header_lines):
        if l.startswith("#SBATCH --job-name="):
            header_lines[i] = "#SBATCH --job-name=eval_auto"
        elif l.startswith("#SBATCH --output="):
            header_lines[i] = "#SBATCH --output=sbatch_outputs/eval_auto.out"

    lines = header_lines + ["", "# --- autogenerated eval lines ---", ""]
    for t in targets:
        gt = GROUND_TRUTH_BY_SPLIT.get(t.split, "data/browsecomp_plus_decrypted.jsonl")
        lines.append(EVAL_LINE_TEMPLATE.format(
            split=t.split, agent_model=t.model,
            run_name=t.run_name, ground_truth=gt,
        ))
        lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


def write_eval_failed(
    slurm_out_path: Path, path: Path | None = None, tail_lines: int = 100,
) -> Path:
    path = path or (PROJECT_ROOT / "eval_failed.md")
    try:
        tail = "\n".join(
            slurm_out_path.read_text(errors="replace").splitlines()[-tail_lines:]
        )
    except OSError:
        tail = "(could not read slurm output)"
    path.write_text(
        f"# Eval Failed\n\n"
        f"SLURM output tail (`{slurm_out_path}`):\n\n```\n{tail}\n```\n"
    )
    return path


_EVAL_STATS_RE = re.compile(
    r"""
    Processed\s+(?P<num_evaluations>\d+)\s+evaluations.*?
    Accuracy:\s+(?P<accuracy>\d+(?:\.\d+)?)%.*?
    Recall:\s+(?P<recall>\d+(?:\.\d+)?)%.*?
    Average\s+Tool\s+Calls:.*?'search':\s*(?P<num_searches>\d+(?:\.\d+)?).*?
    Summary\s+saved\s+to\s+(?P<summary_path>\S+evaluation_summary\.json)
    """,
    re.DOTALL | re.VERBOSE,
)


def _extract_eval_stats(text: str) -> list[dict[str, Any]]:
    """Parse evaluation stats from eval SLURM output. Mirrors src_utils/parse_eval_out."""
    rows: list[dict[str, Any]] = []
    for m in _EVAL_STATS_RE.finditer(text):
        summary_path = m.group("summary_path")
        parts = Path(summary_path).parts
        run_name = "/".join(parts[-4:-1]) if len(parts) >= 4 else summary_path
        rows.append({
            "run_name": run_name,
            "num_evaluations": int(m.group("num_evaluations")),
            "accuracy": float(m.group("accuracy")),
            "recall": float(m.group("recall")),
            "num_searches": float(m.group("num_searches")),
        })
    return rows


def write_summary(
    state: PipelineState,
    eval_out_path: Path | None = None,
    path: Path | None = None,
) -> Path:
    """Generate pipeline_summary.md from PipelineState + parsed eval output."""
    path = path or (PROJECT_ROOT / "pipeline_summary.md")
    eval_out_path = eval_out_path or (PROJECT_ROOT / "sbatch_outputs" / "eval_auto.out")

    rows: list[dict[str, Any]] = []
    if eval_out_path.is_file():
        rows = _extract_eval_stats(eval_out_path.read_text())

    lines = [
        "# Pipeline Summary",
        "",
        f"- Started: {state.started_at}",
        f"- Ended: {state.last_check_at}",
        f"- Cycles: {state.cycle_count}",
        f"- Targets: {len(state.targets)}",
        "",
        "## Eval Results",
        "",
        "| Run | # | Accuracy | Recall | Avg Searches |",
        "|-----|---|----------|--------|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['num_evaluations']} | "
            f"{r['accuracy']:.1f}% | {r['recall']:.1f}% | "
            f"{r['num_searches']:.2f} |"
        )
    if not rows:
        lines.append("| _(no rows parsed from eval output)_ | | | | |")

    lines.extend(["", "## Target Status", ""])
    for s in state.targets:
        lines.append(
            f"- `{s.target.run_name}` — status: {s.status}, "
            f"resubmits: {len(s.submitted_job_ids)}"
        )
    path.write_text("\n".join(lines))
    return path


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


def _tail_sbatch_out_for(target: Target, n_lines: int = 80) -> str:
    """Return a tail of the latest sbatch_outputs/*.out matching this target."""
    import glob as gb

    split_tag = f"_{target.split}" if target.split != "full" else ""
    pat = f"sbatch_outputs/{target.run_name}{split_tag}*.out"
    candidates = sorted(gb.glob(pat), key=os.path.getmtime, reverse=True)
    if not candidates:
        return "_(no matching sbatch output file found)_"
    path = candidates[0]
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        return f"_(failed to read {path}: {e})_"
    lines = text.splitlines()[-n_lines:]
    return f"Tail of `{path}` (last {len(lines)} lines):\n\n```\n" + "\n".join(lines) + "\n```"


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
        job_ids_str = ", ".join(str(j) for j in s.submitted_job_ids) or "(none)"
        lines.extend([
            f"## Stuck: {s.target.run_name}",
            f"- Dataset: {s.target.dataset} | Split: {s.target.split} | Model: {s.target.model}",
            f"- Mode: {s.target.mode} | Seed: {s.target.seed} | Traj model: {s.target.traj_model}",
            f"- Stuck cycles: {s.stuck_cycles}",
            f"- Resubmissions ({len(s.submitted_job_ids)}): {job_ids_str}",
            f"- Missing query IDs ({len(s.missing_qids)}): {missing_preview}",
            "",
            "### Diagnostic log excerpts (error-pattern scan)",
            diagnose_slurm_out(s.target),
            "",
            "### Latest sbatch output tail",
            _tail_sbatch_out_for(s.target),
            "",
        ])
    lines.extend(["## All targets summary", ""])
    for s in all_states:
        flag = " (stuck)" if s in stuck_states else ""
        lines.append(
            f"- `{s.target.run_name}`: status={s.status}, "
            f"missing={len(s.missing_qids)}, "
            f"resubmits={len(s.submitted_job_ids)}{flag}"
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


def _delete_empty_runs_for(target: Target, retriever: str = "Qwen3-Embedding-8B") -> None:
    """Delete empty trajectory JSONs in a target's run dir (reuses filter_empty_runs)."""
    import sys as _sys

    _sys.path.insert(0, str(PROJECT_ROOT / "src_utils"))
    try:
        from filter_empty_runs import scan_empty_runs, _delete_paths  # type: ignore
    except ImportError:
        return
    run_subdir = _resolve_run_subdir(target)
    run_dir = (
        PROJECT_ROOT / "runs" / target.dataset / retriever
        / target.split / target.model / run_subdir
    )
    if not run_dir.is_dir():
        return
    scan = scan_empty_runs(run_dir)
    if scan.empty_paths:
        log.info("deleting %d empty runs in %s", len(scan.empty_paths), run_dir)
        _delete_paths(scan.empty_paths)


def run_pipeline(args) -> int:
    """Orchestrate preflight → submit → monitor → eval → summary."""
    if args.resume and STATE_PATH.is_file():
        state = load_state()
        log.info("resumed state at cycle %d, phase=%s", state.cycle_count, state.phase)
    else:
        targets = collect_targets()
        state = PipelineState(
            pid=os.getpid(),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            interval_seconds=args.interval_seconds,
            stuck_threshold=args.stuck_threshold,
            targets=[TargetState(target=t) for t in targets],
        )
        if not state.targets:
            log.warning("no targets — all MISSING* dicts are empty")
            return 0

    if not args.skip_preflight and state.phase in ("init", "preflight"):
        state.phase = "preflight"
        save_state(state)
        errors = preflight(
            [s.target for s in state.targets], run_sbatch_check=args.submit,
        )
        if errors:
            p = write_preflight_failed(errors)
            log.error("preflight failed — see %s", p)
            return 1

    state.phase = "submitting"
    save_state(state)
    for ts in state.targets:
        missing_shards, missing_qids = compute_actual_missing(ts.target)
        ts.missing_qids = missing_qids
        if not missing_shards:
            ts.status = "complete"
            continue
        jid = submit_target(ts.target, shards=missing_shards, submit=args.submit)
        if jid is not None:
            ts.submitted_job_ids.append(jid)
            ts.status = "submitting"
        else:
            ts.status = "running" if args.submit else "pending"
    save_state(state)

    state.phase = "monitoring"
    save_state(state)
    while True:
        state.cycle_count += 1
        state.last_check_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        all_active_ids = [j for ts in state.targets for j in ts.submitted_job_ids]
        states_map = poll_jobs(all_active_ids) if args.submit else {}
        still_running = [j for j, st in states_map.items() if st != "DONE"]
        if still_running:
            log.info("cycle %d: %d jobs still active, sleeping %ds",
                     state.cycle_count, len(still_running), state.interval_seconds)
            save_state(state)
            time.sleep(state.interval_seconds)
            continue

        for ts in state.targets:
            ts.last_missing_qids = list(ts.missing_qids)
            missing_shards, missing_qids = compute_actual_missing(ts.target)
            ts.missing_qids = missing_qids
            if not missing_shards:
                _delete_empty_runs_for(ts.target)
                missing_shards, missing_qids = compute_actual_missing(ts.target)
                ts.missing_qids = missing_qids
                if not missing_shards:
                    ts.status = "complete"
                    continue
            ts.status = "running"

        stuck = detect_stuck_targets(state.targets, threshold=state.stuck_threshold)
        if stuck:
            for s in stuck:
                s.status = "stuck"
            p = write_pipeline_stopped(stuck, state.targets)
            state.phase = "stuck"
            save_state(state)
            log.error("pipeline halted — see %s", p)
            return 2

        any_submitted = False
        for ts in state.targets:
            if ts.status == "complete":
                continue
            missing_shards, _ = compute_actual_missing(ts.target)
            if missing_shards:
                jid = submit_target(ts.target, shards=missing_shards, submit=args.submit)
                if jid is not None:
                    ts.submitted_job_ids.append(jid)
                    any_submitted = True

        save_state(state)
        if all(ts.status == "complete" for ts in state.targets):
            break
        if not any_submitted and not args.submit:
            break
        time.sleep(state.interval_seconds)

    if args.skip_eval:
        state.phase = "done"
        save_state(state)
        log.info("skipping eval (--skip-eval)")
        return 0

    state.phase = "eval"
    save_state(state)
    complete_targets = [ts.target for ts in state.targets if ts.status == "complete"]
    sbatch_path = build_eval_sbatch(complete_targets)
    if not args.submit:
        log.info("[dry-run] would submit %s", sbatch_path)
        state.phase = "done"
        save_state(state)
        return 0
    result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    if result.returncode != 0:
        write_eval_failed(PROJECT_ROOT / "sbatch_outputs" / "eval_auto.out")
        return 3
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    eval_jid = int(m.group(1)) if m else None
    if eval_jid is not None:
        while poll_jobs([eval_jid]).get(eval_jid, "DONE") != "DONE":
            time.sleep(state.interval_seconds)

    eval_out_path = PROJECT_ROOT / "sbatch_outputs" / "eval_auto.out"
    if eval_out_path.is_file() and "Processed" not in eval_out_path.read_text():
        write_eval_failed(eval_out_path)
        return 3

    state.phase = "done"
    save_state(state)
    write_summary(state, eval_out_path=eval_out_path)
    log.info("done — see pipeline_summary.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Actually submit (default: dry-run)")
    parser.add_argument("--interval-seconds", type=int, default=14400,
                        help="Seconds between completion checks (default: 14400 = 4h)")
    parser.add_argument("--stuck-threshold", type=int, default=6,
                        help="Halt after this many consecutive no-progress checks (default: 6)")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _setup_logging()
    log.info("auto_pipeline.py starting (submit=%s, interval=%ds)",
             args.submit, args.interval_seconds)
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
