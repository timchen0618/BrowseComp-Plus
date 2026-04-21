# Auto-Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `auto_pipeline.py` — a single-command automation that takes the `MISSING*` dicts in `submit_missing.py`, pre-flights SBATCH files, submits only actually-missing shards, polls SLURM on a configurable interval, resubmits gaps, deletes empty runs, detects stuck targets, submits eval, and writes a summary.

**Architecture:** One main script at project root. Reuses existing modules verbatim: `submit_missing` (MISSING dicts + patch_sbatch + parse_run_name), `shard_monitor` (completeness oracle), `src_utils/filter_empty_runs` (empty-run cleanup), `src_utils/parse_eval_out` (eval result parsing). Pure helpers exposed for unit testing via pytest. State persisted to `auto_pipeline_state.json` for `--resume`.

**Tech Stack:** Python 3.10+ stdlib only (subprocess, json, logging, argparse, dataclasses, pathlib, re); pytest (tests); bash (nohup wrapper).

---

## File Structure

| File | Purpose |
|------|---------|
| `auto_pipeline.py` | Main single-file script with pure helpers + orchestrator |
| `auto_pipeline.sh` | Thin `nohup`-friendly bash wrapper |
| `tests/__init__.py` | Empty package marker |
| `tests/conftest.py` | Shared fixtures: temp project dir, fake `MISSING` dicts, fake slurm logs |
| `tests/test_auto_pipeline.py` | Unit tests for pure helpers + dry-run smoke test |

Spec reference: `docs/superpowers/specs/2026-04-20-auto-pipeline-design.md`.

---

### Task 1: Scaffolding — types, skeleton module, pytest config

**Files:**
- Create: `auto_pipeline.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_auto_pipeline.py`
- Modify: `pyproject.toml` (add pytest to dev deps + tool.pytest.ini_options)

- [ ] **Step 1: Add pytest config and dev dep**

Edit `pyproject.toml` — append after `[tool.poetry]` block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create empty package + conftest**

`tests/__init__.py` — empty file.

`tests/conftest.py`:

```python
import json
from pathlib import Path
import pytest


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch):
    """Temp project dir with minimal subdirs the pipeline writes to."""
    (tmp_path / "sbatch_outputs").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "evals").mkdir()
    (tmp_path / "topics-qrels" / "bcp" / "bcp_test150_3_shards").mkdir(parents=True)
    # 3 shards, 2 qids each
    for i, qids in enumerate([["q1", "q2"], ["q3", "q4"], ["q5", "q6"]]):
        (tmp_path / "topics-qrels" / "bcp" / "bcp_test150_3_shards" / f"q_{i}.tsv").write_text(
            "\n".join(f"{q}\tsome text" for q in qids) + "\n"
        )
    (tmp_path / "topics-qrels" / "bcp" / "queries_test150.tsv").write_text(
        "\n".join(f"{q}\tsome text" for q in ["q1", "q2", "q3", "q4", "q5", "q6"]) + "\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sample_slurm_out(tmp_project: Path) -> Path:
    """An example SLURM output with a common error pattern."""
    path = tmp_project / "slurm-9999.out"
    path.write_text(
        "Starting job ...\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        "Traceback (most recent call last):\n"
        '  File "/app/oss_client.py", line 42, in <module>\n'
        "    main()\n"
        "RuntimeError: CUDA error\n"
    )
    return path
```

- [ ] **Step 3: Create skeleton `auto_pipeline.py` with dataclasses + module docstring**

```python
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
```

- [ ] **Step 4: Add import-smoke test to `tests/test_auto_pipeline.py`**

```python
import auto_pipeline


def test_module_imports():
    assert hasattr(auto_pipeline, "Target")
    assert hasattr(auto_pipeline, "TargetState")
    assert hasattr(auto_pipeline, "PipelineState")
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `cd /scratch/hc3337/projects/BrowseComp-Plus && python -m pytest tests/test_auto_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add auto_pipeline.py tests/__init__.py tests/conftest.py tests/test_auto_pipeline.py pyproject.toml
git commit -m "feat(auto-pipeline): scaffold module with dataclasses and pytest config"
```

---

### Task 2: `collect_targets()` — flatten `submit_missing.MISSING*` dicts into `Target` list

**Files:**
- Modify: `auto_pipeline.py` (add function)
- Modify: `tests/test_auto_pipeline.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auto_pipeline.py`:

```python
def test_collect_targets_reads_missing_dicts(monkeypatch):
    import submit_missing
    # Clear all MISSING* dicts, then inject one per split/dataset
    monkeypatch.setattr(submit_missing, "MISSING", {"tongyi_seed3": [0, 1]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FRAMES_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_MUSIQUE_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TEST150",
                        {"gpt-oss-120b_seed0": [2]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TRAIN680", {}, raising=False)

    targets = auto_pipeline.collect_targets()
    assert len(targets) == 2
    names = {t.run_name for t in targets}
    assert names == {"tongyi_seed3", "gpt-oss-120b_seed0"}
    tongyi = next(t for t in targets if t.run_name == "tongyi_seed3")
    assert tongyi.split == "full"
    assert tongyi.dataset == "bcp"
    assert tongyi.model == "tongyi"
    assert tongyi.seed == 3
    assert tongyi.declared_shards == [0, 1]
    test150 = next(t for t in targets if t.run_name == "gpt-oss-120b_seed0")
    assert test150.split == "test150"
    assert test150.declared_shards == [2]
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python -m pytest tests/test_auto_pipeline.py::test_collect_targets_reads_missing_dicts -v`
Expected: FAIL — `AttributeError: module 'auto_pipeline' has no attribute 'collect_targets'`.

- [ ] **Step 3: Implement `collect_targets()`**

Add to `auto_pipeline.py` (above `main`):

```python
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
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/test_auto_pipeline.py::test_collect_targets_reads_missing_dicts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): collect_targets flattens MISSING dicts"
```

---

### Task 3: `preflight()` + `write_preflight_failed()` — validate SBATCH + paths

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_preflight_detects_missing_template(tmp_project, monkeypatch):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="does_not_exist.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    errors = auto_pipeline.preflight([t], run_sbatch_check=False)
    assert len(errors) == 1
    assert "does_not_exist.SBATCH" in errors[0].reason


def test_preflight_runs_sbatch_test_only(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=test\n"
        "#SBATCH --output=sbatch_outputs/test.out\n"
        "#SBATCH --array=0\n"
        'MODEL_NAME="tongyi"\n'
        'mode="org"\n'
        'seed=0\n'
        'dataset="bcp"\n'
        "echo ok\n"
    )
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, stdout="sbatch: Job 123 would be submitted\n", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    errors = auto_pipeline.preflight([t], run_sbatch_check=True)
    assert errors == []
    assert any("--test-only" in c for c in calls[0])
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_auto_pipeline.py -k preflight -v`
Expected: FAIL — `preflight` not defined.

- [ ] **Step 3: Implement `preflight()` + `write_preflight_failed()`**

Add:

```python
@dataclass
class PreflightError:
    target: Target
    reason: str


def preflight(targets: list[Target], run_sbatch_check: bool = True) -> list[PreflightError]:
    """Validate SBATCH templates and referenced paths. Returns list of errors (empty = OK)."""
    import submit_missing as sm
    import tempfile

    errors: list[PreflightError] = []
    for t in targets:
        if not os.path.isfile(t.template_path):
            errors.append(PreflightError(t, f"template not found: {t.template_path}"))
            continue
        with open(t.template_path) as f:
            template = f.read()
        try:
            patched = sm.patch_sbatch(
                template, t.run_name, t.model, t.mode, t.seed,
                shards=t.declared_shards if t.declared_shards else [0],
                dataset=t.dataset, split=t.split, traj_model=t.traj_model,
            )
        except Exception as e:
            errors.append(PreflightError(t, f"patch_sbatch failed: {e}"))
            continue

        if run_sbatch_check:
            with tempfile.NamedTemporaryFile("w", suffix=".SBATCH", delete=False) as tmp:
                tmp.write(patched)
                tmp_path = tmp.name
            try:
                result = subprocess.run(
                    ["sbatch", "--test-only", tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                # sbatch --test-only prints to stderr and exits 0 on success
                if result.returncode != 0:
                    errors.append(PreflightError(
                        t, f"sbatch --test-only failed: {result.stderr.strip()}"
                    ))
            finally:
                os.unlink(tmp_path)
    return errors


def write_preflight_failed(errors: list[PreflightError], path: Path = None) -> Path:
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
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python -m pytest tests/test_auto_pipeline.py -k preflight -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): preflight validates templates + sbatch --test-only"
```

---

### Task 4: `compute_actual_missing()` — ground-truth completeness per target

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_actual_missing_reads_run_dir(tmp_project):
    run_dir = tmp_project / "runs" / "bcp" / "Qwen3-Embedding-8B" / "test150" / "gpt-oss-120b" / "x_seed0"
    run_dir.mkdir(parents=True)
    # Completed q1, q3
    (run_dir / "run_q1.json").write_text('{"query_id": "q1", "result": [1]}')
    (run_dir / "run_q3.json").write_text('{"query_id": "q3", "result": [1]}')

    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="run_qwen3_test150.SBATCH", declared_shards=[0, 1, 2],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    missing_shards, missing_qids = auto_pipeline.compute_actual_missing(
        t, retriever="Qwen3-Embedding-8B", agent_model="gpt-oss-120b"
    )
    # q2 missing from shard 0, q4 from shard 1, q5+q6 from shard 2
    assert set(missing_shards) == {0, 1, 2}
    assert set(missing_qids) == {"q2", "q4", "q5", "q6"}
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python -m pytest tests/test_auto_pipeline.py::test_compute_actual_missing_reads_run_dir -v`

- [ ] **Step 3: Implement**

Add:

```python
def compute_actual_missing(
    target: Target,
    retriever: str = "Qwen3-Embedding-8B",
    agent_model: str | None = None,
) -> tuple[list[int], list[str]]:
    """Return (missing_shard_indices, missing_query_ids) for a target by scanning its run dir."""
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

    missing_shards = sorted(int(s.split("_")[-1]) for s in gaps.keys()
                            if s.split("_")[-1].isdigit())
    missing_qids = sorted({q for ids in gaps.values() for q in ids})
    return (missing_shards, missing_qids)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): compute_actual_missing wraps shard_monitor"
```

---

### Task 5: `submit_target()` — render SBATCH + call sbatch

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_submit_target_dry_run_returns_none(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(
        "#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --output=sbatch_outputs/x.out\n"
        "#SBATCH --array=0\n"
        'MODEL_NAME="tongyi"\nmode="org"\nseed=0\ndataset="bcp"\necho ok\n'
    )
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0, 1],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    jid = auto_pipeline.submit_target(t, shards=[0, 1], submit=False)
    assert jid is None  # dry-run


def test_submit_target_submits_and_parses_jobid(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(
        "#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --output=sbatch_outputs/x.out\n"
        "#SBATCH --array=0\n"
        'MODEL_NAME="tongyi"\nmode="org"\nseed=0\ndataset="bcp"\necho ok\n'
    )
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0, 1],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, stdout="Submitted batch job 987654\n", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    jid = auto_pipeline.submit_target(t, shards=[0, 1], submit=True)
    assert jid == 987654
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement**

```python
def submit_target(target: Target, shards: list[int], submit: bool = False) -> int | None:
    """Render and submit target's SBATCH for the given shards. Returns SLURM job_id or None (dry-run/failure)."""
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
        os.unlink(tmp_path)
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
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): submit_target wraps patch_sbatch + sbatch call"
```

---

### Task 6: `poll_jobs()` — squeue status for a set of job_ids

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_poll_jobs_parses_squeue(monkeypatch):
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        out = "JOBID STATE\n987654 RUNNING\n987655 PENDING\n"
        return sp.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    states = auto_pipeline.poll_jobs([987654, 987655, 987656])
    assert states[987654] == "RUNNING"
    assert states[987655] == "PENDING"
    assert states[987656] == "DONE"  # not in squeue → treat as finished


def test_poll_jobs_empty_list():
    assert auto_pipeline.poll_jobs([]) == {}
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement**

```python
def poll_jobs(job_ids: list[int]) -> dict[int, str]:
    """Return {job_id: state}. Jobs absent from squeue are 'DONE'."""
    if not job_ids:
        return {}
    jobs_arg = ",".join(str(j) for j in job_ids)
    result = subprocess.run(
        ["squeue", "-j", jobs_arg, "-h", "-o", "%i %T"],
        capture_output=True, text=True,
    )
    states: dict[int, str] = {jid: "DONE" for jid in job_ids}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                states[int(parts[0])] = parts[1]
            elif len(parts) >= 2:
                # array jobs appear as "12345_0"
                head = parts[0].split("_")[0]
                if head.isdigit() and int(head) in states:
                    # Mark parent as still running if any element is
                    if states[int(head)] == "DONE":
                        states[int(head)] = parts[1]
    return states
```

Note the test uses a header row `JOBID STATE` — update the test to match the `-h` format (no header). Fix test:

```python
def test_poll_jobs_parses_squeue(monkeypatch):
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        out = "987654 RUNNING\n987655 PENDING\n"
        return sp.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    states = auto_pipeline.poll_jobs([987654, 987655, 987656])
    assert states[987654] == "RUNNING"
    assert states[987655] == "PENDING"
    assert states[987656] == "DONE"
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): poll_jobs wraps squeue -h"
```

---

### Task 7: `detect_stuck_targets()` — compare missing-qid set across cycles

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def _mk_target_state(name, missing, last, stuck):
    t = auto_pipeline.Target(
        run_name=name, dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    return auto_pipeline.TargetState(
        target=t, missing_qids=missing, last_missing_qids=last, stuck_cycles=stuck,
    )


def test_detect_stuck_targets_flags_unchanged():
    states = [
        _mk_target_state("A", ["q1", "q2"], ["q1", "q2"], 2),  # unchanged again
        _mk_target_state("B", ["q3"],      ["q4", "q3"], 1),   # shrank
        _mk_target_state("C", [],          [],           0),   # complete
    ]
    stuck = auto_pipeline.detect_stuck_targets(states, threshold=3)
    # A hits threshold this cycle (stuck_cycles becomes 3)
    assert [s.target.run_name for s in stuck] == ["A"]
    # B's stuck_cycles resets to 0
    assert states[1].stuck_cycles == 0
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
def detect_stuck_targets(
    states: list[TargetState], threshold: int,
) -> list[TargetState]:
    """Mutates each state's stuck_cycles based on missing-qid comparison.
    Returns targets whose stuck_cycles >= threshold."""
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
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): detect_stuck_targets flags no-progress runs"
```

---

### Task 8: `diagnose_slurm_out()` + `write_pipeline_stopped()`

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_diagnose_slurm_out_finds_oom(tmp_project, sample_slurm_out):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    # sample_slurm_out is already in tmp_project root
    text = auto_pipeline.diagnose_slurm_out(t, slurm_glob="slurm-*.out", sbatch_glob="sbatch_outputs/*.out")
    assert "CUDA out of memory" in text
    assert "slurm-9999.out" in text
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
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
    """Grep recent SLURM/sbatch out files for common error patterns. Returns Markdown."""
    import glob as gb
    candidates = sorted(gb.glob(slurm_glob), key=os.path.getmtime, reverse=True)[:max_files]
    candidates += sorted(gb.glob(sbatch_glob), key=os.path.getmtime, reverse=True)[:max_files]
    out = []
    for path in candidates:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        # Only surface files mentioning the run_name OR matching a global error
        if target.run_name not in text and not any(p.search(text) for p in ERROR_PATTERNS):
            continue
        lines = text.splitlines()
        matches = []
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
    stuck_states: list[TargetState], all_states: list[TargetState], path: Path = None,
) -> Path:
    path = path or (PROJECT_ROOT / "pipeline_stopped.md")
    lines = ["# Pipeline Stopped — Stuck Target(s) Detected", ""]
    for s in stuck_states:
        lines.extend([
            f"## Stuck: {s.target.run_name}",
            f"- Dataset: {s.target.dataset} | Split: {s.target.split} | Model: {s.target.model}",
            f"- Stuck cycles: {s.stuck_cycles}",
            f"- Missing query IDs ({len(s.missing_qids)}): {', '.join(s.missing_qids[:30])}"
            + ("..." if len(s.missing_qids) > 30 else ""),
            "",
            "### Diagnostic log excerpts",
            diagnose_slurm_out(s.target),
            "",
        ])
    lines.extend(["## All targets summary", ""])
    for s in all_states:
        flag = " (stuck)" if s in stuck_states else ""
        lines.append(f"- `{s.target.run_name}`: status={s.status}, "
                     f"missing={len(s.missing_qids)}{flag}")
    path.write_text("\n".join(lines))
    return path
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): diagnose_slurm_out + pipeline_stopped.md writer"
```

---

### Task 9: State persistence — `save_state()` / `load_state()`

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_save_and_load_state_roundtrip(tmp_project):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0, 1],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    ts = auto_pipeline.TargetState(
        target=t, missing_qids=["q1"], last_missing_qids=[], stuck_cycles=0,
        submitted_job_ids=[42], status="running",
    )
    state = auto_pipeline.PipelineState(
        pid=123, started_at="2026-04-20T10:00:00", phase="monitoring",
        interval_seconds=600, stuck_threshold=3, targets=[ts],
    )
    path = tmp_project / "auto_pipeline_state.json"
    auto_pipeline.save_state(state, path)
    loaded = auto_pipeline.load_state(path)
    assert loaded.pid == 123
    assert loaded.phase == "monitoring"
    assert len(loaded.targets) == 1
    assert loaded.targets[0].target.run_name == "x_seed0"
    assert loaded.targets[0].missing_qids == ["q1"]
    assert loaded.targets[0].submitted_job_ids == [42]
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
def save_state(state: PipelineState, path: Path = None) -> None:
    path = path or STATE_PATH
    path.write_text(json.dumps(asdict(state), indent=2))


def load_state(path: Path = None) -> PipelineState:
    path = path or STATE_PATH
    data = json.loads(path.read_text())
    targets = []
    for ts in data.get("targets", []):
        t = Target(**ts["target"])
        ts_copy = dict(ts)
        ts_copy["target"] = t
        targets.append(TargetState(**ts_copy))
    data["targets"] = targets
    return PipelineState(**data)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): JSON save_state/load_state with roundtrip"
```

---

### Task 10: `build_eval_sbatch()` + `write_eval_failed()`

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_eval_sbatch_generates_one_exec_per_target(tmp_project):
    header_path = tmp_project / "eval.SBATCH"
    header_path.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=eval\n"
        "#SBATCH --output=sbatch_outputs/eval.out\n"
        "SINGULARITY_IMAGE=/img.sif\n"
        "OVERLAY_FILE=/ov.ext3\n"
        "BASE=\"runs/bcp/Qwen3-Embedding-8B\"\n"
        "CMD_PREFIX=\"python eval.py\"\n"
        "# --- autogenerated eval lines below ---\n"
    )
    t1 = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    t2 = auto_pipeline.Target(
        run_name="y_seed0", dataset="bcp", split="full",
        template_path="y.SBATCH", declared_shards=[0],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    out = tmp_project / "eval_auto.SBATCH"
    auto_pipeline.build_eval_sbatch(
        [t1, t2], eval_template=header_path, out_path=out,
        retriever="Qwen3-Embedding-8B",
    )
    content = out.read_text()
    assert "x_seed0" in content
    assert "y_seed0" in content
    # expect ${BASE}/{split}/{model}/{run_name}/
    assert "test150/gpt-oss-120b/x_seed0" in content
    assert "full/tongyi/y_seed0" in content
    assert "singularity exec" in content
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
EVAL_LINE_TEMPLATE = (
    'singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE '
    '/bin/bash -c "${CMD_PREFIX} '
    '--input_dir ${BASE}/{split}/{agent_model}/{run_name}/ '
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
    eval_template: Path = None,
    out_path: Path = None,
    retriever: str = "Qwen3-Embedding-8B",
) -> Path:
    eval_template = eval_template or (PROJECT_ROOT / "eval.SBATCH")
    out_path = out_path or (PROJECT_ROOT / "eval_auto.SBATCH")
    header_text = eval_template.read_text()
    # Preserve through the first executable 'CMD_PREFIX' assignment; drop old exec lines
    header_lines = []
    for line in header_text.splitlines():
        header_lines.append(line)
        if line.strip().startswith("CMD_PREFIX="):
            break
    lines = header_lines + ["", "# --- autogenerated eval lines ---", ""]
    # Override header's job-name + output so it doesn't clash with eval.SBATCH
    for i, l in enumerate(lines):
        if l.startswith("#SBATCH --job-name="):
            lines[i] = "#SBATCH --job-name=eval_auto"
        elif l.startswith("#SBATCH --output="):
            lines[i] = "#SBATCH --output=sbatch_outputs/eval_auto.out"
    for t in targets:
        gt = GROUND_TRUTH_BY_SPLIT.get(t.split, "data/browsecomp_plus_decrypted.jsonl")
        agent_model = t.model
        lines.append(EVAL_LINE_TEMPLATE.format(
            split=t.split, agent_model=agent_model,
            run_name=t.run_name, ground_truth=gt,
        ))
        lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


def write_eval_failed(slurm_out_path: Path, path: Path = None, tail_lines: int = 100) -> Path:
    path = path or (PROJECT_ROOT / "eval_failed.md")
    try:
        tail = "\n".join(slurm_out_path.read_text(errors="replace").splitlines()[-tail_lines:])
    except OSError:
        tail = "(could not read slurm output)"
    path.write_text(
        f"# Eval Failed\n\n"
        f"SLURM output tail (`{slurm_out_path}`):\n\n```\n{tail}\n```\n"
    )
    return path
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): generate eval_auto.SBATCH + eval_failed.md"
```

---

### Task 11: `write_summary()` — parse eval output → `pipeline_summary.md`

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_write_summary_from_eval_out(tmp_project):
    eval_out = tmp_project / "sbatch_outputs" / "eval_auto.out"
    eval_out.write_text(
        "Processed 150 evaluations\n"
        "Accuracy: 65.3%\n"
        "Recall: 72.1%\n"
        "Average Tool Calls: {'search': 8.4}\n"
        "Summary saved to evals/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b/x_seed0/evaluation_summary.json\n"
    )
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    state = auto_pipeline.PipelineState(
        pid=1, started_at="2026-04-20T10:00:00", last_check_at="2026-04-20T12:00:00",
        cycle_count=1, phase="done", interval_seconds=7200, stuck_threshold=3,
        targets=[auto_pipeline.TargetState(target=t, status="complete")],
    )
    out = tmp_project / "pipeline_summary.md"
    auto_pipeline.write_summary(state, eval_out_path=eval_out, path=out)
    text = out.read_text()
    assert "65.3" in text
    assert "x_seed0" in text
    assert "cycle_count" in text.lower() or "cycles" in text.lower()
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
def write_summary(
    state: PipelineState,
    eval_out_path: Path = None,
    path: Path = None,
) -> Path:
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src_utils"))
    import parse_eval_out as peo  # type: ignore

    path = path or (PROJECT_ROOT / "pipeline_summary.md")
    eval_out_path = eval_out_path or (PROJECT_ROOT / "sbatch_outputs" / "eval_auto.out")
    rows = []
    if eval_out_path.is_file():
        rows = peo.extract_all_eval_stats(eval_out_path.read_text())

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
        lines.append(f"| {r['run_name']} | {r['num_evaluations']} | "
                     f"{r['accuracy']:.1f}% | {r['recall']:.1f}% | "
                     f"{r['num_searches']:.2f} |")
    if not rows:
        lines.append("| _(no rows parsed from eval output)_ | | | | |")
    lines.extend(["", "## Target Status", ""])
    for s in state.targets:
        lines.append(f"- `{s.target.run_name}` — status: {s.status}, "
                     f"resubmits: {len(s.submitted_job_ids)}")
    path.write_text("\n".join(lines))
    return path
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): write_summary generates pipeline_summary.md"
```

---

### Task 12: Main orchestrator + CLI wiring

**Files:**
- Modify: `auto_pipeline.py`
- Modify: `tests/test_auto_pipeline.py`

- [ ] **Step 1: Write the integration test (dry-run end-to-end)**

Append:

```python
def test_main_dry_run_does_not_submit(tmp_project, monkeypatch):
    """Smoke test: dry-run flow completes without calling sbatch."""
    import submit_missing
    # Use an existing (real) template copied in
    for name in ("run_qwen3_planning.SBATCH", "run_qwen3_first50.SBATCH",
                 "run_qwen3_test150.SBATCH", "run_qwen3_train680.SBATCH"):
        dest = tmp_project / name
        src = Path("/scratch/hc3337/projects/BrowseComp-Plus") / name
        dest.write_text(src.read_text())
    # Force one fake target and clear others
    monkeypatch.setattr(submit_missing, "MISSING", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FRAMES_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_MUSIQUE_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TEST150",
                        {"gpt-oss-120b_seed0": [0]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TRAIN680", {}, raising=False)
    # Pretend everything's already done so the loop exits immediately
    run_dir = tmp_project / "runs" / "bcp" / "Qwen3-Embedding-8B" / "test150" / "gpt-oss-120b" / "gpt-oss-120b_seed0"
    run_dir.mkdir(parents=True)
    for qid in ["q1", "q2", "q3", "q4", "q5", "q6"]:
        (run_dir / f"run_{qid}.json").write_text(f'{{"query_id": "{qid}"}}')
    sbatch_calls = []
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        sbatch_calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    # Provide the eval.SBATCH header file
    (tmp_project / "eval.SBATCH").write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=eval\n#SBATCH --output=sbatch_outputs/eval.out\n"
        "BASE=\"runs/bcp/Qwen3-Embedding-8B\"\n"
        "CMD_PREFIX=\"python eval.py\"\n"
    )
    monkeypatch.setattr(sys, "argv", ["auto_pipeline.py", "--skip-preflight", "--skip-eval"])
    rc = auto_pipeline.main()
    assert rc == 0
    # No sbatch calls expected in dry-run + everything-complete scenario
    assert not any("sbatch" in c[0] for c in sbatch_calls)
```

- [ ] **Step 2: Run test — expect FAIL (main is currently a stub)**

- [ ] **Step 3: Implement the orchestrator**

Replace `main()` in `auto_pipeline.py`:

```python
def run_pipeline(args) -> int:
    """Orchestrate preflight → submit → monitor → eval → summary."""
    # Resume or init state
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
            log.warning("no targets found — all MISSING* dicts are empty or commented out")
            return 0

    # Preflight
    if not args.skip_preflight and state.phase in ("init", "preflight"):
        state.phase = "preflight"
        save_state(state)
        errors = preflight([s.target for s in state.targets],
                           run_sbatch_check=args.submit)
        if errors:
            p = write_preflight_failed(errors)
            log.error("preflight failed — see %s", p)
            return 1

    # Initial submission — only actually-missing shards per target
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

    # Monitor loop
    state.phase = "monitoring"
    save_state(state)
    while True:
        state.cycle_count += 1
        state.last_check_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # If any submitted job is still queued/running, sleep
        all_active_ids = [j for ts in state.targets for j in ts.submitted_job_ids]
        states_map = poll_jobs(all_active_ids) if args.submit else {}
        still_running = [j for j, st in states_map.items() if st != "DONE"]
        if still_running:
            log.info("cycle %d: %d jobs still active, sleeping %ds",
                     state.cycle_count, len(still_running), state.interval_seconds)
            save_state(state)
            time.sleep(state.interval_seconds)
            continue

        # All finished — re-scan each target
        for ts in state.targets:
            ts.last_missing_qids = list(ts.missing_qids)
            missing_shards, missing_qids = compute_actual_missing(ts.target)
            ts.missing_qids = missing_qids
            if not missing_shards:
                # Complete → check for empty runs; delete them if any
                _delete_empty_runs_for(ts.target)
                # Re-check in case deletion created new gaps
                missing_shards, missing_qids = compute_actual_missing(ts.target)
                ts.missing_qids = missing_qids
                if not missing_shards:
                    ts.status = "complete"
                    continue
            # Still incomplete
            ts.status = "running"

        # Stuck detection
        stuck = detect_stuck_targets(state.targets, threshold=state.stuck_threshold)
        if stuck:
            for s in stuck:
                s.status = "stuck"
            p = write_pipeline_stopped(stuck, state.targets)
            state.phase = "stuck"
            save_state(state)
            log.error("pipeline halted — see %s", p)
            return 2

        # Resubmit any non-stuck incomplete targets
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
        # All complete?
        if all(ts.status == "complete" for ts in state.targets):
            break
        if not any_submitted and not args.submit:
            # dry-run: break to avoid infinite loop
            break
        time.sleep(state.interval_seconds)

    # Eval phase
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
    # Check eval_auto.out for success markers
    eval_out_path = PROJECT_ROOT / "sbatch_outputs" / "eval_auto.out"
    if eval_out_path.is_file() and "Processed" not in eval_out_path.read_text():
        write_eval_failed(eval_out_path)
        return 3

    state.phase = "done"
    save_state(state)
    write_summary(state, eval_out_path=eval_out_path)
    log.info("done — see pipeline_summary.md")
    return 0


def _delete_empty_runs_for(target: Target, retriever: str = "Qwen3-Embedding-8B") -> None:
    """Invoke filter_empty_runs delete-empty mode on a target's run dir."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src_utils"))
    from filter_empty_runs import scan_empty_runs, _delete_paths  # type: ignore
    run_dir = PROJECT_ROOT / "runs" / target.dataset / retriever / target.split / target.model / target.run_name
    if not run_dir.is_dir():
        return
    scan = scan_empty_runs(run_dir)
    if scan.empty_paths:
        log.info("deleting %d empty runs in %s", len(scan.empty_paths), run_dir)
        _delete_paths(scan.empty_paths)


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
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
```

- [ ] **Step 4: Run all tests — expect PASS**

Run: `python -m pytest tests/test_auto_pipeline.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add auto_pipeline.py tests/test_auto_pipeline.py
git commit -m "feat(auto-pipeline): orchestrator wires preflight→submit→monitor→eval"
```

---

### Task 13: `auto_pipeline.sh` wrapper

**Files:**
- Create: `auto_pipeline.sh`

- [ ] **Step 1: Create the wrapper**

```bash
#!/bin/bash
# Run the auto-pipeline in the background.
# Usage:
#   nohup bash auto_pipeline.sh --submit > auto_pipeline.log 2>&1 &
#   tail -f auto_pipeline.log
#   cat auto_pipeline_state.json | jq .
set -euo pipefail
cd "$(dirname "$0")"
exec python auto_pipeline.py "$@"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x auto_pipeline.sh`

- [ ] **Step 3: Smoke-check wrapper help prints**

Run: `bash auto_pipeline.sh --help`
Expected: argparse help output listing `--submit`, `--interval-seconds`, etc.

- [ ] **Step 4: Commit**

```bash
git add auto_pipeline.sh
git commit -m "feat(auto-pipeline): nohup-friendly bash wrapper"
```

---

### Task 14: Manual end-to-end verification

**No files changed. Plan for hand-verification.**

- [ ] **Step 1: Configure a small real target**

Edit `submit_missing.py`: uncomment or add a single small entry in `MISSING_TEST150`, e.g.
```python
MISSING_TEST150 = {
    "gpt-oss-120b_seed0": [0],  # tiny: just shard 0 of test150
}
```
Do NOT commit this uncomment — it's test state only.

- [ ] **Step 2: Dry-run the pipeline**

Run: `python auto_pipeline.py --skip-eval --interval-seconds 60`
Expected: preflight passes, one dry-run "would submit" log, loop exits (no real jobs).

- [ ] **Step 3: Real run with short interval**

Run: `nohup bash auto_pipeline.sh --submit --interval-seconds 300 --skip-eval > auto_pipeline.log 2>&1 &`
Monitor: `tail -f auto_pipeline.log` and `cat auto_pipeline_state.json | jq .phase,.cycle_count`

- [ ] **Step 4: Verify completion**

When the target completes, check:
- `auto_pipeline_state.json` phase is `done`
- No `pipeline_stopped.md` or `preflight_failed.md` written
- If re-running with `--skip-eval` removed: `eval_auto.SBATCH` exists and `pipeline_summary.md` lists the run

- [ ] **Step 5: Revert test state**

Re-comment the uncommented `MISSING_TEST150` entry.

- [ ] **Step 6: Commit documentation update**

Append to `CLAUDE.md` "Root-Level Scripts" table:
```
| `auto_pipeline.py` | Automated submit → monitor → resubmit → eval → summary pipeline over submit_missing.py targets |
| `auto_pipeline.sh` | nohup-friendly wrapper for auto_pipeline.py |
```

```bash
git add CLAUDE.md
git commit -m "docs: add auto_pipeline.py to CLAUDE.md root scripts"
```

---

## Self-Review

**Spec coverage:**
- Q1 (single-command, reads MISSING*) → Task 2
- Q2 (background nohup) → Task 13
- Q3 (sbatch --test-only + path checks) → Task 3
- Q4 (filter_empty_runs as-is) → Task 12 (`_delete_empty_runs_for`)
- Q5 (configurable interval, pipeline_stopped.md) → Tasks 7, 8, 12
- Q6 (eval_auto.SBATCH, pipeline_summary.md) → Tasks 10, 11
- State persistence + `--resume` → Task 9 + Task 12
- Preflight failed / eval failed markdown → Tasks 3, 10
- Unit tests + smoke test → all tasks + Task 12 smoke
- Stuck halts entire pipeline → Task 12 (`run_pipeline` returns 2)
- Partial/complete targets use shard_monitor as ground truth → Task 4 + Task 12 initial submission block

**Placeholder scan:** no TBD / "add appropriate error handling" / "similar to Task N" — all steps contain concrete code.

**Type consistency:** `Target`, `TargetState`, `PipelineState` match across tasks. Functions: `collect_targets`, `preflight`, `compute_actual_missing`, `submit_target`, `poll_jobs`, `detect_stuck_targets`, `diagnose_slurm_out`, `save_state`, `load_state`, `build_eval_sbatch`, `write_preflight_failed`, `write_pipeline_stopped`, `write_eval_failed`, `write_summary`, `run_pipeline`, `main` — each used with the signature declared at creation.

All good.
