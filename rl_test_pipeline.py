"""
rl_test_pipeline.py — self-testing orchestrator for the GRPO RL pipeline.

Submits 4 tiered SLURM jobs in sequence, validates each one before advancing.
State is persisted to rl_test_state.json so the script can be called
repeatedly (e.g., every 15 min) until the pipeline passes or fails.

Tiers:
  tier0  CPU   — existing pytest suite (imports, GRPO math, buffer, rollout worker)
  tier1  CPU   — CPU pipeline tests (shape regression, config validation, e2e)
  tier2  1 GPU — daemon-only: explorer + buffer fill (--max-iterations 1)
  tier3  2 GPU — full integration: daemon + GRPO learner

Usage:
  python rl_test_pipeline.py --submit        # advance pipeline (submit/check)
  python rl_test_pipeline.py --submit --reset  # start fresh
  python rl_test_pipeline.py                # dry-run (no sbatch calls)

Output files:
  rl_test_state.json    — current pipeline state
  rl_test_failed.md     — failure report with SLURM output tail
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
from typing import Optional

_ROOT = Path(__file__).resolve().parent
STATE_PATH = _ROOT / "rl_test_state.json"

TIERS = ["tier0", "tier1", "tier2", "tier3"]

SBATCH_FILES: dict[str, Path] = {
    "tier0": _ROOT / "rl/trl/tests/tier0_cpu.SBATCH",
    "tier1": _ROOT / "rl/trl/tests/tier1_cpu.SBATCH",
    "tier2": _ROOT / "rl/trl/tests/tier2_1gpu.SBATCH",
    "tier3": _ROOT / "rl/trl/tests/tier3_2gpu.SBATCH",
}

log = logging.getLogger("rl_test_pipeline")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class TestState:
    phase: str = "init"          # init | tier0 | tier1 | tier2 | tier3 | done | failed
    job_ids: dict = field(default_factory=dict)  # tier -> job_id (int) or None
    failed_tier: Optional[str] = None
    failed_reason: Optional[str] = None
    started_at: str = ""
    last_check_at: str = ""
    poll_count: int = 0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def save_state(state: TestState) -> None:
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2))


def load_state() -> TestState:
    data = json.loads(STATE_PATH.read_text())
    return TestState(**data)


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def submit_job(sbatch_path: Path, submit: bool) -> Optional[int]:
    if not submit:
        log.info("[dry-run] would submit %s", sbatch_path.name)
        return None
    result = subprocess.run(
        ["sbatch", str(sbatch_path)], capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error("sbatch failed for %s: %s", sbatch_path.name, result.stderr.strip())
        return None
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        log.error("could not parse job_id from: %s", result.stdout.strip())
        return None
    jid = int(m.group(1))
    log.info("submitted %s → job %d", sbatch_path.name, jid)
    return jid


def poll_job(job_id: int) -> str:
    """Return RUNNING, PENDING, COMPLETING, or DONE (absent from squeue)."""
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True, text=True,
    )
    state = result.stdout.strip()
    return state if state else "DONE"


def get_exit_code(job_id: int) -> Optional[int]:
    """Query sacct for job exit code. Returns None if unavailable yet."""
    for attempt in range(3):
        try:
            result = subprocess.run(
                [
                    "sacct", "-j", str(job_id), "-n",
                    "--format=JobID,ExitCode", "--parsable2",
                ],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                job_part = parts[0].split(".")[0]
                if job_part == str(job_id):
                    m = re.match(r"(\d+):\d+", parts[1].strip())
                    if m:
                        return int(m.group(1))
        except Exception as exc:
            log.warning("sacct attempt %d failed: %s", attempt + 1, exc)
        if attempt < 2:
            time.sleep(5)
    return None


def find_output_file(tier: str, job_id: Optional[int]) -> Optional[Path]:
    if job_id is None:
        return None
    p = _ROOT / f"sbatch_outputs/rl_test_{tier}_{job_id}.out"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tier(
    tier: str,
    exit_code: Optional[int],
    output_file: Optional[Path],
) -> tuple[bool, str]:
    """Return (success, failure_reason). Empty reason on success."""

    if exit_code is not None and exit_code != 0:
        tail = _read_tail(output_file)
        return False, f"exit code {exit_code}\n{tail}"

    if output_file is None or not output_file.exists():
        return False, f"output file not found for {tier}"

    text = output_file.read_text(errors="replace")

    if tier in ("tier0", "tier1"):
        # pytest output: look for "X passed" or "passed" without "failed"
        has_passed = bool(re.search(r"\d+ passed", text))
        has_failed = bool(re.search(r"\d+ failed", text, re.IGNORECASE))
        has_error  = "ERROR" in text and not has_passed

        if has_failed:
            failures = [l.strip() for l in text.splitlines() if "FAILED" in l][:5]
            return False, "pytest failures:\n  " + "\n  ".join(failures)
        if has_error and not has_passed:
            errors = [l.strip() for l in text.splitlines() if "ERROR" in l][:3]
            return False, "errors without passing tests:\n  " + "\n  ".join(errors)
        if not has_passed:
            return False, f"'X passed' not found in {output_file.name}"
        return True, ""

    if tier == "tier2":
        if "TIER2_FAIL" in text:
            for line in text.splitlines():
                if "TIER2_FAIL" in line:
                    return False, line.strip()
            return False, "TIER2_FAIL marker found"
        if "TIER2_PASS" not in text:
            return False, f"TIER2_PASS marker missing from {output_file.name}"
        return True, ""

    if tier == "tier3":
        if "TIER3_FAIL" in text:
            for line in text.splitlines():
                if "TIER3_FAIL" in line:
                    return False, line.strip()
            return False, "TIER3_FAIL marker found"
        if "TIER3_PASS" not in text:
            return False, f"TIER3_PASS marker missing from {output_file.name}"
        # Check file artifacts
        runs_dir = _ROOT / "rl/trl/runs/val_2gpu"
        if not runs_dir.exists():
            return False, f"model output dir not found: {runs_dir}"
        ckpt_dir = _ROOT / "rl/trl/tmp/val_2gpu_buffer/ckpt_latest"
        if not ckpt_dir.exists():
            return False, f"checkpoint dir not found: {ckpt_dir}"
        return True, ""

    return True, ""


def _read_tail(path: Optional[Path], n: int = 60) -> str:
    if path is None or not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()[-n:]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failure report
# ---------------------------------------------------------------------------

def write_failed_report(state: TestState, output_file: Optional[Path]) -> Path:
    path = _ROOT / "rl_test_failed.md"
    lines = [
        "# RL Test Pipeline Failed",
        "",
        f"- **Failed tier:** `{state.failed_tier}`",
        f"- **Reason:** {state.failed_reason}",
        f"- **Started:** {state.started_at}",
        f"- **Failed at:** {state.last_check_at}",
        "",
        "## SLURM output tail",
        "",
    ]
    if output_file and output_file.exists():
        tail = _read_tail(output_file, n=100)
        lines += [f"File: `{output_file}`", "", "```", tail, "```"]
    else:
        lines.append("_(output file not found)_")

    lines += [
        "",
        "## Recovery",
        "",
        "1. Fix the issue (edit code in the worktree)",
        "2. Re-run: `python rl_test_pipeline.py --submit --reset` (fresh start)",
        "   or `python rl_test_pipeline.py --submit --retry` (retry same tier)",
    ]
    path.write_text("\n".join(lines))
    log.info("Failure report written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------

def _next_tier(current: str) -> Optional[str]:
    idx = TIERS.index(current)
    return TIERS[idx + 1] if idx + 1 < len(TIERS) else None


def run(args) -> int:
    (_ROOT / "sbatch_outputs").mkdir(exist_ok=True)

    # Load or initialize state
    if args.reset or not STATE_PATH.exists():
        state = TestState(started_at=_now())
        save_state(state)
        log.info("Fresh pipeline state created.")
    else:
        state = load_state()

    state.last_check_at = _now()
    state.poll_count += 1

    # Terminal states
    if state.phase == "done":
        log.info("Pipeline already DONE. All tiers passed.")
        return 0

    if state.phase == "failed" and not args.retry:
        log.error(
            "Pipeline FAILED at %s: %s\nSee rl_test_failed.md",
            state.failed_tier, state.failed_reason,
        )
        return 1

    # init → submit tier0
    if state.phase in ("init", "failed"):
        if args.retry:
            retry_tier = state.failed_tier or "tier0"
            log.info("Retrying from %s", retry_tier)
            state.phase = retry_tier
            state.failed_tier = None
            state.failed_reason = None
        else:
            state.phase = "init"

    if state.phase == "init":
        jid = submit_job(SBATCH_FILES["tier0"], args.submit)
        state.phase = "tier0"
        state.job_ids["tier0"] = jid
        save_state(state)
        log.info("Submitted tier0 (job %s). Poll again in ~15 min.", jid)
        return 0

    # In a running tier
    current_tier = state.phase
    job_id = state.job_ids.get(current_tier)

    # Dry-run: no real job_id → pretend job is done with code 0
    if not args.submit:
        log.info("[dry-run] treating %s as complete", current_tier)
        success, reason = validate_tier(
            current_tier,
            exit_code=0,
            output_file=find_output_file(current_tier, job_id),
        )
        if not success:
            log.warning("[dry-run] validation would fail: %s", reason)
        return 0

    job_status = poll_job(job_id) if job_id else "DONE"

    if job_status in ("RUNNING", "PENDING", "COMPLETING"):
        log.info(
            "Poll #%d — %s job %s is %s. Check back in ~15 min.",
            state.poll_count, current_tier, job_id, job_status,
        )
        save_state(state)
        return 0

    # Job finished — validate
    log.info("%s job %s is DONE, validating ...", current_tier, job_id)
    exit_code = get_exit_code(job_id)
    output_file = find_output_file(current_tier, job_id)
    success, reason = validate_tier(current_tier, exit_code, output_file)

    if not success:
        state.phase = "failed"
        state.failed_tier = current_tier
        state.failed_reason = reason
        save_state(state)
        write_failed_report(state, output_file)
        log.error("FAILED at %s: %s", current_tier, reason)
        return 1

    log.info("%s PASSED.", current_tier)
    next_t = _next_tier(current_tier)

    if next_t is None:
        state.phase = "done"
        save_state(state)
        log.info("=" * 60)
        log.info("ALL TIERS PASSED. RL pipeline is validated.")
        log.info("=" * 60)
        return 0

    # Submit next tier
    jid = submit_job(SBATCH_FILES[next_t], args.submit)
    state.phase = next_t
    state.job_ids[next_t] = jid
    save_state(state)
    log.info("Advanced to %s (job %s). Poll again in ~15 min.", next_t, jid)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit SLURM jobs (default: dry-run)")
    parser.add_argument("--reset", action="store_true",
                        help="Start fresh, ignoring any saved state")
    parser.add_argument("--retry", action="store_true",
                        help="Retry from the failed tier (requires --submit)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        return run(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception:
        log.exception("unhandled exception")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
