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
