"""
Filesystem JSONL buffer for async actor-learner communication.

Each sample is written to its own file (sample_<ns>_<pid>.jsonl) under
buffer_dir.  This avoids cross-process file locks while still providing
atomic visibility: a rename is atomic on POSIX, so a reader never sees a
half-written file.

Checkpoint flag protocol:
  - Learner calls write_ckpt_flag(cfg["ckpt_flag"]) after saving weights.
  - Daemon calls check_and_clear_flag(cfg["ckpt_flag"]); returns True once,
    then deletes the flag so subsequent calls return False.
"""

import json
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Write side (rollout daemon)
# ---------------------------------------------------------------------------

def append_sample(buffer_dir: str | Path, sample: dict) -> None:
    """Atomically write one (trajectory, reward) sample to the buffer."""
    buffer_dir = Path(buffer_dir)
    buffer_dir.mkdir(parents=True, exist_ok=True)

    ts = time.monotonic_ns()
    pid = os.getpid()
    dest = buffer_dir / f"sample_{ts}_{pid}.jsonl"
    tmp = dest.with_suffix(".tmp")

    tmp.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.rename(dest)  # atomic on POSIX


# ---------------------------------------------------------------------------
# Read side (learner)
# ---------------------------------------------------------------------------

def buffer_size(buffer_dir: str | Path) -> int:
    """Return the number of unread sample files in the buffer."""
    return len(list(Path(buffer_dir).glob("sample_*.jsonl")))


def pop_samples(
    buffer_dir: str | Path,
    n: int,
    timeout: float = 3600.0,
    poll_interval: float = 5.0,
) -> list[dict]:
    """Block until n samples are available, then read and delete them."""
    buffer_dir = Path(buffer_dir)
    deadline = time.monotonic() + timeout

    while True:
        files = sorted(buffer_dir.glob("sample_*.jsonl"))
        if len(files) >= n:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Buffer has only {len(files)}/{n} samples after {timeout:.0f}s"
            )
        time.sleep(poll_interval)

    samples = []
    for f in files[:n]:
        try:
            text = f.read_text(encoding="utf-8").strip()
            samples.append(json.loads(text))
            f.unlink()
        except Exception as exc:
            print(f"[buffer] Warning: could not read {f}: {exc}")
    return samples


def peek_samples(buffer_dir: str | Path, n: int) -> list[dict]:
    """Read n samples without deleting them (for inspection/testing)."""
    files = sorted(Path(buffer_dir).glob("sample_*.jsonl"))[:n]
    samples = []
    for f in files:
        try:
            samples.append(json.loads(f.read_text(encoding="utf-8").strip()))
        except Exception:
            pass
    return samples


# ---------------------------------------------------------------------------
# Checkpoint flag protocol
# ---------------------------------------------------------------------------

def write_ckpt_flag(flag_path: str | Path) -> None:
    """Signal that a new checkpoint is ready (learner side)."""
    p = Path(flag_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def check_and_clear_flag(flag_path: str | Path) -> bool:
    """Return True and delete the flag if it exists (daemon side)."""
    p = Path(flag_path)
    if p.exists():
        try:
            p.unlink()
            return True
        except FileNotFoundError:
            pass  # race — another process beat us
    return False
