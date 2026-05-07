"""
Unit tests for rl/trl/buffer.py.

All tests use a temporary directory so they leave no state behind.
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from rl.trl.buffer import (
    append_sample,
    buffer_size,
    check_and_clear_flag,
    peek_samples,
    pop_samples,
    write_ckpt_flag,
)


# ---------------------------------------------------------------------------
# append_sample / buffer_size
# ---------------------------------------------------------------------------

def test_append_creates_file(tmp_path):
    sample = {"query_id": "Q1", "reward": 1}
    append_sample(tmp_path, sample)
    files = list(tmp_path.glob("sample_*.jsonl"))
    assert len(files) == 1


def test_append_content_roundtrips(tmp_path):
    sample = {"query_id": "Q42", "reward": 0, "messages": [{"role": "user", "content": "hello"}]}
    append_sample(tmp_path, sample)
    f = next(tmp_path.glob("sample_*.jsonl"))
    loaded = json.loads(f.read_text(encoding="utf-8").strip())
    assert loaded == sample


def test_buffer_size_empty(tmp_path):
    assert buffer_size(tmp_path) == 0


def test_buffer_size_after_appends(tmp_path):
    for i in range(5):
        append_sample(tmp_path, {"i": i})
    assert buffer_size(tmp_path) == 5


def test_append_multiple_samples_distinct_files(tmp_path):
    for i in range(10):
        append_sample(tmp_path, {"n": i})
        time.sleep(0.001)  # ensure monotonic_ns differs
    files = list(tmp_path.glob("sample_*.jsonl"))
    assert len(files) == 10


# ---------------------------------------------------------------------------
# pop_samples
# ---------------------------------------------------------------------------

def test_pop_returns_n_samples(tmp_path):
    for i in range(6):
        append_sample(tmp_path, {"i": i})
    samples = pop_samples(tmp_path, 4)
    assert len(samples) == 4


def test_pop_deletes_consumed_files(tmp_path):
    for i in range(5):
        append_sample(tmp_path, {"i": i})
    pop_samples(tmp_path, 3)
    assert buffer_size(tmp_path) == 2


def test_pop_returns_correct_data(tmp_path):
    payloads = [{"query_id": f"Q{i}", "reward": i % 2} for i in range(4)]
    for p in payloads:
        append_sample(tmp_path, p)
    samples = pop_samples(tmp_path, 4)
    # Order should match insertion order (sorted by filename which encodes timestamp)
    assert {s["query_id"] for s in samples} == {f"Q{i}" for i in range(4)}


def test_pop_blocks_then_unblocks(tmp_path):
    """Writer thread appends 3 samples after a delay; reader blocks until available."""
    results = []

    def writer():
        time.sleep(0.3)
        for i in range(3):
            append_sample(tmp_path, {"i": i})

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    samples = pop_samples(tmp_path, 3, timeout=10.0, poll_interval=0.05)
    results.extend(samples)
    t.join()

    assert len(results) == 3


def test_pop_timeout_raises(tmp_path):
    with pytest.raises(TimeoutError):
        pop_samples(tmp_path, 5, timeout=0.2, poll_interval=0.05)


# ---------------------------------------------------------------------------
# peek_samples
# ---------------------------------------------------------------------------

def test_peek_does_not_delete(tmp_path):
    for i in range(3):
        append_sample(tmp_path, {"i": i})
    peek_samples(tmp_path, 2)
    assert buffer_size(tmp_path) == 3


def test_peek_returns_up_to_n(tmp_path):
    for i in range(3):
        append_sample(tmp_path, {"i": i})
    samples = peek_samples(tmp_path, 5)
    assert len(samples) == 3  # only 3 available


# ---------------------------------------------------------------------------
# Checkpoint flag
# ---------------------------------------------------------------------------

def test_write_and_check_flag(tmp_path):
    flag = tmp_path / "ckpt_ready.flag"
    assert not flag.exists()
    write_ckpt_flag(flag)
    assert flag.exists()
    assert check_and_clear_flag(flag) is True
    assert not flag.exists()


def test_check_absent_flag(tmp_path):
    flag = tmp_path / "missing.flag"
    assert check_and_clear_flag(flag) is False


def test_flag_only_triggers_once(tmp_path):
    flag = tmp_path / "ckpt.flag"
    write_ckpt_flag(flag)
    assert check_and_clear_flag(flag) is True
    assert check_and_clear_flag(flag) is False


def test_write_flag_creates_parent_dirs(tmp_path):
    flag = tmp_path / "a" / "b" / "c.flag"
    write_ckpt_flag(flag)
    assert flag.exists()


# ---------------------------------------------------------------------------
# Concurrent writes (basic race-condition check)
# ---------------------------------------------------------------------------

def test_concurrent_appends(tmp_path):
    n_threads = 8
    samples_each = 20

    def worker(tid):
        for i in range(samples_each):
            append_sample(tmp_path, {"tid": tid, "i": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert buffer_size(tmp_path) == n_threads * samples_each
    # Verify every file is valid JSON
    for f in tmp_path.glob("sample_*.jsonl"):
        data = json.loads(f.read_text(encoding="utf-8").strip())
        assert "tid" in data and "i" in data
