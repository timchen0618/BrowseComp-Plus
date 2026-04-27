"""Smoke tests for Tongyi/GLM format_context_for_prompt_* truncators."""

import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TONGYI_TRAJ = _ROOT / "runs/bcp/Qwen3-Embedding-8B/full/tongyi/seed3/run_20260418T031417964456Z.json"
_GLM_TRAJ = _ROOT / "runs/bcp/Qwen3-Embedding-8B/full/glm/seed0/run_20260420T100002858009Z.json"

sys.path.insert(0, str(_ROOT / "src_select_tool_calls"))
from select_useful_tool_calls_tongyi import format_context_for_prompt_tongyi  # noqa: E402
from select_useful_tool_calls_glm import format_context_for_prompt_glm  # noqa: E402


@pytest.mark.skipif(not _TONGYI_TRAJ.is_file(), reason="Tongyi trajectory fixture not in workspace")
def test_tongyi_global_max_chars_truncation():
    with _TONGYI_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_tongyi(
        traj, max_chars=500, reasoning_max_chars=0, tool_output_max_chars=0
    )
    assert out.endswith("... (trajectory truncated)")
    assert len(out) < len(json.dumps(traj.get("original_messages", []), ensure_ascii=False)) + 200


@pytest.mark.skipif(not _TONGYI_TRAJ.is_file(), reason="Tongyi trajectory fixture not in workspace")
def test_tongyi_tool_response_truncation():
    with _TONGYI_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_tongyi(
        traj,
        max_chars=0,
        reasoning_max_chars=0,
        tool_output_max_chars=80,
    )
    m = re.search(
        r"<tool_response>[\s\S]{0,200}\.\.\.</tool_response>", out
    )
    assert m is not None, "expected a shortened <tool_response> block with ... inside"


@pytest.mark.skipif(not _TONGYI_TRAJ.is_file(), reason="Tongyi trajectory fixture not in workspace")
def test_tongyi_redacted_thinking_truncation():
    with _TONGYI_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_tongyi(
        traj,
        max_chars=0,
        reasoning_max_chars=120,
        tool_output_max_chars=0,
    )
    m = re.search(
        r"<think>[\s\S]*?\.\.\.</think>", out
    )
    assert m is not None, "expected truncated redacted_thinking (with ... before closing tag)"


@pytest.mark.skipif(not _GLM_TRAJ.is_file(), reason="GLM trajectory fixture not in workspace")
def test_glm_global_max_chars_truncation():
    with _GLM_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_glm(
        traj, max_chars=500, reasoning_max_chars=0, tool_output_max_chars=0
    )
    assert out.endswith("... (trajectory truncated)")


@pytest.mark.skipif(not _GLM_TRAJ.is_file(), reason="GLM trajectory fixture not in workspace")
def test_glm_tool_content_truncation():
    with _GLM_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_glm(
        traj, max_chars=0, reasoning_max_chars=0, tool_output_max_chars=200
    )
    assert "..." in out
    for line in out.split("\n\n"):
        if '"role": "tool"' in line and '"content":' in line:
            # Truncated tool JSON line should be short-ish (not full retrieval blob)
            assert len(line) < 2000
            break
    else:
        pytest.fail("no tool message found in output")


@pytest.mark.skipif(not _GLM_TRAJ.is_file(), reason="GLM trajectory fixture not in workspace")
def test_glm_redacted_thinking_truncation():
    with _GLM_TRAJ.open(encoding="utf-8") as f:
        traj = json.load(f)
    out = format_context_for_prompt_glm(
        traj, max_chars=0, reasoning_max_chars=100, tool_output_max_chars=0
    )
    m = re.search(
        r"<think>[\s\S]*?\.\.\.</think>", out
    )
    assert m is not None, "expected truncated redacted_thinking in GLM output"
