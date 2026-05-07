"""
Unit tests for rl/trl/rollout_worker.py.

Tests cover pure-Python helpers and the search execution logic.
The vLLM HTTP call (_call_vllm) is mocked to avoid a live server.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from rl.trl.rollout_worker import (
    EXPLORER_SYSTEM_PROMPT,
    _execute_search,
    extract_answer,
)


# ---------------------------------------------------------------------------
# EXPLORER_SYSTEM_PROMPT sanity checks
# ---------------------------------------------------------------------------

def test_system_prompt_contains_tool_call():
    assert "<tool_call>" in EXPLORER_SYSTEM_PROMPT


def test_system_prompt_contains_answer_tags():
    assert "<answer>" in EXPLORER_SYSTEM_PROMPT


def test_system_prompt_contains_search_tool():
    assert '"name": "search"' in EXPLORER_SYSTEM_PROMPT


def test_system_prompt_has_date():
    # Should contain "Current date:" followed by an ISO date
    assert "Current date:" in EXPLORER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# _execute_search
# ---------------------------------------------------------------------------

def _make_searcher(results=None):
    searcher = MagicMock()
    searcher.search.return_value = results or [
        {"docid": "d1", "score": 0.9, "text": "Paris is the capital of France."},
        {"docid": "d2", "score": 0.8, "text": "France is a country in Europe."},
    ]
    return searcher


def test_execute_search_returns_tool_response_tags():
    raw = '{"name": "search", "arguments": {"query": "capital of France"}}'
    searcher = _make_searcher()
    result = _execute_search(raw, searcher, k=5)
    assert result.startswith("<tool_response>")
    assert result.endswith("</tool_response>")


def test_execute_search_calls_searcher(mocker=None):
    raw = '{"name": "search", "arguments": {"query": "who wrote hamlet"}}'
    searcher = _make_searcher()
    _execute_search(raw, searcher, k=3)
    searcher.search.assert_called_once_with("who wrote hamlet", 3)


def test_execute_search_result_is_json():
    raw = '{"name": "search", "arguments": {"query": "test"}}'
    searcher = _make_searcher()
    result = _execute_search(raw, searcher, k=5)
    inner = result.replace("<tool_response>", "").replace("</tool_response>", "").strip()
    parsed = json.loads(inner)
    assert isinstance(parsed, list)
    assert parsed[0]["docid"] == "d1"


def test_execute_search_invalid_json_returns_error():
    raw = "not valid json !!!"
    searcher = _make_searcher()
    result = _execute_search(raw, searcher, k=5)
    assert "Error" in result


def test_execute_search_searcher_raises_returns_error():
    raw = '{"name": "search", "arguments": {"query": "boom"}}'
    searcher = MagicMock()
    searcher.search.side_effect = RuntimeError("index not found")
    result = _execute_search(raw, searcher, k=5)
    assert "Error" in result


def test_execute_search_missing_query_key():
    # arguments dict has a different key — should handle gracefully
    raw = '{"name": "search", "arguments": {"q": "something"}}'
    searcher = _make_searcher()
    _execute_search(raw, searcher, k=5)
    # Should call searcher with empty string for missing key
    searcher.search.assert_called_once_with("", 5)


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

def test_extract_answer_present():
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>Easy</think>\n<answer>4</answer>"},
    ]
    assert extract_answer(messages) == "4"


def test_extract_answer_multiline():
    messages = [
        {"role": "assistant", "content": "<answer>\nParis\n</answer>"},
    ]
    assert extract_answer(messages) == "Paris"


def test_extract_answer_missing():
    messages = [
        {"role": "assistant", "content": "I don't know."},
    ]
    assert extract_answer(messages) == ""


def test_extract_answer_uses_last_assistant():
    messages = [
        {"role": "assistant", "content": "<answer>wrong</answer>"},
        {"role": "user", "content": "<tool_response>data</tool_response>"},
        {"role": "assistant", "content": "<answer>correct</answer>"},
    ]
    assert extract_answer(messages) == "correct"


def test_extract_answer_empty_messages():
    assert extract_answer([]) == ""


def test_extract_answer_non_assistant_only():
    messages = [
        {"role": "user", "content": "<answer>trick</answer>"},
    ]
    # extract_answer only looks at assistant messages
    assert extract_answer(messages) == ""


# ---------------------------------------------------------------------------
# generate_trajectory (mocked vLLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_trajectory_terminates_on_answer():
    """generate_trajectory stops as soon as an <answer> tag appears."""
    import aiohttp
    from rl.trl.rollout_worker import generate_trajectory

    cfg = {
        "rollout_port": 9999,
        "explorer_model": "test-model",
        "max_turns": 10,
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "k": 3,
    }
    searcher = _make_searcher()

    # Mock _call_vllm to return a direct answer on the first call
    with patch("rl.trl.rollout_worker._call_vllm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "<think>I know</think>\n<answer>Paris</answer>"

        async with aiohttp.ClientSession() as session:
            result = await generate_trajectory("Capital of France?", "Q1", searcher, cfg, session)

    assert result["terminated"] is True
    assert result["query_id"] == "Q1"
    assert extract_answer(result["messages"]) == "Paris"
    assert mock_call.call_count == 1  # stopped after first call


@pytest.mark.asyncio
async def test_generate_trajectory_executes_tool_call():
    """generate_trajectory calls the searcher when a tool_call is in the response."""
    import aiohttp
    from rl.trl.rollout_worker import generate_trajectory

    cfg = {
        "rollout_port": 9999,
        "explorer_model": "test-model",
        "max_turns": 5,
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "k": 3,
    }
    searcher = _make_searcher()

    _responses = [
        '<tool_call>{"name": "search", "arguments": {"query": "France capital"}}</tool_call>',
        "<answer>Paris</answer>",
    ]
    _iter = iter(_responses)

    async def _side_effect(*a, **kw):
        return next(_iter)

    with patch("rl.trl.rollout_worker._call_vllm", side_effect=_side_effect):
        async with aiohttp.ClientSession() as session:
            result = await generate_trajectory("Capital of France?", "Q1", searcher, cfg, session)

    searcher.search.assert_called_once()


@pytest.mark.asyncio
async def test_generate_trajectory_stops_at_max_turns():
    """generate_trajectory gives up after max_turns even without an answer."""
    import aiohttp
    from rl.trl.rollout_worker import generate_trajectory

    cfg = {
        "rollout_port": 9999,
        "explorer_model": "test-model",
        "max_turns": 3,
        "max_new_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "k": 3,
    }
    searcher = _make_searcher()

    with patch("rl.trl.rollout_worker._call_vllm", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "I am still thinking ..."

        async with aiohttp.ClientSession() as session:
            result = await generate_trajectory("Hard question?", "Q2", searcher, cfg, session)

    assert result["terminated"] is False
    assert mock_call.call_count == 3
