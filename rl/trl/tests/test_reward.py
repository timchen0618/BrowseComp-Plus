"""
Unit tests for rl/trl/reward.py.

Tests cover the pure-Python helpers that do not require a live vLLM server.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from rl.trl.reward import (
    _extract_exact_answer,
    _format_traj_for_main_agent,
    parse_judge_response,
)


# ---------------------------------------------------------------------------
# parse_judge_response
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def test_plain_yes(self):
        assert parse_judge_response("correct: yes") == 1

    def test_plain_no(self):
        assert parse_judge_response("correct: no") == 0

    def test_bold_colon_yes(self):
        text = "extracted_final_answer: Paris\n**correct:** yes\nconfidence: 95%"
        assert parse_judge_response(text) == 1

    def test_bold_colon_no(self):
        text = "extracted_final_answer: Berlin\n**correct:** no\nconfidence: 40%"
        assert parse_judge_response(text) == 0

    def test_bold_wrapped_yes(self):
        text = "**correct**: yes"
        assert parse_judge_response(text) == 1

    def test_case_insensitive_yes(self):
        assert parse_judge_response("correct: YES") == 1

    def test_case_insensitive_no(self):
        assert parse_judge_response("CORRECT: No") == 0

    def test_missing_correct_field_returns_0(self):
        assert parse_judge_response("extracted_final_answer: Unknown\nreasoning: unclear") == 0

    def test_empty_string_returns_0(self):
        assert parse_judge_response("") == 0

    def test_multiline_real_judge_output(self):
        text = (
            "extracted_final_answer: Claude Shannon\n\n"
            "[correct_answer]: Claude Shannon\n\n"
            "reasoning: The answers are identical.\n\n"
            "correct: yes\n\n"
            "confidence: 99%"
        )
        assert parse_judge_response(text) == 1

    def test_multiline_judge_incorrect(self):
        text = (
            "extracted_final_answer: Alan Turing\n\n"
            "[correct_answer]: Claude Shannon\n\n"
            "reasoning: Different people.\n\n"
            "correct: no\n\n"
            "confidence: 95%"
        )
        assert parse_judge_response(text) == 0

    def test_correct_appears_in_reasoning_does_not_confuse(self):
        # 'correct' appears mid-sentence in reasoning but the final field says 'no'
        text = (
            "reasoning: The answer is correct in spirit but wrong in name.\n\n"
            "correct: no\n\n"
            "confidence: 70%"
        )
        assert parse_judge_response(text) == 0


# ---------------------------------------------------------------------------
# _extract_exact_answer
# ---------------------------------------------------------------------------

class TestExtractExactAnswer:
    def test_basic(self):
        text = "Explanation: Some reasoning.\nExact Answer: Paris\nConfidence: 90%"
        assert _extract_exact_answer(text) == "Paris"

    def test_case_insensitive(self):
        text = "exact answer: 42"
        assert _extract_exact_answer(text) == "42"

    def test_no_exact_answer_returns_stripped(self):
        text = "  some plain text  "
        assert _extract_exact_answer(text) == "some plain text"

    def test_multiline_stops_at_newline(self):
        text = "Exact Answer: Marie Curie\nConfidence: 80%"
        assert _extract_exact_answer(text) == "Marie Curie"

    def test_empty_string(self):
        assert _extract_exact_answer("") == ""


# ---------------------------------------------------------------------------
# _format_traj_for_main_agent
# ---------------------------------------------------------------------------

class TestFormatTrajForMainAgent:
    def _make_messages(self):
        return [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Who invented the telephone?"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "user", "content": "<tool_response>...</tool_response>"},
            {"role": "assistant", "content": "<answer>Alexander Graham Bell</answer>"},
        ]

    def test_system_messages_excluded(self):
        messages = self._make_messages()
        result = _format_traj_for_main_agent(messages)
        assert "You are a research assistant" not in result

    def test_user_and_assistant_included(self):
        messages = self._make_messages()
        result = _format_traj_for_main_agent(messages)
        assert "[USER]" in result
        assert "[ASSISTANT]" in result

    def test_roles_are_uppercased(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _format_traj_for_main_agent(messages)
        assert "[USER]" in result

    def test_separator_present(self):
        messages = self._make_messages()
        result = _format_traj_for_main_agent(messages)
        assert "---" in result

    def test_empty_messages(self):
        assert _format_traj_for_main_agent([]) == ""

    def test_only_system_message(self):
        messages = [{"role": "system", "content": "system only"}]
        result = _format_traj_for_main_agent(messages)
        assert result == ""
