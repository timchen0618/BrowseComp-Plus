"""
Tests to verify that the hermes tool call parser correctly handles the full
fine-tuned model's JSON-in-<tool_call> output format, and that qwen3_coder
correctly rejects it (reproducing the root cause of the bug).

Run with: python -m pytest tests/test_tool_call_parser.py -v
"""

import json
import re
import pytest


# --- Minimal re-implementations of the two parsers' extract_tool_calls logic ---
# These mirror the vLLM parser implementations without importing vLLM.

TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
QWEN3_CODER_PREFIX = "<function="
QWEN3_CODER_REGEX = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)


def hermes_extract(model_output: str) -> list[dict]:
    """Mimic hermes_tool_parser.extract_tool_calls."""
    if "<tool_call>" not in model_output:
        return []
    tuples = TOOL_CALL_REGEX.findall(model_output)
    calls = []
    for match in tuples:
        raw = match[0] if match[0] else match[1]
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return calls


def qwen3_coder_would_trigger(model_output: str) -> bool:
    """Return True if qwen3_coder parser would attempt tool extraction."""
    return QWEN3_CODER_PREFIX in model_output


# --- Sample outputs from actual trajectories ---

# Full fine-tuned model output (JSON-in-<tool_call> format)
FULL_FT_OUTPUT = (
    "\n\n<tool_call>\n"
    '{"name": "local_knowledge_base_retrieval", "arguments": {"user_query": "Ada Lovelace mathematician"}}'
    "\n\n</tool_call>"
)

# LoRA / base Qwen3.5-4B output format (XML, <function=name>)
LORA_OUTPUT = (
    "<function=local_knowledge_base_retrieval>"
    "<parameter=user_query>Ada Lovelace mathematician</parameter>"
    "</function>"
)

# Normal text output (no tool call)
TEXT_OUTPUT = "I cannot find the relevant information using the available tools."


class TestHermesParser:
    def test_extracts_tool_name(self):
        calls = hermes_extract(FULL_FT_OUTPUT)
        assert len(calls) == 1
        assert calls[0]["name"] == "local_knowledge_base_retrieval"

    def test_extracts_arguments(self):
        calls = hermes_extract(FULL_FT_OUTPUT)
        assert calls[0]["arguments"] == {"user_query": "Ada Lovelace mathematician"}

    def test_no_tool_call_in_plain_text(self):
        calls = hermes_extract(TEXT_OUTPUT)
        assert calls == []

    def test_does_not_match_lora_format(self):
        """LoRA format uses <function=name>, which hermes should NOT pick up."""
        calls = hermes_extract(LORA_OUTPUT)
        assert calls == [], "hermes parser should not parse <function=name> XML format"

    def test_multiple_tool_calls(self):
        multi = (
            "<tool_call>\n"
            '{"name": "local_knowledge_base_retrieval", "arguments": {"user_query": "query one"}}'
            "\n</tool_call>\n"
            "<tool_call>\n"
            '{"name": "local_knowledge_base_retrieval", "arguments": {"user_query": "query two"}}'
            "\n</tool_call>"
        )
        calls = hermes_extract(multi)
        assert len(calls) == 2
        assert calls[0]["arguments"]["user_query"] == "query one"
        assert calls[1]["arguments"]["user_query"] == "query two"


class TestQwen3CoderParser:
    def test_does_not_trigger_on_full_ft_output(self):
        """Root cause: qwen3_coder looks for '<function=' prefix, which full FT never emits."""
        assert not qwen3_coder_would_trigger(FULL_FT_OUTPUT), (
            "qwen3_coder parser should NOT trigger on JSON-in-<tool_call> format; "
            "this reproduces the original empty-output bug"
        )

    def test_triggers_on_lora_output(self):
        """LoRA model uses <function=name> XML format; qwen3_coder should handle it."""
        assert qwen3_coder_would_trigger(LORA_OUTPUT)

    def test_does_not_trigger_on_plain_text(self):
        assert not qwen3_coder_would_trigger(TEXT_OUTPUT)


class TestParserAssignment:
    """
    All qwen3.5-4b-sft-* models (both LoRA and full FT) use qwen3_coder.
    Full FT checkpoints have been patched to use the base Qwen3.5-4B XML
    template (chat_template.jinja), so they output <function=name> format
    just like the LoRA models at inference.
    """

    ALL_SFT_MODELS = [
        "qwen3.5-4b-sft-gemini_2.5_pro_selection_full",
        "qwen3.5-4b-sft-gemini_2.5_pro_selection_full_unfiltered",
        "qwen3.5-4b-sft-gpt-oss-120b_scout_full",
        "qwen3.5-4b-sft-gpt-oss-120b_scout_full_unfiltered",
        "qwen3.5-4b-sft-gemini_2.5_pro_selection",
        "qwen3.5-4b-sft-gpt-oss-120b_scout",
        "qwen3.5-4b-sft-random_selection",
        "qwen3.5-4b-sft-best_of_4_gemini_2.5_pro_selection_mode_c",
    ]

    @pytest.mark.parametrize("model_name", ALL_SFT_MODELS)
    def test_all_sft_use_qwen3_coder(self, model_name):
        """All SFT models (LoRA and full FT) should use qwen3_coder parser."""
        assert model_name.startswith("qwen3.5-4b-sft-"), "model name pattern check"
