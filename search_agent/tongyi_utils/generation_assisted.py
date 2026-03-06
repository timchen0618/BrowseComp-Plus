"""Generation-assisted retrieval helpers for Tongyi search tool."""

from dataclasses import dataclass
import random
import re
import time
from typing import Any, Dict, Optional

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


GENERATION_SYSTEM_PROMPT = (
    "You are assisting an autonomous search agent mid-trajectory. "
    "Your output is used only to improve retrieval for the current search query.\n"
    "Provide compact, high-value cues from parametric knowledge: relevant facts, "
    "entity names, document or source titles, dates, aliases, and key relations.\n"
    "Focus on the current search query — do not try to guess the final answer. "
    "Only include cues you are confident about. "
    "Wrong cues actively harm retrieval and are worse than saying nothing.\n"
    "If you are not confident about any specific facts related to this query, "
    "output exactly: I don't know\n"
    "Do not guess. Do not fabricate."
)


def build_generation_user_prompt(original_question: str, subquery: str) -> str:
    return (
        f"Original question: {original_question.strip()}\n\n"
        "Situation:\n"
        "- The agent is deciding what evidence to retrieve next.\n"
        "- Your note is a retrieval expansion signal, not final evidence.\n\n"
        "Output requirements:\n"
        "- If you don't know, output exactly: I don't know\n"
        "- Otherwise, list relevant facts, entity names, document/source titles, "
        "aliases, or relations that could help retrieve the right evidence\n"
        "- Be concise — only include cues you are confident about\n\n"
        f"Current search query: {subquery.strip()}"
    )


def build_generation_user_prompt_with_context(
    original_question: str,
    subquery: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    ctx = context or {}
    step_idx = ctx.get("step_index")
    previous_subqueries = ctx.get("previous_subqueries") or []
    previous_observations = ctx.get("previous_observations") or []
    latest_reasoning = (ctx.get("latest_reasoning") or "").strip()

    lines = [
        f"Original question: {original_question.strip()}",
    ]

    lines.append(
        "\nSituation:\n"
        "- The agent is deciding what evidence to retrieve next.\n"
        "- Your note is a retrieval expansion signal, not final evidence."
    )

    if step_idx is not None:
        lines.append(f"- Current trajectory step: {step_idx}")

    if previous_subqueries:
        lines.append("\nRecent previous subqueries:")
        for i, q in enumerate(previous_subqueries, 1):
            lines.append(f"{i}. {str(q).strip()}")

    if previous_observations:
        lines.append("\nRecent observations from previous retrieval:")
        for i, obs in enumerate(previous_observations, 1):
            lines.append(f"{i}. {str(obs).strip()}")

    if latest_reasoning:
        lines.append("\nAgent reasoning right before this search:")
        lines.append(latest_reasoning)

    lines.append(
        "\nOutput requirements:\n"
        "- If you don't know, output exactly: I don't know\n"
        "- Otherwise, list relevant facts, entity names, document/source titles, "
        "aliases, or relations that could help retrieve the right evidence\n"
        "- Be concise — only include cues you are confident about"
    )

    lines.append(f"\nCurrent search query: {subquery.strip()}")
    return "\n".join(lines)


def strip_thinking_blocks(text: str) -> str:
    # Strip closed <think>...</think> blocks
    no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip unclosed <think> blocks (model hit max_tokens mid-think)
    no_think = re.sub(r"<think>.*", "", no_think, flags=re.DOTALL | re.IGNORECASE)
    no_think = re.sub(r"\n{3,}", "\n\n", no_think).strip()
    return no_think


def maybe_extract_answer_block(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    return match.group(1).strip()


@dataclass
class GenerationAssistedConfig:
    enabled: bool = False
    port: int = 6008
    model: Optional[str] = None
    max_tokens: int = 384
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_s: float = 180.0
    max_retries: int = 3
    allow_thinking: bool = False
    strip_thinking: bool = True


class GenerationAssistedRetriever:
    """Calls a base model to synthesize parametric evidence per search query."""

    def __init__(self, cfg: GenerationAssistedConfig):
        self.cfg = cfg
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://127.0.0.1:{cfg.port}/v1",
            timeout=cfg.timeout_s,
        )

    def generate_document(
        self,
        original_question: str,
        subquery: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.cfg.enabled or not self.cfg.model:
            return None
        if not original_question or not subquery:
            return None

        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_generation_user_prompt_with_context(
                    original_question, subquery, context=context
                ),
            },
        ]

        for attempt in range(self.cfg.max_retries):
            try:
                request_kwargs = {
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": self.cfg.allow_thinking
                        }
                    }
                }
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens,
                    logprobs=False,
                    **request_kwargs,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    if self.cfg.strip_thinking:
                        content = strip_thinking_blocks(content)
                    content = maybe_extract_answer_block(content)
                    return content
            except (APIConnectionError, APIError, APITimeoutError):
                pass
            except Exception:
                pass

            if attempt < self.cfg.max_retries - 1:
                sleep_time = min(2**attempt + random.uniform(0.0, 0.5), 5.0)
                time.sleep(sleep_time)

        return None
