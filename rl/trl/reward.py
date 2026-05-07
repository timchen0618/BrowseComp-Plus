"""
Reward functions for the GRPO training loop.

call_main_agent():
  Formats the explorer trajectory with QUERY_TEMPLATE_GIVEN_TRAJECTORY, then
  runs a full tool-calling loop with GPT-OSS-120B on GPU 2 (port 8010) until
  it emits "Exact Answer:" or exhausts max_turns.  Returns the final answer.

call_judge():
  Sends (question, response, correct_answer) to Qwen3-32B on GPU 3 (port 8011,
  which is in judge mode when this is called).  Returns 1 if correct, 0 otherwise.
  Uses the same GRADER_TEMPLATE as scripts_evaluation/evaluate_run.py.
"""

import asyncio
import json
import re

import aiohttp


# ---------------------------------------------------------------------------
# Templates (verbatim from search_agent/prompts.py and evaluate_run.py)
# ---------------------------------------------------------------------------

QUERY_TEMPLATE_GIVEN_TRAJECTORY = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

You are also given a previous trajectory of the agent's search and tool calls for the question, and the outputs of the tool calls. The trajectory is enclosed within <trajectory></trajectory> tags. Use the information in the trajectory to search more efficiently.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}

<trajectory>
{trajectory}
</trajectory>""".strip()


GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.""".strip()

# OpenAI-style function definition for the main agent's search tool
_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "local_knowledge_base_retrieval",
        "description": (
            "Search the local knowledge base for documents relevant to a query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "Query to search the knowledge base.",
                }
            },
            "required": ["user_query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Main agent call
# ---------------------------------------------------------------------------

async def call_main_agent(
    trajectory_messages: list[dict],
    query_id: str,
    query: str,
    cfg: dict,
    session: aiohttp.ClientSession,
    searcher=None,
) -> str:
    """
    Run GPT-OSS-120B with the explorer trajectory as context.

    Returns the extracted "Exact Answer" string, or the last response if the
    format is not found.
    """
    port = cfg["main_agent_port"]
    model = cfg["main_agent_model"]
    max_turns = cfg.get("max_turns", 50)
    max_new_tokens = cfg.get("max_new_tokens", 2048)
    k = cfg.get("k", 5)

    traj_text = _format_traj_for_main_agent(trajectory_messages)
    user_content = QUERY_TEMPLATE_GIVEN_TRAJECTORY.format(
        Question=query,
        trajectory=traj_text,
    )

    messages: list[dict] = [{"role": "user", "content": user_content}]
    tools = [_SEARCH_TOOL_DEF] if searcher is not None else None

    for _ in range(max_turns):
        content, tool_calls = await _chat_completions(
            session, port, model, messages, tools, max_new_tokens
        )
        if not content and not tool_calls:
            break

        assistant_msg: dict = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # Check for final answer
        if content and "Exact Answer:" in content:
            return _extract_exact_answer(content)

        # Execute tool calls if searcher is available
        if tool_calls and searcher is not None:
            for tc in tool_calls:
                fn_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(fn_args)
                    results = searcher.search(args.get("user_query", ""), k)
                    result_text = json.dumps(results, ensure_ascii=False, indent=2)
                except Exception as exc:
                    result_text = f"Search error: {exc}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_text,
                })
        elif not tool_calls:
            # No tool calls and no "Exact Answer:" — return whatever we have
            return content or ""

    return ""


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

async def call_judge(
    query: str,
    answer: str,
    ground_truth: str,
    cfg: dict,
    session: aiohttp.ClientSession,
) -> int:
    """Score an answer with the Qwen3-32B judge.  Returns 1 if correct, 0 otherwise."""
    port = cfg["rollout_port"]   # GPU 3 is in judge mode when this is called
    model = cfg["judge_model"]

    prompt = GRADER_TEMPLATE.format(
        question=query,
        response=answer,
        correct_answer=ground_truth,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(5):
        try:
            async with session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return parse_judge_response(content)
        except Exception as exc:
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"[reward] Judge call failed: {exc}")
    return 0


# ---------------------------------------------------------------------------
# Parsing helpers (public so tests can exercise them independently)
# ---------------------------------------------------------------------------

def parse_judge_response(text: str) -> int:
    """Extract 'correct: yes/no' from judge output.  Returns 1 or 0."""
    for pattern in (
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return 1 if m.group(1).lower() == "yes" else 0
    return 0


def _extract_exact_answer(text: str) -> str:
    """Pull the value after 'Exact Answer:' from an OSS-style response."""
    m = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _format_traj_for_main_agent(messages: list[dict]) -> str:
    """Render explorer messages as a human-readable block for the trajectory template."""
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        lines.append(f"[{role.upper()}]\n{content}")
    return "\n\n---\n\n".join(lines)


async def _chat_completions(
    session: aiohttp.ClientSession,
    port: int,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_new_tokens: int = 2048,
    max_retries: int = 5,
) -> tuple[str, list[dict]]:
    """Call /v1/chat/completions; return (content, tool_calls)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
    }
    if tools:
        payload["tools"] = tools

    for attempt in range(max_retries):
        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=1200),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                tool_calls = choice.get("tool_calls") or []
                return content, tool_calls
        except Exception as exc:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"[reward] main_agent call failed: {exc}")
    return "", []
