"""
Reward functions for the GRPO training loop.

call_main_agent():
  Formats the explorer trajectory with QUERY_TEMPLATE_GIVEN_TRAJECTORY, then
  runs a full tool-calling loop with GPT-OSS-120B on GPU 2 (port 8010) via the
  OpenAI Responses API (same as oss_client.py) until the model emits a final
  text message or exhausts max_turns.  Returns the extracted "Exact Answer".

call_judge():
  Sends (question, response, correct_answer) to Qwen3-32B on GPU 3 (port 8011,
  which is in judge mode when this is called).  Returns 1 if correct, 0 otherwise.
  Uses the same GRADER_TEMPLATE as scripts_evaluation/evaluate_run.py.
"""

import asyncio
import json
import re
import string

import aiohttp
import openai


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

# Responses API tool definition (matches oss_client.py SearchToolHandler.get_tool_definitions)
_SEARCH_TOOL_DEF = {
    "type": "function",
    "name": "local_knowledge_base_retrieval",
    "description": "Search the local knowledge base for documents relevant to a query.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_query": {
                "type": "string",
                "description": "Query to search the knowledge base.",
            }
        },
        "required": ["user_query"],
        "additionalProperties": False,
    },
    "strict": True,
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

    Uses the OpenAI Responses API, mirroring run_conversation_with_tools in
    oss_client.py.  The loop continues until the model emits a final text
    message (no pending tool calls) or max_turns is exhausted.

    Returns the extracted "Exact Answer" string, or "" on failure.
    """
    port = cfg["main_agent_port"]
    model = cfg["main_agent_model"]
    max_iterations = cfg.get("main_agent_max_turns", cfg.get("max_turns", 10))
    max_new_tokens = cfg.get("max_new_tokens", 2048)
    k = cfg.get("k", 5)

    traj_text = _format_traj_for_main_agent(trajectory_messages)
    user_content = QUERY_TEMPLATE_GIVEN_TRAJECTORY.format(
        Question=query,
        trajectory=traj_text,
    )

    messages: list[dict] = [{"role": "user", "content": user_content}]
    tools = [_SEARCH_TOOL_DEF] if searcher is not None else []

    aclient = openai.AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="EMPTY",
    )

    request = {
        "model": model,
        "max_output_tokens": max_new_tokens,
        "tools": tools,
        "truncation": "auto",
    }

    iteration = 1
    while iteration <= max_iterations:
        try:
            response = await aclient.responses.create(input=messages, **request)
        except Exception as exc:
            print(f"[reward] main_agent call failed (iter {iteration}): {exc}")
            iteration += 1
            continue

        output = response.model_dump(mode="python")["output"]
        messages.extend(output)

        # Pure reasoning turn — don't count against iteration budget
        if output and output[-1]["type"] == "reasoning":
            messages.pop()
            continue

        if not output:
            iteration += 1
            continue

        function_calls = [item for item in output if item["type"] == "function_call"]
        last_item = output[-1]

        # Done: no tool calls, or model emitted a final text message
        if (not function_calls) or (
            "content" in last_item
            and last_item["content"][0]["type"] == "output_text"
        ):
            for item in reversed(output):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            return _extract_exact_answer(part["text"])
            return ""

        # Execute tool calls and append results
        snippet_max_chars = cfg.get("snippet_max_chars", 2000)
        for tc in function_calls:
            try:
                args = json.loads(tc["arguments"])
                results = searcher.search(args.get("user_query", ""), k)
                for r in results:
                    if "text" in r and len(r["text"]) > snippet_max_chars:
                        r["text"] = r["text"][:snippet_max_chars] + " …"
                result_text = json.dumps(results, ensure_ascii=False, indent=2)
            except Exception as exc:
                result_text = f"Search error: {exc}"
            messages.append({
                "type": "function_call_output",
                "call_id": tc["call_id"],
                "output": result_text,
            })

        iteration += 1

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
# Reward dispatch
# ---------------------------------------------------------------------------

async def score_answer(
    query: str,
    answer: str,
    ground_truth: str,
    cfg: dict,
    session: aiohttp.ClientSession,
) -> int:
    """Dispatch to the reward function named by cfg['reward_function'].

    'llm_judge' — Qwen3-32B LLM judge (requires judge server to be running).
    'sub_em'    — substring exact match: reward=1 if normalized ground truth
                  is contained in the normalized answer.
    """
    mode = cfg.get("reward_function", "llm_judge")
    if mode == "sub_em":
        return sub_em_reward(answer, ground_truth)
    return await call_judge(query, answer, ground_truth, cfg, session)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def sub_em_reward(answer: str, ground_truth: str) -> int:
    """Return 1 if normalized ground_truth is a substring of normalized answer."""
    if not ground_truth:
        return 0
    return 1 if normalize_text(ground_truth) in normalize_text(answer) else 0


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


