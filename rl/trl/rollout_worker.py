"""
Multi-turn trajectory generator for the explorer model (Qwen3.5-4B).

Each call to generate_trajectory() makes HTTP requests to the vLLM server
at cfg["rollout_port"] and executes search tool calls via a FAISS searcher.
G independent trajectories per query are produced concurrently via asyncio.

Message format follows the tongyi/Qwen3 chat template:
  <|im_start|>system ... <|im_end|>
  <|im_start|>user  ... <|im_end|>
  <|im_start|>assistant <tool_call>{"name":"search",...}</tool_call> <|im_end|>
  <|im_start|>user <tool_response>...</tool_response> <|im_end|>
  ...
  <|im_start|>assistant <answer>...</answer> <|im_end|>
"""

import asyncio
import json
import re
from datetime import date

import aiohttp
import json5


_EXPLORER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a deep research assistant. Your core function is to conduct "
    "thorough, multi-source investigations into any topic. When you have "
    "gathered sufficient information, enclose the final answer within "
    "<answer></answer> tags.\n\n"
    "# Tools\n\n"
    "You are provided with the search tool:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "search", "description": '
    '"Search the local knowledge base for relevant documents.", '
    '"parameters": {"type": "object", "properties": {"query": {"type": '
    '"string", "description": "The search query."}}, "required": ["query"]}}}\n'
    "</tools>\n\n"
    "For each search tool call, return a JSON object within "
    "<tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": "search", "arguments": {"query": "your query here"}}\n'
    "</tool_call>\n\n"
    "Current date: {today}"
)


def _explorer_system_prompt() -> str:
    return _EXPLORER_SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_trajectory(
    query: str,
    query_id: str,
    searcher,
    cfg: dict,
    session: aiohttp.ClientSession,
) -> dict:
    """Generate one multi-turn explorer trajectory for a single query."""
    port = cfg["rollout_port"]
    model = cfg["explorer_model"]
    max_turns = cfg.get("max_turns", 15)
    max_new_tokens = cfg.get("max_new_tokens", 2048)
    temperature = cfg.get("temperature", 0.6)
    top_p = cfg.get("top_p", 0.95)
    k = cfg.get("k", 5)

    messages = [
        {"role": "system", "content": _explorer_system_prompt()},
        {"role": "user", "content": query},
    ]
    terminated = False

    for _ in range(max_turns):
        content = await _call_vllm(
            session, port, model, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["\n<tool_response>", "<tool_response>"],
        )
        if not content:
            break

        # Trim any partial <tool_response> that leaked past the stop token
        if "<tool_response>" in content:
            content = content[: content.find("<tool_response>")]

        messages.append({"role": "assistant", "content": content.strip()})

        if "<answer>" in content and "</answer>" in content:
            terminated = True
            break

        if "<tool_call>" in content and "</tool_call>" in content:
            raw = content.split("<tool_call>")[1].split("</tool_call>")[0]
            tool_response = _execute_search(raw, searcher, k)
            messages.append({"role": "user", "content": tool_response})

    return {
        "query_id": query_id,
        "query": query,
        "messages": messages,
        "terminated": terminated,
    }


async def generate_group(
    query: str,
    query_id: str,
    searcher,
    cfg: dict,
    G: int | None = None,
) -> list[dict]:
    """Generate G independent trajectories for a query concurrently."""
    G = G or cfg.get("group_size", 4)
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[
            generate_trajectory(query, query_id, searcher, cfg, session)
            for _ in range(G)
        ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _execute_search(raw_tool_call: str, searcher, k: int) -> str:
    """Parse a <tool_call> JSON blob, run the search, return <tool_response>."""
    try:
        call = json5.loads(raw_tool_call)
        search_query = call.get("arguments", {}).get("query", "")
        results = searcher.search(search_query, k)
        body = json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:
        body = f"Error executing search: {exc}"
    return f"<tool_response>\n{body}\n</tool_response>"


def extract_answer(messages: list[dict]) -> str:
    """Extract the <answer> text from the last assistant message, or ''."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""


async def _call_vllm(
    session: aiohttp.ClientSession,
    port: int,
    model: str,
    messages: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    max_retries: int = 5,
) -> str:
    """POST to vLLM chat/completions with exponential backoff."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop:
        payload["stop"] = stop

    for attempt in range(max_retries):
        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"[rollout_worker] vLLM call failed after {max_retries} attempts: {exc}")
    return ""
