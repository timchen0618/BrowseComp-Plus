"""
Pairwise plan comparison using an LLM judge served by an external vLLM server.

Reads two plan JSONL files, matches entries by query_id, renders a comparison
prompt from a user-supplied template, and writes judge verdicts to output JSONL.

Usage:
    python eval_plan_quality/compare_plans.py \
        --plan-file-1 gemini_2.5_pro_plans.jsonl \
        --plan-file-2 gpt-oss-120b_plans.jsonl \
        --prompt-template eval_plan_quality/prompts/naive_prompt.txt \
        --output-file eval_plan_quality/comparison_results.jsonl \
        --model-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen3-32B \
        --num-threads 8
"""

import argparse
import json
import os
import re
import random
import sys
import time
import threading
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import openai
from tqdm import tqdm

T = TypeVar("T")

DEFAULT_PROMPT_TEMPLATE = "eval_plan_quality/prompts/naive_prompt.txt"
DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Retry helpers (mirrors vllm_planner.py)
# ---------------------------------------------------------------------------

def _call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as e:
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"request exceeded {timeout_s:.1f}s") from e
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _retry_with_deadline(
    fn: Callable[[], T],
    max_retries: int = 3,
    per_attempt_timeout_s: float = 180.0,
    total_deadline_s: float = 600.0,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 20.0,
) -> T:
    start = time.time()
    last_err: Optional[BaseException] = None
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        if time.time() - start > total_deadline_s:
            raise TimeoutError(
                f"total deadline exceeded ({total_deadline_s:.1f}s)"
            ) from last_err
        try:
            return _call_with_timeout(fn, per_attempt_timeout_s)
        except Exception as e:
            log(f"Attempt {attempt + 1}/{attempts} failed: {e!r}")
            last_err = e
            if attempt >= attempts - 1:
                raise
            backoff = min(max_backoff_s, base_backoff_s * (2 ** attempt))
            backoff *= 0.5 + random.random()
            time.sleep(backoff)

    raise RuntimeError("retry loop fell through") from last_err


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_plans(path: Path) -> Dict[str, Dict[str, str]]:
    """Load plan JSONL keyed by query_id.

    Expected format per line: {"query_id": "...", "query_text": "...", "output": "..."}
    """
    plans: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj["query_id"])
            plans[qid] = {
                "query_text": obj.get("query_text", ""),
                "plan": obj.get("output", ""),
            }
    return plans


@dataclass
class PromptTemplate:
    """Parsed prompt template with separate system and user sections."""
    system: str
    user: str  # contains {question}, {plan_1}, {plan_2} placeholders

    def render(self, **kwargs) -> List[Dict[str, str]]:
        """Format the user template with the given kwargs and return a messages list."""
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user.format(**kwargs)})
        return messages


def load_prompt_template(path: Path) -> PromptTemplate:
    """Parse a prompt template file with <start_system_prompt>/<end_system_prompt>
    and <start_user_prompt>/<end_user_prompt> tags.

    The user section may contain {question}, {plan_1}, {plan_2} placeholders.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt template at {path} is empty")

    sys_match = re.search(
        r"<start_system_prompt>(.*?)<end_system_prompt>", text, re.DOTALL
    )
    usr_match = re.search(
        r"<start_user_prompt>(.*?)<end_user_prompt>", text, re.DOTALL
    )

    system_text = sys_match.group(1).strip() if sys_match else ""
    user_text = usr_match.group(1).strip() if usr_match else ""

    if not user_text:
        raise ValueError(
            f"Prompt template at {path} has no <start_user_prompt>...</end_user_prompt> section"
        )

    return PromptTemplate(system=system_text, user=user_text)


def load_completed_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("query_id")
                if qid is not None:
                    ids.add(str(qid))
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# VLLMJudge — thin OpenAI-compatible client for an external vLLM server
# ---------------------------------------------------------------------------

class VLLMJudge:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_MODEL_URL,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=600.0,
        )

    def _judge_single(self, messages: List[Dict[str, str]]) -> str:
        """Send one judgment request with retries."""

        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""

        return _retry_with_deadline(_call, max_retries=self.max_retries)

    def judge_and_write(
        self,
        items: List[Dict[str, Any]],
        output_file: Path,
        num_threads: int = 4,
    ) -> None:
        """Run pairwise comparisons concurrently and write each result to file when done.

        Each item in *items* must have keys:
            query_id, question, plan_1, plan_2, messages.
        Each completed result is appended to output_file immediately.
        """
        file_lock = threading.Lock()

        def _process(item: Dict[str, Any]) -> None:
            qid = item["query_id"]
            try:
                judge_output = self._judge_single(item["messages"])
            except Exception as e:
                log(f"[{qid}] judge failed: {e!r}")
                judge_output = ""

            result = {
                "query_id": qid,
                "question": item["question"],
                "plan_1": item["plan_1"],
                "plan_2": item["plan_2"],
                "judge_messages": item["messages"],
                "judge_output": judge_output,
            }
            with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if num_threads <= 1:
            for item in tqdm(items, desc="Judging"):
                _process(item)
        else:
            with (
                concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool,
                tqdm(total=len(items), desc="Judging") as pbar,
            ):
                futures = [pool.submit(_process, it) for it in items]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
                    pbar.update(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pairs of plans using an LLM judge via an external vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--plan-file-1", required=True, type=Path,
        help="JSONL file with plans from the first model/method",
    )
    parser.add_argument(
        "--plan-file-2", required=True, type=Path,
        help="JSONL file with plans from the second model/method",
    )
    parser.add_argument(
        "--prompt-template", type=Path, default=Path(DEFAULT_PROMPT_TEMPLATE),
        help="Prompt template with <start_system_prompt>/<end_system_prompt> and "
             "<start_user_prompt>/<end_user_prompt> tags. "
             "User section must contain {question}, {plan_1}, {plan_2} placeholders.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=Path("eval_plan_quality/comparison_results.jsonl"),
        help="Output JSONL for judge verdicts",
    )

    # vLLM server
    parser.add_argument(
        "--model-url", default=DEFAULT_MODEL_URL,
        help="Base URL for the external vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="vLLM server port (shorthand: sets --model-url to http://127.0.0.1:{port}/v1)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--api-key", default="EMPTY",
        help="API key for the vLLM server",
    )

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)

    # Concurrency
    parser.add_argument(
        "--num-threads", type=int, default=4,
        help="Number of concurrent requests to the vLLM server",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per request on transient failures",
    )

    args = parser.parse_args()

    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    log(f"Loading plans from {args.plan_file_1}")
    plans_1 = load_plans(args.plan_file_1)
    log(f"  → {len(plans_1)} entries")

    log(f"Loading plans from {args.plan_file_2}")
    plans_2 = load_plans(args.plan_file_2)
    log(f"  → {len(plans_2)} entries")

    log(f"Loading prompt template from {args.prompt_template}")
    template: PromptTemplate = load_prompt_template(args.prompt_template)
    log(f"  system prompt: {len(template.system)} chars | user template: {len(template.user)} chars")

    common_ids = sorted(set(plans_1.keys()) & set(plans_2.keys()))
    only_1 = set(plans_1.keys()) - set(plans_2.keys())
    only_2 = set(plans_2.keys()) - set(plans_1.keys())

    if only_1:
        log(f"  {len(only_1)} query_ids only in plan-file-1 (skipped)")
    if only_2:
        log(f"  {len(only_2)} query_ids only in plan-file-2 (skipped)")
    log(f"  {len(common_ids)} query_ids in common")

    if not common_ids:
        print("No overlapping query_ids between the two plan files.", file=sys.stderr)
        sys.exit(1)

    # Resume support
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids(args.output_file)
    remaining_ids = [qid for qid in common_ids if qid not in completed_ids]
    log(f"Total: {len(common_ids)} | Done: {len(common_ids) - len(remaining_ids)} | Remaining: {len(remaining_ids)}")

    if not remaining_ids:
        print("All pairs already judged. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    items: List[Dict[str, Any]] = []
    for qid in remaining_ids:
        question = plans_1[qid]["query_text"] or plans_2[qid]["query_text"]
        plan_1 = plans_1[qid]["plan"]
        plan_2 = plans_2[qid]["plan"]

        messages = template.render(question=question, plan_1=plan_1, plan_2=plan_2)
        items.append({
            "query_id": qid,
            "question": question,
            "plan_1": plan_1,
            "plan_2": plan_2,
            "messages": messages,
        })

    # ------------------------------------------------------------------
    # Run judge
    # ------------------------------------------------------------------
    judge = VLLMJudge(
        model=args.model,
        base_url=args.model_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )

    judge.judge_and_write(items, args.output_file, num_threads=args.num_threads)

    log(f"Wrote {len(items)} results to {args.output_file}")
    print(f"\nDone. {len(items)} pairs judged → {args.output_file}")


if __name__ == "__main__":
    main()

    # python eval_plan_quality/compare_plans.py  --plan-file-1 gemini_2.5_pro_plans.jsonl --plan-file-2 gpt-oss-120b_plans.jsonl --output-file eval_plan_quality/comparison_results.jsonl --model Qwen/Qwen3.5-122B-A10B  --num-threads 8