"""
Pairwise plan comparison using an LLM judge served by an external vLLM server.

Two modes:
  inference  — Run LLM judge on plan pairs and write verdicts to JSONL.
  compute    — Read an output JSONL and compute win-rate statistics.

Usage:
    # Inference (default when no subcommand given)
    python eval_plan_quality/compare_plans.py inference \
        --plan-file-1 gemini_2.5_pro_plans.jsonl \
        --plan-file-2 gpt-oss-120b_plans.jsonl \
        --output-file eval_plan_quality/comparison_results.jsonl \
        --model Qwen/Qwen3-32B --num-threads 8

    # Compute win rates from an existing output file
    python eval_plan_quality/compare_plans.py compute \
        --output-file eval_plan_quality/comparison_results.jsonl
"""

import argparse
import json
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

    JUDGE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "plan_comparison",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "thinking": {
                        "type": "string",
                        "description": "Step-by-step reasoning about the strengths and weaknesses of each plan.",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["Plan 1 is better", "Plan 2 is better"],
                        "description": "Which plan is better.",
                    },
                },
                "required": ["thinking", "verdict"],
                "additionalProperties": False,
            },
        },
    }

    def _judge_single(self, messages: List[Dict[str, str]]) -> str:
        """Send one judgment request with retries."""

        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                response_format=self.JUDGE_SCHEMA,
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
# Verdict parsing
# ---------------------------------------------------------------------------

VERDICT_PATTERN = re.compile(r"<verdict>\s*(.*?)\s*</verdict>", re.DOTALL | re.IGNORECASE)
PLAN1_PATTERN = re.compile(r"plan\s*1\s*(is\s+)?better", re.IGNORECASE)
PLAN2_PATTERN = re.compile(r"plan\s*2\s*(is\s+)?better", re.IGNORECASE)


def _match_verdict_text(text: str) -> Optional[str]:
    if PLAN1_PATTERN.search(text):
        return "plan_1"
    if PLAN2_PATTERN.search(text):
        return "plan_2"
    return None


def extract_verdict(judge_output: str) -> Optional[str]:
    """Extract the verdict from a judge output string.

    Supports two formats:
      1. JSON with a "verdict" field (structured output from the schema).
      2. Legacy <verdict>...</verdict> XML tags.

    Returns "plan_1", "plan_2", or None if unparseable.
    """
    # Try JSON first
    try:
        obj = json.loads(judge_output)
        if isinstance(obj, dict) and "verdict" in obj:
            return _match_verdict_text(obj["verdict"])
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to <verdict> tags
    m = VERDICT_PATTERN.search(judge_output)
    if m:
        return _match_verdict_text(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Compute mode
# ---------------------------------------------------------------------------

def run_compute(args: argparse.Namespace) -> None:
    """Read an output JSONL and compute win-rate statistics."""
    output_file = args.output_file
    if not output_file.exists():
        print(f"Output file {output_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    verbose = getattr(args, "verbose", False)

    total = 0
    plan_1_wins = 0
    plan_2_wins = 0
    parse_failures = 0
    failed_query_ids: List[str] = []
    failed_records: List[Dict[str, Any]] = []

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            judge_output = obj.get("judge_output", "")
            verdict = extract_verdict(judge_output)
            if verdict == "plan_1":
                plan_1_wins += 1
            elif verdict == "plan_2":
                plan_2_wins += 1
            else:
                parse_failures += 1
                failed_query_ids.append(str(obj.get("query_id", "?")))
                if verbose:
                    failed_records.append(obj)

    if total == 0:
        print("No records found in the output file.")
        return

    parseable = total - parse_failures
    p1_pct = (plan_1_wins / parseable * 100) if parseable else 0.0
    p2_pct = (plan_2_wins / parseable * 100) if parseable else 0.0

    print(f"File: {output_file}")
    print(f"Total records:      {total}")
    print(f"Parseable verdicts: {parseable}")
    print(f"Parse failures:     {parse_failures}")
    if failed_query_ids:
        print(f"  Failed query_ids: {', '.join(failed_query_ids[:20])}"
              + (" ..." if len(failed_query_ids) > 20 else ""))
    print()
    print(f"Plan 1 wins: {plan_1_wins:>4d} / {parseable}  ({p1_pct:.1f}%)")
    print(f"Plan 2 wins: {plan_2_wins:>4d} / {parseable}  ({p2_pct:.1f}%)")

    if verbose and failed_records:
        print(f"\n{'=' * 72}")
        print(f"Failed queries ({len(failed_records)}):")
        print(f"{'=' * 72}")
        for rec in failed_records:
            qid = rec.get("query_id", "?")
            judge_output = rec.get("judge_output", "")
            print(f"\n--- query_id: {qid} ---")
            print(judge_output if judge_output else "(empty judge output)")
            print()


# ---------------------------------------------------------------------------
# Inference mode
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    """Run LLM judge on plan pairs and write verdicts to JSONL."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise plan comparison: run LLM judge (inference) or compute win rates (compute).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # -- inference ---------------------------------------------------------
    inf = subparsers.add_parser(
        "inference",
        help="Run LLM judge on plan pairs and write verdicts to JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inf.add_argument("--plan-file-1", required=True, type=Path,
                     help="JSONL file with plans from the first model/method")
    inf.add_argument("--plan-file-2", required=True, type=Path,
                     help="JSONL file with plans from the second model/method")
    inf.add_argument(
        "--prompt-template", type=Path, default=Path(DEFAULT_PROMPT_TEMPLATE),
        help="Prompt template with <start_system_prompt>/<end_system_prompt> and "
             "<start_user_prompt>/<end_user_prompt> tags. "
             "User section must contain {question}, {plan_1}, {plan_2} placeholders.",
    )
    inf.add_argument("--output-file", type=Path,
                     default=Path("eval_plan_quality/comparison_results.jsonl"),
                     help="Output JSONL for judge verdicts")
    inf.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                     help="Base URL for the external vLLM OpenAI-compatible server")
    inf.add_argument("--port", type=int, default=None,
                     help="vLLM server port (shorthand: sets --model-url to http://127.0.0.1:{port}/v1)")
    inf.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    inf.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top-p", type=float, default=0.95)
    inf.add_argument("--max-tokens", type=int, default=4096)
    inf.add_argument("--num-threads", type=int, default=4,
                     help="Number of concurrent requests to the vLLM server")
    inf.add_argument("--max-retries", type=int, default=3,
                     help="Max retries per request on transient failures")

    # -- compute -----------------------------------------------------------
    comp = subparsers.add_parser(
        "compute",
        help="Compute win-rate statistics from an existing output JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    comp.add_argument("--output-file", required=True, type=Path,
                      help="JSONL file produced by the inference mode")
    comp.add_argument("--verbose", "-v", action="store_true",
                      help="Print the judge output for each failed (unparseable) query")

    # -- dispatch ----------------------------------------------------------
    args = parser.parse_args()

    if args.mode is None or args.mode == "inference":
        if args.mode is None:
            args = inf.parse_args(sys.argv[1:])
        run_inference(args)
    elif args.mode == "compute":
        run_compute(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

    # python eval_plan_quality/compare_plans.py inference  --plan-file-1 gemini_2.5_pro_plans.jsonl --plan-file-2 gpt-oss-120b_plans.jsonl --output-file eval_plan_quality/comparison_results.jsonl --model Qwen/Qwen3.5-122B-A10B  --num-threads 8

    # python eval_plan_quality/compare_plans.py compute --output-file eval_plan_quality/comparison_results.jsonl