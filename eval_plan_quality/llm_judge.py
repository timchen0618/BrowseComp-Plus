"""
Pointwise plan scoring using an LLM judge served by an external vLLM server.

Two modes:
  inference  — Run LLM judge on individual plans and write scores to JSONL.
  compute    — Read an output JSONL and compute score statistics.

Usage:
    # Inference (default when no subcommand given)
    python eval_plan_quality/llm_judge.py inference \
        --plan-file plans.jsonl \
        --output-file eval_plan_quality/judge_scores.jsonl \
        --model Qwen/Qwen3-32B --num-threads 8

    # Compute score statistics from an existing output file
    python eval_plan_quality/llm_judge.py compute \
        --output-file eval_plan_quality/judge_scores.jsonl
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

DEFAULT_PROMPT_TEMPLATE = "eval_plan_quality/prompts/pointwise_prompt.txt"
DEFAULT_MODEL_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_MAX_TOKENS = 8192


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Retry helpers
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
    user: str  # contains {question}, {plan} placeholders

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

    The user section may contain {question}, {plan} placeholders.
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


def _is_valid_judge_output(judge_output: str) -> bool:
    """Return True if a score can be extracted from the judge output."""
    return extract_score(judge_output) is not None


def load_completed_ids(path: Path) -> set:
    """Load query_ids that have *valid* judge outputs (parseable JSON with a score).

    Records with empty or truncated judge_output are excluded so they get
    retried on the next run.
    """
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
                if qid is not None and _is_valid_judge_output(obj.get("judge_output", "")):
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
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
        """Send one judgment request with retries.

        Raises on truncated output (finish_reason='length') so the retry
        logic can re-attempt with a fresh request.
        """

        def _call() -> str:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                raise RuntimeError(
                    f"response truncated (finish_reason='length', "
                    f"got {len(content)} chars). Increase --max-tokens."
                )
            return content

        return _retry_with_deadline(_call, max_retries=self.max_retries)

    def judge_and_write(
        self,
        items: List[Dict[str, Any]],
        output_file: Path,
        num_threads: int = 4,
    ) -> None:
        """Run pointwise scoring concurrently and write each result to file when done.

        Each item in *items* must have keys:
            query_id, question, plan, messages.
        Each completed result is appended to output_file immediately.
        """
        file_lock = threading.Lock()

        def _process(item: Dict[str, Any]) -> None:
            qid = item["query_id"]
            try:
                judge_output = self._judge_single(item["messages"])
            except Exception as e:
                log(f"[{qid}] judge failed after retries: {e!r} — skipping (will retry on next run)")
                return

            result = {
                "query_id": qid,
                "question": item["question"],
                "plan": item["plan"],
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
# Score extraction
# ---------------------------------------------------------------------------

def extract_score(judge_output: str) -> Optional[int]:
    """Extract the integer score from a judge output string.

    Tries these formats in order:
      1. <score>N</score> tags (primary — used by the current prompt).
      2. JSON with a "score" field (legacy structured output).
      3. Fallback regex for "score": N patterns.

    Returns an integer 1-5, or None if unparseable.
    """
    if not judge_output:
        return None

    # 1. <score>N</score> tags
    m = re.search(r"<score>\s*(\d)\s*</score>", judge_output)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # 2. JSON with a "score" field (legacy)
    try:
        obj = json.loads(judge_output)
        if isinstance(obj, dict) and "score" in obj:
            score = int(obj["score"])
            if 1 <= score <= 5:
                return score
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 3. Fallback: look for "score": N or score: N
    m = re.search(r'"?score"?\s*[:=]\s*(\d)', judge_output)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    return None


# ---------------------------------------------------------------------------
# Compute mode
# ---------------------------------------------------------------------------

def run_compute(args: argparse.Namespace) -> None:
    """Read an output JSONL and compute score statistics."""
    output_file = args.output_file
    if not output_file.exists():
        print(f"Output file {output_file} does not exist.", file=sys.stderr)
        sys.exit(1)

    verbose = getattr(args, "verbose", False)
    purge = getattr(args, "purge", False)

    total = 0
    scores: List[int] = []
    parse_failures = 0
    failed_query_ids: List[str] = []
    failed_records: List[Dict[str, Any]] = []
    valid_lines: List[str] = []
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    with output_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            judge_output = obj.get("judge_output", "")
            score = extract_score(judge_output)
            if score is not None:
                scores.append(score)
                score_counts[score] += 1
                valid_lines.append(line)
            else:
                parse_failures += 1
                failed_query_ids.append(str(obj.get("query_id", "?")))
                if verbose:
                    failed_records.append(obj)

    if total == 0:
        print("No records found in the output file.")
        return

    parseable = total - parse_failures
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"File: {output_file}")
    print(f"Total records:      {total}")
    print(f"Parseable scores:   {parseable}")
    print(f"Parse failures:     {parse_failures}")
    if failed_query_ids:
        print(f"  Failed query_ids: {', '.join(failed_query_ids[:20])}"
              + (" ..." if len(failed_query_ids) > 20 else ""))
    print()
    print(f"Average score: {avg_score:.2f}")
    print()
    print("Score distribution:")
    for s in range(1, 6):
        count = score_counts[s]
        pct = (count / parseable * 100) if parseable else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {s}: {count:>4d} ({pct:5.1f}%)  {bar}")

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

    if purge and parse_failures > 0:
        with output_file.open("w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        print(f"Purged {parse_failures} invalid record(s) from {output_file}")


# ---------------------------------------------------------------------------
# Inference mode
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    """Run LLM judge on individual plans and write scores to JSONL."""
    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    log(f"Loading plans from {args.plan_file}")
    plans = load_plans(args.plan_file)
    log(f"  → {len(plans)} entries")

    log(f"Loading prompt template from {args.prompt_template}")
    template: PromptTemplate = load_prompt_template(args.prompt_template)
    log(f"  system prompt: {len(template.system)} chars | user template: {len(template.user)} chars")

    all_ids = sorted(plans.keys())
    log(f"  {len(all_ids)} query_ids to score")

    if not all_ids:
        print("No plans found in the input file.", file=sys.stderr)
        sys.exit(1)

    # Resume support
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = load_completed_ids(args.output_file)
    remaining_ids = [qid for qid in all_ids if qid not in completed_ids]
    log(f"Total: {len(all_ids)} | Done: {len(all_ids) - len(remaining_ids)} | Remaining: {len(remaining_ids)}")

    if not remaining_ids:
        print("All plans already scored. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    items: List[Dict[str, Any]] = []
    for qid in remaining_ids:
        question = plans[qid]["query_text"]
        plan = plans[qid]["plan"]

        messages = template.render(question=question, plan=plan)
        items.append({
            "query_id": qid,
            "question": question,
            "plan": plan,
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
    print(f"\nDone. {len(items)} plans scored → {args.output_file}")

    # Compute score statistics over the full output file
    print()
    compute_args = argparse.Namespace(output_file=args.output_file, verbose=False)
    run_compute(compute_args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pointwise plan scoring: run LLM judge (inference) or compute score stats (compute).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # -- inference ---------------------------------------------------------
    inf = subparsers.add_parser(
        "inference",
        help="Run LLM judge on plans and write scores to JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inf.add_argument("--plan-file", required=True, type=Path,
                     help="JSONL file with plans to score")
    inf.add_argument(
        "--prompt-template", type=Path, default=Path(DEFAULT_PROMPT_TEMPLATE),
        help="Prompt template with <start_system_prompt>/<end_system_prompt> and "
             "<start_user_prompt>/<end_user_prompt> tags. "
             "User section must contain {question}, {plan} placeholders.",
    )
    inf.add_argument("--output-file", type=Path,
                     default=Path("eval_plan_quality/judge_scores.jsonl"),
                     help="Output JSONL for judge scores")
    inf.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                     help="Base URL for the external vLLM OpenAI-compatible server")
    inf.add_argument("--port", type=int, default=None,
                     help="vLLM server port (shorthand: sets --model-url to http://127.0.0.1:{port}/v1)")
    inf.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    inf.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top-p", type=float, default=0.95)
    inf.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    inf.add_argument("--num-threads", type=int, default=4,
                     help="Number of concurrent requests to the vLLM server")
    inf.add_argument("--max-retries", type=int, default=5,
                     help="Max retries per request on transient failures")

    # -- compute -----------------------------------------------------------
    comp = subparsers.add_parser(
        "compute",
        help="Compute score statistics from an existing output JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    comp.add_argument("--output-file", required=True, type=Path,
                      help="JSONL file produced by the inference mode")
    comp.add_argument("--verbose", "-v", action="store_true",
                      help="Print the judge output for each failed (unparseable) query")
    comp.add_argument("--purge", action="store_true",
                      help="Remove records with unparseable scores from the output file")

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

    # python eval_plan_quality/llm_judge.py inference --plan-file gemini_2.5_pro_plans.jsonl --output-file pointwise_judge_gemini.jsonl --model  Qwen/Qwen3.5-122B-A10B