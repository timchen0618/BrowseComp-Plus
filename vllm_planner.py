"""
vLLM-based plan generator with the same interface/params as portkey.py.

- Uses OpenAI-compatible chat.completions API served by a vLLM instance.
- Reads queries from a TSV file, generates plans via prompts/planning_prompt_v6.md, writes JSONL.
- Supports resume (skips already-processed query_ids) and threaded concurrency.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional, Callable, TypeVar
import argparse
import csv
import json
import os
import sys
import time
import random
import threading
import concurrent.futures
from datetime import datetime
from pathlib import Path

import openai
from tqdm import tqdm

DEFAULT_STOPS = None

_PLANNER_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "planning_prompt_v6.1.md"
PLANNER_SYSTEM_PROMPT = _PLANNER_PROMPT_PATH.read_text(encoding="utf-8")

T = TypeVar("T")


def log_err(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


@dataclass
class GenParams:
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_new_tokens: Optional[int] = 16384
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None


@dataclass
class RetryConfig:
    max_retries: int = 5
    per_attempt_timeout_s: float = 90.0
    total_deadline_s: float = 240.0
    base_backoff_s: float = 1.0
    max_backoff_s: float = 20.0


def _call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    log_err(f"_call_with_timeout: waiting up to {timeout_s:.1f}s")

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)

    try:
        out = fut.result(timeout=timeout_s)
        log_err("_call_with_timeout: completed")
        return out

    except concurrent.futures.TimeoutError as e:
        log_err("_call_with_timeout: TIMEOUT fired")
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


def _retry_with_deadline(fn: Callable[[], T], cfg: RetryConfig) -> T:
    start = time.time()
    last_err: Optional[BaseException] = None

    attempts_total = max(1, int(cfg.max_retries) + 1)

    for attempt in range(attempts_total):
        elapsed = time.time() - start
        if elapsed > cfg.total_deadline_s:
            raise TimeoutError(
                f"total deadline exceeded ({cfg.total_deadline_s:.1f}s)"
            ) from last_err
        log_err(f"Attempt {attempt+1}/{attempts_total}")
        try:
            return _call_with_timeout(fn, cfg.per_attempt_timeout_s)

        except Exception as e:
            log_err(f"Attempt {attempt+1} failed: {repr(e)}")
            last_err = e
            if attempt >= attempts_total - 1:
                raise

            backoff = min(cfg.max_backoff_s, cfg.base_backoff_s * (2 ** attempt))
            backoff *= 0.5 + random.random()
            time.sleep(backoff)

    raise RuntimeError("retry loop fell through") from last_err


class VLLMPlanner:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-32B",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        retry_cfg: Optional[RetryConfig] = None,
    ):
        self.model_name = model

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=600.0,
        )

        self._defaults: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "stop": DEFAULT_STOPS,
        }

        if retry_cfg is None:
            retry_cfg = RetryConfig(
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
                per_attempt_timeout_s=float(os.getenv("LLM_PER_ATTEMPT_TIMEOUT_S", "180")),
                total_deadline_s=float(os.getenv("LLM_TOTAL_DEADLINE_S", "720")),
                base_backoff_s=float(os.getenv("LLM_BASE_BACKOFF_S", "1.0")),
                max_backoff_s=float(os.getenv("LLM_MAX_BACKOFF_S", "20.0")),
            )
        self._retry_cfg = retry_cfg

        self.supports_think = False

    def _effective_params(self, override: Optional[GenParams]) -> Dict[str, Any]:
        d = dict(self._defaults)
        if override is not None:
            if override.temperature is not None:
                d["temperature"] = override.temperature
            if override.top_p is not None:
                d["top_p"] = override.top_p
            if override.top_k is not None:
                d["top_k"] = override.top_k
            if override.max_new_tokens is not None:
                d["max_new_tokens"] = override.max_new_tokens
            if override.repetition_penalty is not None:
                d["repetition_penalty"] = override.repetition_penalty
            if override.stop is not None:
                d["stop"] = override.stop
        return d

    def _build_api_kwargs(
        self,
        messages: List[Dict[str, str]],
        eff: Dict[str, Any],
    ) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(eff["max_new_tokens"]),
            "top_p": float(eff["top_p"]),
            "temperature": float(eff["temperature"]),
        }

        if eff["stop"]:
            kw["stop"] = eff["stop"]

        return kw

    def name(self) -> str:
        return f"vllm/{self.model_name}"

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        params: Optional[GenParams] = None,
    ) -> str:

        if isinstance(messages, str):
            msg_list: List[Dict[str, str]] = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        eff = self._effective_params(params)

        def one_call() -> str:
            kwargs_api = self._build_api_kwargs(msg_list, eff)

            log_err(f"[{self.model_name}] Starting API call "
                    f"(max_tokens={kwargs_api['max_tokens']})")

            start = time.time()
            resp = self.client.chat.completions.create(**kwargs_api)
            duration = time.time() - start

            log_err(f"[{self.model_name}] API call finished in {duration:.2f}s")

            return resp.choices[0].message.content or ""

        log_err(f"[{self.model_name}] generate() called")

        try:
            result = _retry_with_deadline(one_call, self._retry_cfg)
            log_err(f"[{self.model_name}] generate() success")
            return result
        except Exception as e:
            log_err(f"[{self.model_name}] generate() FAILED: {repr(e)}")
            raise

    def describe_params(self) -> Dict[str, Any]:
        d = self._defaults
        return {
            "model_name": self.model_name,
            "model_path": f"vllm://{self.model_name}",
            "backend": "vllm",
            "dtype": "n/a",
            "temperature": d["temperature"],
            "top_p": d["top_p"],
            "top_k": d["top_k"],
            "max_new_tokens": d["max_new_tokens"],
            "repetition_penalty": d["repetition_penalty"],
            "stop": d["stop"],
            "tensor_parallel_size": None,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tsv(file_path: str) -> List[List[str]]:
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        return [row for row in reader]


def append_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_completed_ids(file_path: str) -> set:
    ids: set = set()
    p = Path(file_path)
    if not p.exists():
        return ids
    with p.open("r", encoding="utf-8") as f:
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate plans for queries using a vLLM server endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-url",
        default="http://127.0.0.1:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-32B",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the vLLM server (usually EMPTY)",
    )
    parser.add_argument(
        "--query-file",
        default="topics-qrels/queries.tsv",
        help="TSV file with (query_id, query_text) rows",
    )
    parser.add_argument(
        "--output-file",
        default="vllm_plans.jsonl",
        help="Output JSONL file for generated plans",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate per query",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Number of queries to skip from the start (offset into TSV)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for generation",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="vLLM server port (alternative to --model-url; sets URL to http://127.0.0.1:{port}/v1)",
    )

    args = parser.parse_args()

    if args.port is not None:
        args.model_url = f"http://127.0.0.1:{args.port}/v1"

    model = VLLMPlanner(
        model=args.model,
        base_url=args.model_url,
        api_key=args.api_key,
    )
    gen_params = GenParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    data = read_tsv(args.query_file)
    queries = data[args.num_queries:]

    completed_ids = load_completed_ids(args.output_file)
    remaining = [q for q in queries if q[0] not in completed_ids]

    print(
        f"Total queries: {len(queries)} | Already done: {len(queries) - len(remaining)} | Remaining: {len(remaining)}"
    )

    write_lock = threading.Lock()

    def process_query(query: List[str]) -> None:
        query_id = query[0]
        query_text = query[1]

        output = model.generate(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Question: {query_text}"},
            ],
            params=gen_params,
        )

        print("-" * 100)
        print(f"Query ID: {query_id}")
        print(f"Query Text: {query_text}")
        print(f"Output: {output}")

        with write_lock:
            append_jsonl(
                args.output_file,
                [{"query_id": query_id, "query_text": query_text, "output": output}],
            )

    if args.num_threads <= 1:
        for query in tqdm(remaining, desc="Generating plans"):
            process_query(query)
    else:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Generating plans") as pbar,
        ):
            futures = [executor.submit(process_query, q) for q in remaining]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                pbar.update(1)


if __name__ == "__main__":
    main()


# python vllm_planner.py --model-url http://localhost:8000/v1 --model Qwen/Qwen3.5-35B-A3B --output-file vllm_plans.jsonl --num-threads 4