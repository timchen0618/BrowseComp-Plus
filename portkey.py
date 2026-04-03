"""
Portkey-based wrapper for Gemini 2.5 Pro (Vertex AI) with the same interface/params
as Llama33_70B / OpenAIChat wrappers.

- Uses Portkey's OpenAI-compatible chat.completions API.
- Expects a list of chat messages: [{"role": "user"/"assistant"/"system", "content": "..."}].
- Returns a raw text string, so it plugs into the same pipeline as vLLM wrappers.
"""

import dataclasses
import os
import random
import sys
import threading
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from portkey_ai import Portkey  # pip install portkey-ai
from tqdm import tqdm

T = TypeVar("T")


def log_err(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


@dataclass
class GenParams:
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20          # kept for structural parity; not forwarded to API
    max_new_tokens: Optional[int] = 16384
    repetition_penalty: Optional[float] = 1.0  # kept for parity; not forwarded to API
    stop: Optional[List[str]] = None


@dataclass
class RetryConfig:
    max_retries: int = 5
    per_attempt_timeout_s: float = 90.0
    total_deadline_s: float = 240.0
    base_backoff_s: float = 1.0
    max_backoff_s: float = 60.0         # longer cap for gateway 504s


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
        raise TimeoutError(f"request exceeded {timeout_s:.1f}s") from e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _retry_with_deadline(fn: Callable[[], T], cfg: RetryConfig) -> T:
    start = time.time()
    last_err: Optional[BaseException] = None
    attempts_total = max(1, int(cfg.max_retries) + 1)

    for attempt in range(attempts_total):
        if time.time() - start > cfg.total_deadline_s:
            raise TimeoutError(
                f"total deadline exceeded ({cfg.total_deadline_s:.1f}s)"
            ) from last_err

        log_err(f"Attempt {attempt + 1}/{attempts_total}")
        try:
            return _call_with_timeout(fn, cfg.per_attempt_timeout_s)
        except Exception as e:
            log_err(f"Attempt {attempt + 1} failed: {repr(e)}")
            last_err = e
            if attempt >= attempts_total - 1:
                raise

            is_504 = "504" in repr(last_err) or "Gateway Time-out" in repr(last_err)
            base = cfg.base_backoff_s * (3 if is_504 else 1)
            backoff = min(cfg.max_backoff_s, base * (2 ** attempt))
            backoff *= 0.5 + random.random()  # jitter in [0.5, 1.5)
            log_err(f"Backoff {backoff:.1f}s before next attempt (504={is_504})")
            time.sleep(backoff)

    raise RuntimeError("retry loop fell through") from last_err


class Gemini25Pro:
    def __init__(
        self,
        model: str = "@vertexai-gemini-ec5413/gemini-2.5-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_cfg: Optional[RetryConfig] = None,
    ):
        self.model_name = model

        api_key = api_key or os.getenv("PORTKEY_API_KEY") or os.getenv("PORTKEY_API_TOKEN")
        if not api_key:
            raise RuntimeError(
                "PORTKEY_API_KEY is not set. Put it in your environment or pass api_key=..."
            )
        log_err("PORTKEY_API_KEY is set")

        base_url = base_url or os.getenv("PORTKEY_BASE_URL")
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = Portkey(**client_kwargs)

        # Canonical defaults come from GenParams — single source of truth
        self._defaults: Dict[str, Any] = dataclasses.asdict(GenParams())

        if retry_cfg is None:
            retry_cfg = RetryConfig(
                max_retries=int(os.getenv("LLM_MAX_RETRIES", os.getenv("PORTKEY_MAX_RETRIES", "4"))),
                per_attempt_timeout_s=float(os.getenv("LLM_PER_ATTEMPT_TIMEOUT_S", os.getenv("PORTKEY_TIMEOUT_S", "180"))),
                total_deadline_s=float(os.getenv("LLM_TOTAL_DEADLINE_S", os.getenv("PORTKEY_DEADLINE_S", "1800"))),
                base_backoff_s=float(os.getenv("LLM_BASE_BACKOFF_S", "2.0")),
                max_backoff_s=float(os.getenv("LLM_MAX_BACKOFF_S", "60.0")),
            )
        self._retry_cfg = retry_cfg
        self.supports_think = False

    def _effective_params(self, override: Optional[GenParams]) -> Dict[str, Any]:
        d = dict(self._defaults)
        if override is not None:
            d.update({k: v for k, v in dataclasses.asdict(override).items() if v is not None})
        return d

    def _build_api_kwargs(
        self,
        messages: List[Dict[str, str]],
        eff: Dict[str, Any],
    ) -> Dict[str, Any]:
        # top_k and repetition_penalty are intentionally omitted — OpenAI-compatible
        # endpoints typically reject them.
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
        return f"portkey/{self.model_name}"

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
            params_summary = {k: v for k, v in kwargs_api.items() if k != "messages"}
            log_err(f"[{self.model_name}] effective params: {params_summary}")
            log_err(f"[{self.model_name}] Starting API call (max_tokens={kwargs_api['max_tokens']})")
            start = time.time()
            # Stream to avoid intermediate gateway 504 timeouts.
            stream = self.client.chat.completions.create(**kwargs_api, stream=True)
            content_parts = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)
            log_err(f"[{self.model_name}] API call finished in {time.time() - start:.2f}s")
            return "".join(content_parts)

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
            "model_path": f"portkey://{self.model_name}",
            "backend": "portkey",
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


if __name__ == "__main__":
    import argparse
    import csv
    import json
    # python portkey.py --prompt-file prompts/planning_prompt_v1.md --queries topics-qrels/frames/queries_first50.tsv --output output_plans/gemini_2.5_pro/frames/gemini_2.5_pro_plans_v1_first50.jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--prompt-file", default="prompts/planning_prompt_v1.md",
                    help="Path to the system prompt markdown file")
    ap.add_argument("--queries", default="topics-qrels/bcp/queries.tsv",
                    help="Path to the queries TSV file")
    ap.add_argument("--output", default="gemini_2.5_pro_plans_updated.jsonl",
                    help="Path to the output JSONL file")
    cli = ap.parse_args()

    model = Gemini25Pro(base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/")
    gen_params = GenParams(max_new_tokens=2048, temperature=0.7, top_p=0.95)

    def read_tsv(file_path: str) -> List[List[str]]:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t"))

    def append_jsonl(file_path: str, item: Dict[str, Any]) -> None:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")

    def load_completed_ids(file_path: str) -> set:
        ids: set = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
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
        except FileNotFoundError:
            pass
        return ids

    planner_system = Path(cli.prompt_file).read_text(encoding="utf-8")

    data = read_tsv(cli.queries)
    output_file = cli.output
    completed_ids = load_completed_ids(output_file)
    remaining = [q for q in data if q[0] not in completed_ids]
    print(f"Total queries: {len(data)} | Already done: {len(data) - len(remaining)} | Remaining: {len(remaining)}")

    write_lock = threading.Lock()

    def process_query(query: List[str]) -> None:
        query_id, query_text = query[0], query[1]
        output = model.generate(
            messages=[
                {"role": "system", "content": planner_system},
                {"role": "user", "content": f"User Question: {query_text}"},
            ],
            params=gen_params,
        )
        print("-" * 100)
        print(f"Query ID: {query_id}")
        print(f"Query Text: {query_text}")
        print(f"Output: {output}")
        with write_lock:
            append_jsonl(output_file, {"query_id": query_id, "query_text": query_text, "output": output})

    if cli.num_threads <= 1:
        for query in tqdm(remaining, desc="Generating plans"):
            process_query(query)
    else:
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=cli.num_threads) as executor,
            tqdm(total=len(remaining), desc="Generating plans") as pbar,
        ):
            futures = [executor.submit(process_query, q) for q in remaining]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                pbar.update(1)


# claude --resume 4e4a91f8-d9ca-4721-8bc4-9140c4c7e050 