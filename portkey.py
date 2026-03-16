# from portkey_ai import Portkey

# portkey = Portkey(
#   api_key = "uES/JwXt2/8K6VmT2zGyU+WhOlNc"
# )

# response = portkey.chat.completions.create(
#     model = "@vertexai-gemini-ec5413/gemini-2.5-pro",
#     messages = [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "What is Portkey"}
#     ],
#     MAX_TOKENS = 512
# )

# print(response.choices[0].message.content)



"""
Portkey-based wrapper for Gemini 2.5 Pro (Vertex AI) with the same interface/params
as your Llama33_70B / OpenAIChat wrappers.

- Uses Portkey's OpenAI-compatible chat.completions API.
- Expects a list of chat messages: [{"role": "user"/"assistant"/"system", "content": "..."}].
- Returns a raw text string (no usage dict), so it plugs into the same pipeline as vLLM wrappers.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional, Callable, TypeVar
import os
import time
import random
import concurrent.futures

from portkey_ai import Portkey  # pip install portkey-ai

import sys
from datetime import datetime
from tqdm import tqdm

# Keep the parameter *structure* aligned with your other wrappers
DEFAULT_GEMINI_STOPS = None  # harmless default; can be overridden

T = TypeVar("T")

def log_err(msg: str) -> None:
    """
    Print timestamped message to stderr (goes to Slurm .err file).
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", file=sys.stderr, flush=True)


@dataclass
class GenParams:
    # Same fields as your Llama GenParams; not all are sent to the API.
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20  # kept for structural parity, not always supported via OpenAI-compatible APIs
    max_new_tokens: Optional[int] = 16384
    repetition_penalty: Optional[float] = 1.0  # kept for parity; not always supported
    stop: Optional[List[str]] = None


@dataclass
class RetryConfig:
    # Generic (non-model-specific) retry/timeout knobs
    max_retries: int = 5                 # total retries after the first attempt
    per_attempt_timeout_s: float = 90.0  # hard wall-clock timeout per attempt
    total_deadline_s: float = 240.0      # total wall-clock deadline across all attempts
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
        # Try to cancel if it hasn't started (usually already running, so this may be False)
        fut.cancel()
        # CRITICAL: don't wait for the stuck thread
        ex.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"request exceeded {timeout_s:.1f}s") from e

    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)
        raise

    finally:
        # In the success path, shutdown cleanly
        # (If already shutdown above, harmless)
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _retry_with_deadline(fn: Callable[[], T], cfg: RetryConfig) -> T:
    """
    Generic retry wrapper:
      - per-attempt hard timeout
      - total deadline
      - jittered exponential backoff
    """
    start = time.time()
    last_err: Optional[BaseException] = None

    # attempts = first try + retries
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

            # jittered exponential backoff (capped)
            backoff = min(cfg.max_backoff_s, cfg.base_backoff_s * (2 ** attempt))
            backoff *= 0.5 + random.random()  # jitter in [0.5, 1.5)
            time.sleep(backoff)

    # unreachable, but keeps type-checkers happy
    raise RuntimeError("retry loop fell through") from last_err


class Gemini25Pro:
    def __init__(
        self,
        model: str = "@vertexai-gemini-ec5413/gemini-2.5-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_cfg: Optional[RetryConfig] = None,
    ):
        """
        model: e.g. "@vertexai-gemini-ec5413/gemini-2.5-pro"
               or "@vertexai/gemini-2.5-pro" depending on what your gateway exposes.
        base_url: optional, e.g. "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"
        """
        self.model_name = model

        api_key = api_key or os.getenv("PORTKEY_API_KEY") or os.getenv("PORTKEY_API_TOKEN")
        if not api_key:
            raise RuntimeError(
                "PORTKEY_API_KEY is not set. Put it in your environment or pass api_key=..."
            )
        else:
            print(f"API key: {api_key}")

        base_url = base_url or os.getenv("PORTKEY_BASE_URL")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = Portkey(**client_kwargs)

        # Hardcoded decoding defaults (mirrors Llama33_70B keys)
        self._defaults: Dict[str, Any] = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 16384,
            "repetition_penalty": 1.0,
            "stop": DEFAULT_GEMINI_STOPS,
        }

        # Generic retry config (can be overridden by env vars without touching call sites)
        if retry_cfg is None:
            retry_cfg = RetryConfig(
                max_retries=int(os.getenv("LLM_MAX_RETRIES", os.getenv("PORTKEY_MAX_RETRIES", "3"))),
                per_attempt_timeout_s=float(os.getenv("LLM_PER_ATTEMPT_TIMEOUT_S", os.getenv("PORTKEY_TIMEOUT_S", "180"))),
                total_deadline_s=float(os.getenv("LLM_TOTAL_DEADLINE_S", os.getenv("PORTKEY_DEADLINE_S", "720"))),
                base_backoff_s=float(os.getenv("LLM_BASE_BACKOFF_S", "1.0")),
                max_backoff_s=float(os.getenv("LLM_MAX_BACKOFF_S", "20.0")),
            )
        self._retry_cfg = retry_cfg

        # Gemini doesn't emit local-style <think> blocks by default in your setup
        self.supports_think = False

    # --- internal helpers -------------------------------------------------

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
        """
        Convert our internal params into Portkey chat.completions kwargs.
        Only send keys that are generally supported by OpenAI-compatible gateways.
        """
        max_tokens = int(eff["max_new_tokens"])
        temperature = float(eff["temperature"])
        top_p = float(eff["top_p"])
        stop = eff["stop"]

        kw: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
        }

        # stop strings (if any)
        if stop:
            kw["stop"] = stop

        # NOTE:
        # We intentionally do NOT send top_k or repetition_penalty by default because
        # OpenAI-compatible chat endpoints often reject them.
        # Keep them for parity + describe_params(), and add later only if you confirm support.
        return kw

    # --- public API -------------------------------------------------------

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
        """
        Mirrors describe_params() from Llama33_70B, but marks backend='portkey'.
        """
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
            # Static placeholders for compatibility with your local wrappers
            "tensor_parallel_size": None,
            "gpu_memory_utilization": None,
            "max_model_len": None,
        }
        
        
if __name__ == "__main__":
    model = Gemini25Pro(base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/")
    gen_params = GenParams(max_new_tokens=4096, temperature=0.7, top_p=0.95)
    # output = model.generate(messages=[{"role": "user", "content": "What is the capital of France?"}], params=gen_params)
    # print(output)
    
    def read_tsv(file_path: str) -> List[Dict[str, str]]:
        import csv
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            return [row for row in reader]
          
    def append_jsonl(file_path: str, data: List[Dict[str, str]]) -> None:
        import json
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
                
      
    
    from search_agent.tongyi_utils.prompts import PROMPT_PLANNER
    NUM_QUERIES = 100
    data = read_tsv("topics-qrels/queries.tsv")
    
    output_file = "gemini_2.5_pro_plans.jsonl"
    for query in tqdm(data[NUM_QUERIES:], desc="Generating plans"):
        query_id = query[0]
        query_text = query[1]
        output = model.generate(messages=[{"role": "system", "content": PROMPT_PLANNER}, {"role": "user", "content": f"User Question: {query_text}"}], params=gen_params)
        print("-"*100)
        print(f"Query ID: {query_id}")
        print(f"Query Text: {query_text}")
        print(f"Output: {output}")
        
        
        append_jsonl(output_file, [{"query_id": query_id, "query_text": query_text, "output": output}])
