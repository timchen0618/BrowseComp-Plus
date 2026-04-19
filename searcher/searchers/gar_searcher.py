"""Generation-Augmented Retrieval (GAR) searcher wrapper.

Wraps any BaseSearcher to augment search with parametric generation:
1. Original retrieval: FAISS/BM25 search on the subquery
2. Generation: Qwen3-8B generates parametric answer
3. Expansion retrieval: FAISS/BM25 search on compressed generation
4. Fusion: Weighted RRF merges both result lists

The agent only sees fused corpus docs — no generated evidence is injected.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Refusal patterns — if generation matches any of these, skip fusion
REFUSAL_PATTERNS = [
    "i don't know",
    "i'm not sure",
    "i cannot",
    "i do not know",
    "unable to determine",
    "no information available",
    "i'm unable",
    "i am not sure",
    "i am unable",
]


@dataclass
class GARConfig:
    mode: str = "query_expansion"  # query_expansion / gen_only / doc_injection / doc_append / placebo / query_rewrite
    alpha: float = 0.7  # Weight for original retrieval branch (RRF fusion)
    k_rrf: int = 60  # RRF constant
    enable_gate: bool = False  # Stub for Phase B gating
    max_compress_words: int = 150  # Truncate generation before embedding
    candidate_k_multiplier: int = 4  # candidate_k = max(20, multiplier * k)
    candidate_k_min: int = 20  # Minimum candidate pool size
    # Gate thresholds (Phase B)
    gate_min_tokens: int = 20
    gate_refusal_patterns: List[str] = field(default_factory=lambda: list(REFUSAL_PATTERNS))
    # Placebo control
    placebo_file: Optional[str] = None  # Path to placebo pool JSON


class GARSearcher:
    """Wraps a BaseSearcher with generation-augmented retrieval via RRF fusion.

    Not registered as a SearcherType — instantiated directly in tongyi_client.py
    by wrapping the inner searcher when --gar-mode is set.
    """

    def __init__(
        self,
        inner_searcher,
        generation_retriever,
        config: GARConfig,
        original_question: str = "",
        gar_generations_path: Optional[str] = None,
        prompt_fn=None,
        trajectory_window: int = 0,
    ):
        self.inner = inner_searcher
        self.generation_retriever = generation_retriever
        self.config = config
        self.prompt_fn = prompt_fn  # Callable for query_rewrite prompts
        self.trajectory_window = trajectory_window  # Recent subqueries to pass to prompt_fn

        # Thread-local storage for per-question state (enables multi-threaded GAR)
        self._local = threading.local()

        # Separate file for full generation text (keeps trajectories small)
        self._gar_gen_path = None
        self._gar_gen_lock = threading.Lock()  # Protects file writes
        if gar_generations_path:
            self._gar_gen_path = Path(gar_generations_path)
            self._gar_gen_path.parent.mkdir(parents=True, exist_ok=True)

        # Placebo control: pre-loaded pool of cross-question generations (read-only, thread-safe)
        self._placebo_pool: Optional[Dict[str, str]] = None
        if config.placebo_file:
            logger.info(f"Loading placebo pool from {config.placebo_file}")
            with open(config.placebo_file) as f:
                self._placebo_pool = json.load(f)
            logger.info(f"Loaded {len(self._placebo_pool)} placebo entries")

        # Initialize thread-local state for main thread
        self._init_local(original_question)

    def _init_local(self, original_question: str = ""):
        """Initialize thread-local per-question state."""
        self._local.original_question = original_question
        self._local.search_log = []
        self._local.total_generation_time_s = 0.0
        self._local.total_generation_tokens = 0
        self._local.generation_call_count = 0
        self._local.query_id = None
        self._local.step_counter = 0
        self._local.query_history = []  # Trajectory history for query_rewrite

    def _ensure_local(self):
        """Ensure thread-local state is initialized (for worker threads)."""
        if not hasattr(self._local, 'search_log'):
            self._init_local()

    def set_original_question(self, question: str):
        self._ensure_local()
        self._local.original_question = question

    def set_query_id(self, query_id: str):
        """Set current query_id for placebo pool lookup."""
        self._ensure_local()
        self._local.query_id = query_id
        self._local.step_counter = 0

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """GAR search: retrieve + generate + (optionally gate) + fuse."""
        self._ensure_local()

        # Query rewrite mode: separate path (single retrieval, no RRF)
        if self.config.mode == "query_rewrite":
            return self._query_rewrite(query, k)

        candidate_k = max(self.config.candidate_k_min, self.config.candidate_k_multiplier * k)

        # Step 1: Original retrieval
        r_orig = self.inner.search(query, candidate_k)

        # Step 2: Parametric generation (or placebo substitution)
        if self._placebo_pool is not None:
            gen_result = self._get_placebo(query)
        else:
            gen_result = self._generate(query)

        log_entry = {
            "subquery": query,
            "generation_text_truncated": "",
            "generation_text_length": 0,
            "generation_time_s": 0.0,
            "generation_tokens": 0,
            "gate_decision": True,
            "gate_reason": "no_gate",
            "r_orig_count": len(r_orig),
            "r_gen_count": 0,
            "fused_count": 0,
            "r_orig_docids": [r["docid"] for r in r_orig[:k]],
            "r_gen_docids": [],
            "fused_docids": [],
        }

        if gen_result is not None:
            log_entry["generation_text_truncated"] = gen_result["text"][:200]
            log_entry["generation_text_length"] = len(gen_result["text"])
            log_entry["generation_time_s"] = gen_result["generation_time_s"]
            log_entry["generation_tokens"] = gen_result["token_count"]

            # Write full generation text to separate file (thread-safe)
            if self._gar_gen_path and gen_result["text"]:
                with self._gar_gen_lock:
                    with open(self._gar_gen_path, "a") as f:
                        f.write(json.dumps({
                            "subquery": query,
                            "generation_text": gen_result["text"],
                            "generation_time_s": gen_result["generation_time_s"],
                            "generation_tokens": gen_result["token_count"],
                        }) + "\n")

        # Step 3: Gate check (fallback if generation failed/empty/refusal)
        if gen_result is None or not gen_result["text"].strip():
            log_entry["gate_decision"] = False
            log_entry["gate_reason"] = "generation_failed_or_empty"
            self._local.search_log.append(log_entry)
            return r_orig[:k]

        if self._is_refusal(gen_result["text"]):
            log_entry["gate_decision"] = False
            log_entry["gate_reason"] = "refusal_detected"
            self._local.search_log.append(log_entry)
            return r_orig[:k]

        if self.config.enable_gate:
            gate_pass, gate_reason = self._gate(gen_result)
            log_entry["gate_decision"] = gate_pass
            log_entry["gate_reason"] = gate_reason
            if not gate_pass:
                self._local.search_log.append(log_entry)
                return r_orig[:k]

        # Step 4: Mode-specific retrieval/fusion
        mode = self.config.mode

        if mode == "doc_injection":
            # Inject generation text directly as a fake document, prepend to original results
            gen_doc = {
                "docid": "parametric-gen",
                "score": 1.0,
                "text": gen_result["text"],
            }
            result = [gen_doc] + r_orig[:k - 1]
            log_entry["r_gen_count"] = 0
            log_entry["fused_count"] = len(result)
            log_entry["fused_docids"] = [r["docid"] for r in result]
            log_entry["gate_reason"] = "doc_injection"
            self._local.search_log.append(log_entry)
            return result

        if mode == "doc_append":
            # Append generation text as an extra document after the top-k retrieved results
            gen_doc = {
                "docid": "parametric-gen",
                "score": 0.0,
                "text": gen_result["text"],
            }
            result = r_orig[:k] + [gen_doc]
            log_entry["r_gen_count"] = 0
            log_entry["fused_count"] = len(result)
            log_entry["fused_docids"] = [r["docid"] for r in result]
            log_entry["gate_reason"] = "doc_append"
            self._local.search_log.append(log_entry)
            return result

        # For query_expansion and gen_only: embed generation and search FAISS
        compressed = self._compress(gen_result["text"])
        r_gen = self.inner.search(compressed, candidate_k)

        log_entry["r_gen_count"] = len(r_gen)
        log_entry["r_gen_docids"] = [r["docid"] for r in r_gen[:k]]

        if mode == "gen_only":
            # Return only generation-retrieved docs (no fusion with original)
            result = r_gen[:k]
            log_entry["fused_count"] = len(result)
            log_entry["fused_docids"] = [r["docid"] for r in result]
            log_entry["gate_reason"] = "gen_only"
            self._local.search_log.append(log_entry)
            return result

        # Default: query_expansion — Weighted RRF fusion
        fused = self._rrf_fuse(r_orig, r_gen, k)

        log_entry["fused_count"] = len(fused)
        log_entry["fused_docids"] = [r["docid"] for r in fused]

        self._local.search_log.append(log_entry)
        return fused

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        return self.inner.get_document(docid)

    @property
    def search_type(self) -> str:
        return "GAR-FAISS"

    def search_description(self, k: int = 10) -> str:
        return self.inner.search_description(k)

    def get_document_description(self) -> str:
        return self.inner.get_document_description()

    # --- GAR internals ---

    def _get_placebo(self, subquery: str) -> Optional[Dict[str, Any]]:
        """Look up pre-assigned placebo text for current (query_id, step_index)."""
        if self._local.query_id is None:
            logger.warning("Placebo mode but query_id not set — falling back to None")
            return None

        key = f"{self._local.query_id}::{self._local.step_counter}"
        self._local.step_counter += 1
        self._local.generation_call_count += 1

        text = self._placebo_pool.get(key)
        if text is None:
            logger.debug(f"No placebo entry for {key} — returning None")
            return None

        token_count = len(text.split())
        self._local.total_generation_tokens += token_count

        return {
            "text": text,
            "generation_time_s": 0.0,  # No actual generation
            "token_count": token_count,
        }

    def _generate(self, subquery: str) -> Optional[Dict[str, Any]]:
        """Call generation model and return result dict."""
        if self.generation_retriever is None:
            return None

        start = time.time()
        text = self.generation_retriever.generate_document(
            self._local.original_question, subquery
        )
        elapsed = time.time() - start

        self._local.generation_call_count += 1
        self._local.total_generation_time_s += elapsed

        if text is None:
            return None

        # Approximate token count from whitespace splitting
        token_count = len(text.split())
        self._local.total_generation_tokens += token_count

        return {
            "text": text,
            "generation_time_s": elapsed,
            "token_count": token_count,
        }

    def _is_refusal(self, text: str) -> bool:
        text_lower = text.lower()
        return any(p in text_lower for p in self.config.gate_refusal_patterns)

    def _gate(self, gen_result: Dict[str, Any]) -> tuple:
        """Phase B gating stub. Returns (should_apply_gar, reason)."""
        # Stub: always pass in Phase A
        if gen_result["token_count"] < self.config.gate_min_tokens:
            return False, "too_short"
        return True, "passed"

    def _compress(self, text: str) -> str:
        """Truncate generation to max_compress_words for embedding."""
        words = text.split()
        if len(words) > self.config.max_compress_words:
            return " ".join(words[: self.config.max_compress_words])
        return text

    def _rrf_fuse(
        self,
        r_orig: List[Dict[str, Any]],
        r_gen: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Weighted Reciprocal Rank Fusion.

        score(d) = alpha/(k_rrf + rank_orig) + (1-alpha)/(k_rrf + rank_gen)
        Documents in only one list get 0 for the missing term.
        """
        alpha = self.config.alpha
        k_rrf = self.config.k_rrf

        # Build rank maps (1-indexed)
        orig_rank = {r["docid"]: i + 1 for i, r in enumerate(r_orig)}
        gen_rank = {r["docid"]: i + 1 for i, r in enumerate(r_gen)}

        # Collect all unique docids
        all_docids = set(orig_rank.keys()) | set(gen_rank.keys())

        # Score each document
        scored = []
        for docid in all_docids:
            score = 0.0
            if docid in orig_rank:
                score += alpha / (k_rrf + orig_rank[docid])
            if docid in gen_rank:
                score += (1 - alpha) / (k_rrf + gen_rank[docid])
            scored.append((docid, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Build result list using document data from whichever list has it
        doc_data = {}
        for r in r_orig:
            doc_data[r["docid"]] = r
        for r in r_gen:
            if r["docid"] not in doc_data:
                doc_data[r["docid"]] = r

        results = []
        for docid, score in scored[:k]:
            entry = dict(doc_data[docid])
            entry["score"] = score
            results.append(entry)

        return results

    # --- Query rewrite mode ---

    def _query_rewrite(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Query rewrite mode: rewrite subquery via generation, single FAISS retrieval."""
        # Build trajectory context
        recent = None
        if self.trajectory_window > 0:
            recent = self._local.query_history[-self.trajectory_window:] or None

        # Generate rewritten query
        gen_result = self._generate_rewrite(query, recent)

        log_entry = {
            "subquery": query,
            "generation_text_truncated": "",
            "generation_text_length": 0,
            "generation_time_s": 0.0,
            "generation_tokens": 0,
            "gate_decision": True,
            "gate_reason": "query_rewrite",
            "rewritten_query": "",
            "recent_queries": list(recent) if recent else [],
            "r_orig_count": 0,
            "r_gen_count": 0,
            "fused_count": 0,
            "r_orig_docids": [],
            "r_gen_docids": [],
            "fused_docids": [],
        }

        # Always append original subquery to trajectory history (before rewriting)
        if self.trajectory_window > 0:
            self._local.query_history.append(query)

        # Fallback to baseline on generation failure / empty
        if gen_result is None or not gen_result["text"].strip():
            log_entry["gate_decision"] = False
            log_entry["gate_reason"] = "generation_failed_or_empty"
            results = self.inner.search(query, k)
            log_entry["fused_count"] = len(results)
            log_entry["fused_docids"] = [r["docid"] for r in results]
            self._local.search_log.append(log_entry)
            return results

        rewritten = gen_result["text"].strip()
        log_entry["generation_text_truncated"] = rewritten[:200]
        log_entry["generation_text_length"] = len(rewritten)
        log_entry["generation_time_s"] = gen_result["generation_time_s"]
        log_entry["generation_tokens"] = gen_result["token_count"]
        log_entry["rewritten_query"] = rewritten

        # Write full generation to separate file (thread-safe)
        if self._gar_gen_path and rewritten:
            with self._gar_gen_lock:
                with open(self._gar_gen_path, "a") as f:
                    f.write(json.dumps({
                        "subquery": query,
                        "rewritten_query": rewritten,
                        "generation_time_s": gen_result["generation_time_s"],
                        "generation_tokens": gen_result["token_count"],
                        "recent_queries": list(recent) if recent else [],
                    }) + "\n")

        # Fallback on refusal
        if self._is_refusal(rewritten):
            log_entry["gate_decision"] = False
            log_entry["gate_reason"] = "refusal_detected"
            results = self.inner.search(query, k)
            log_entry["fused_count"] = len(results)
            log_entry["fused_docids"] = [r["docid"] for r in results]
            self._local.search_log.append(log_entry)
            return results

        # Single FAISS retrieval with the rewritten query string
        results = self.inner.search(rewritten, k)
        log_entry["fused_count"] = len(results)
        log_entry["fused_docids"] = [r["docid"] for r in results]
        self._local.search_log.append(log_entry)
        return results

    def _generate_rewrite(self, subquery: str, recent_queries: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Call generation model with query rewrite prompt from PROMPTS registry."""
        if self.generation_retriever is None or self.prompt_fn is None:
            return None

        messages = self.prompt_fn(
            self._local.original_question, subquery, recent_queries=recent_queries
        )

        start = time.time()
        text = self.generation_retriever.generate_raw(messages, max_tokens=64)
        elapsed = time.time() - start

        self._local.generation_call_count += 1
        self._local.total_generation_time_s += elapsed

        if text is None:
            return None

        token_count = len(text.split())
        self._local.total_generation_tokens += token_count

        return {
            "text": text,
            "generation_time_s": elapsed,
            "token_count": token_count,
        }

    # --- Logging accessors ---

    def get_gar_metadata(self) -> Dict[str, Any]:
        """Return GAR-specific metadata for trajectory persistence."""
        self._ensure_local()
        return {
            "gar_alpha": self.config.alpha,
            "gar_k_rrf": self.config.k_rrf,
            "gar_enable_gate": self.config.enable_gate,
            "total_generation_time_s": self._local.total_generation_time_s,
            "total_generation_tokens": self._local.total_generation_tokens,
            "generation_call_count": self._local.generation_call_count,
            "gar_subquery_log": self._local.search_log,
        }

    def reset_logs(self):
        """Reset per-question logs (call before each new question)."""
        self._init_local()
