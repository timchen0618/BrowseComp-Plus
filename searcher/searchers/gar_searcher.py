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
    alpha: float = 0.7  # Weight for original retrieval branch (TODO: tune)
    k_rrf: int = 60  # RRF constant
    enable_gate: bool = False  # Stub for Phase B gating
    max_compress_words: int = 150  # Truncate generation before embedding
    candidate_k_multiplier: int = 4  # candidate_k = max(20, multiplier * k)
    candidate_k_min: int = 20  # Minimum candidate pool size
    # Gate thresholds (Phase B)
    gate_min_tokens: int = 20
    gate_refusal_patterns: List[str] = field(default_factory=lambda: list(REFUSAL_PATTERNS))


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
    ):
        self.inner = inner_searcher
        self.generation_retriever = generation_retriever
        self.config = config
        self.original_question = original_question
        self._search_log: List[Dict[str, Any]] = []
        self._total_generation_time_s: float = 0.0
        self._total_generation_tokens: int = 0
        self._generation_call_count: int = 0

        # Separate file for full generation text (keeps trajectories small)
        self._gar_gen_path = None
        if gar_generations_path:
            self._gar_gen_path = Path(gar_generations_path)
            self._gar_gen_path.parent.mkdir(parents=True, exist_ok=True)

    def set_original_question(self, question: str):
        self.original_question = question

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """GAR search: retrieve + generate + (optionally gate) + fuse."""
        candidate_k = max(self.config.candidate_k_min, self.config.candidate_k_multiplier * k)

        # Step 1: Original retrieval
        r_orig = self.inner.search(query, candidate_k)

        # Step 2: Parametric generation
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

            # Write full generation text to separate file
            if self._gar_gen_path and gen_result["text"]:
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
            self._search_log.append(log_entry)
            return r_orig[:k]

        if self._is_refusal(gen_result["text"]):
            log_entry["gate_decision"] = False
            log_entry["gate_reason"] = "refusal_detected"
            self._search_log.append(log_entry)
            return r_orig[:k]

        if self.config.enable_gate:
            gate_pass, gate_reason = self._gate(gen_result)
            log_entry["gate_decision"] = gate_pass
            log_entry["gate_reason"] = gate_reason
            if not gate_pass:
                self._search_log.append(log_entry)
                return r_orig[:k]

        # Step 4: Expansion retrieval on compressed generation
        compressed = self._compress(gen_result["text"])
        r_gen = self.inner.search(compressed, candidate_k)

        log_entry["r_gen_count"] = len(r_gen)
        log_entry["r_gen_docids"] = [r["docid"] for r in r_gen[:k]]

        # Step 5: Weighted RRF fusion
        fused = self._rrf_fuse(r_orig, r_gen, k)

        log_entry["fused_count"] = len(fused)
        log_entry["fused_docids"] = [r["docid"] for r in fused]

        self._search_log.append(log_entry)
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

    def _generate(self, subquery: str) -> Optional[Dict[str, Any]]:
        """Call generation model and return result dict."""
        if self.generation_retriever is None:
            return None

        start = time.time()
        text = self.generation_retriever.generate_document(
            self.original_question, subquery
        )
        elapsed = time.time() - start

        self._generation_call_count += 1
        self._total_generation_time_s += elapsed

        if text is None:
            return None

        # Approximate token count from whitespace splitting
        token_count = len(text.split())
        self._total_generation_tokens += token_count

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

    # --- Logging accessors ---

    def get_gar_metadata(self) -> Dict[str, Any]:
        """Return GAR-specific metadata for trajectory persistence."""
        return {
            "gar_alpha": self.config.alpha,
            "gar_k_rrf": self.config.k_rrf,
            "gar_enable_gate": self.config.enable_gate,
            "total_generation_time_s": self._total_generation_time_s,
            "total_generation_tokens": self._total_generation_tokens,
            "generation_call_count": self._generation_call_count,
            "gar_subquery_log": self._search_log,
        }

    def reset_logs(self):
        """Reset per-question logs (call before each new question)."""
        self._search_log = []
        self._total_generation_time_s = 0.0
        self._total_generation_tokens = 0
        self._generation_call_count = 0
