#!/usr/bin/env python3
"""
Convert a selected-tool-calls JSONL into the messages-style JSONL that
Axolotl's `type: chat_template` dataset expects.

Input line shape (e.g. selected_tool_calls/*.jsonl):
    {"source_file": "run_XXX.json",
     "excerpt": "<JSON item 1>\\n\\n<JSON item 2>\\n\\n...",
     ...}

where each JSON item in `excerpt` is an OpenAI Responses-API item of type
`reasoning`, `function_call`, or `function_call_output`.

The system + user prompt is pulled from the source trajectory at
`<trajectory_folder>/<source_file>` via its `original_messages[0]` entry
(a single merged `user` message containing both the system prompt and
the "User: <question>" line).

Single-input mode requires --eval-folder pointing to a directory of
*_eval.json files (same format as multi-input subsequent_eval_folder).
By default only successful trajectories (judge_result.correct == True)
are kept. Pass --keep-failed to include unsuccessful ones as well.
A hard error is raised if any record's query_id is absent from the eval
folder, regardless of --keep-failed.

Output line shape (one JSON object per line):
    {"messages": [
        {"role": "user",      "content": "<system+question merged>"},
        {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
        {"role": "user",      "content": "<tool_response>...</tool_response>"},
        ...
    ]}

Loss masking happens in Axolotl (`roles_to_train: ["assistant"]`); this
script only reshapes data.

Train/val split: by default (`--split bcp-train530-test300`) examples are
assigned using `topics-qrels/bcp/queries_train530.tsv` and `queries_test300.tsv`.
Use `--split bcp-train680-test150` for the alternative 680/150 split using
`queries_train680.tsv` and `queries_test150.tsv`. Each input row needs a
`query_id` field or a resolvable `source_file` trajectory containing `query_id`.
Use `--split random` for the previous fractional holdout behavior.

Multi-input mode (--multi-input):
    Instead of a single --input + --trajectory-folder pair, provide a JSON
    config file listing multiple (input, trajectory_folder, subsequent_folder,
    subsequent_eval_folder) tuples. Use --mode to choose how candidates for
    each query_id are selected across the multiple inputs:

      mode a: pick one candidate per query_id — the shortest successful run;
              if none succeed, the shortest run overall.

      mode b: include ALL successful candidates per query_id; if none succeed,
              include exactly one chosen at random.

      mode c: pick one candidate per query_id — the shortest successful run;
              if none succeed, include exactly one chosen at random.

      mode d: pick one candidate per query_id — the shortest successful run;
              if none succeed, skip that query entirely (no fallback).

    Config file format (list of objects):
      [
        {
          "input":                 "path/to/selected_tool_calls.jsonl",
          "trajectory_folder":     "path/to/source_trajectory_folder",
          "subsequent_folder":     "path/to/subsequent_run_folder",
          "subsequent_eval_folder": "path/to/subsequent_eval_folder"
        },
        ...
      ]

    * trajectory_folder     — source trajectories referenced by source_file
                              (used for original_messages / the prompt).
    * subsequent_folder     — trajectories produced when the agent continued
                              executing from the selected tool calls; used to
                              measure trajectory length.
    * subsequent_eval_folder — eval results (*_eval.json) for the subsequent
                              runs; used to determine success
                              (judge_result.correct). Optional: if omitted,
                              all candidates are treated as unsuccessful.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# BrowseComp-Plus fixed splits; see scripts/split_bcp_test150.py.
DEFAULT_BCP_QUERIES_TRAIN680_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_train680.tsv"
DEFAULT_BCP_QUERIES_TEST150_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_test150.tsv"
DEFAULT_BCP_QUERIES_TRAIN530_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_train530.tsv"
DEFAULT_BCP_QUERIES_TEST300_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_test300.tsv"

# Keep legacy alias for backward compatibility.
DEFAULT_BCP_QUERIES_TRAIN_TSV = DEFAULT_BCP_QUERIES_TRAIN680_TSV
DEFAULT_BCP_QUERIES_TEST_TSV = DEFAULT_BCP_QUERIES_TEST150_TSV

SPLIT_RANDOM = "random"
SPLIT_BCP_TRAIN680_TEST150 = "bcp-train680-test150"
SPLIT_BCP_TRAIN530_TEST300 = "bcp-train530-test300"
SPLIT_CHOICES = (SPLIT_RANDOM, SPLIT_BCP_TRAIN680_TEST150, SPLIT_BCP_TRAIN530_TEST300)

MODE_A = "a"
MODE_B = "b"
MODE_C = "c"
MODE_D = "d"
MODE_CHOICES = (MODE_A, MODE_B, MODE_C, MODE_D)

_INF_LENGTH: float = float("inf")


# ---------------------------------------------------------------------------
# Source-trajectory loading
# ---------------------------------------------------------------------------

class _SourceTrajectoryCache:
    """Tiny cache so repeated source_files aren't re-read from disk."""

    def __init__(self, trajectory_folder: Path) -> None:
        self.trajectory_folder = trajectory_folder
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def load(self, source_file: str) -> Optional[Dict[str, Any]]:
        if source_file in self._cache:
            return self._cache[source_file]
        path = self.trajectory_folder / source_file
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            self._cache[source_file] = None
            return None
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse source trajectory {path}: {e}")
            self._cache[source_file] = None
            return None
        self._cache[source_file] = data
        return data


def _load_query_ids_from_topics_tsv(path: Path) -> Set[str]:
    """First column of each line is query_id (tab-separated from question text)."""
    qids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            qid, _, _ = line.partition("\t")
            qid = qid.strip()
            if qid:
                qids.add(qid)
    return qids


def _query_id_from_record(
    record: Dict[str, Any], source_cache: _SourceTrajectoryCache
) -> Optional[str]:
    """Prefer JSONL `query_id`; else read from the source trajectory file."""
    raw = record.get("query_id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    src = record.get("source_file")
    if not src:
        return None
    traj = source_cache.load(str(src))
    if traj is None:
        return None
    q = traj.get("query_id")
    if q is None or not str(q).strip():
        return None
    return str(q).strip()


def _system_user_from_source(traj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    The first item of original_messages is a `{role: "user"}` entry whose
    content already concatenates the system prompt and the "User: <question>"
    line. We keep it as a single merged user message.
    """
    om = traj.get("original_messages")
    if not isinstance(om, list) or not om:
        return None
    first = om[0]
    if not isinstance(first, dict):
        return None
    role = str(first.get("role", "")).lower()
    content = first.get("content", "")
    if role != "user" or not isinstance(content, str) or not content.strip():
        return None
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Excerpt -> messages
# ---------------------------------------------------------------------------

def _parse_excerpt_items(excerpt: str) -> List[Dict[str, Any]]:
    """Split an excerpt string into a list of Responses-API item dicts."""
    items: List[Dict[str, Any]] = []
    for chunk in excerpt.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "type" in obj:
            items.append(obj)
    return items


# Template presets govern how reasoning text is emitted and how tool-call
# name/arguments are rendered.
#
#   gpt-oss   — keeps source trajectory format as-is (name=local_knowledge_base_retrieval,
#               arg=user_query, reasoning as plain text).
#
#   qwen      — legacy Tongyi/react_agent format: rewrites name→search, arg→query,
#               wraps reasoning in <think>...</think>, tool responses in role=user
#               with <tool_response> wrapper.
#
#   qwen-oss  — oss_client.py / Responses-API format for Qwen3.5 models:
#               keeps name=local_knowledge_base_retrieval / arg=user_query,
#               wraps reasoning in <think>...</think>, tool responses emitted as
#               role=tool with raw content (the Qwen3 chat template adds the
#               <tool_response> wrapper itself when tokenising).
TEMPLATE_GPT_OSS = "gpt-oss"
TEMPLATE_QWEN = "qwen"
TEMPLATE_QWEN_OSS = "qwen-oss"
TEMPLATE_CHOICES = (TEMPLATE_GPT_OSS, TEMPLATE_QWEN, TEMPLATE_QWEN_OSS)

# Map (source_name, source_arg_key) -> (target_name, target_arg_key) for the
# legacy `qwen` (Tongyi) template only.
_QWEN_TOOL_NAME_MAP = {
    "local_knowledge_base_retrieval": "search",
    "search": "search",
}
_QWEN_ARG_KEY_MAP = {
    "user_query": "query",
    "query": "query",
}


def _reasoning_text(item: Dict[str, Any], template: str) -> str:
    """Extract text from a Responses-API reasoning item.

    For the `qwen` and `qwen-oss` templates the text is wrapped in
    <think>...</think> so the assistant turn looks like
    "<think>...</think>\\n<tool_call>...</tool_call>".
    """
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for c in content:
        if isinstance(c, dict):
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t)
    text = "\n".join(parts).strip()
    if not text:
        return ""
    if template in (TEMPLATE_QWEN, TEMPLATE_QWEN_OSS):
        return f"<think>\n{text}\n</think>"
    return text


def _rewrite_qwen_tool_call(name: str, parsed_args: Any) -> Tuple[str, Any]:
    """Rewrite a tool-call name/arguments dict into the Qwen/Tongyi schema."""
    new_name = _QWEN_TOOL_NAME_MAP.get(name, name)
    if isinstance(parsed_args, dict):
        new_args: Dict[str, Any] = {}
        for k, v in parsed_args.items():
            new_args[_QWEN_ARG_KEY_MAP.get(k, k)] = v
        return new_name, new_args
    return new_name, parsed_args


def _fmt_tool_call(item: Dict[str, Any], template: str) -> str:
    """Render a function_call item as an inline <tool_call>...</tool_call>."""
    name = item.get("name", "")
    raw_args = item.get("arguments", "")
    if isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = raw_args
    else:
        parsed_args = raw_args

    if template == TEMPLATE_QWEN:
        # Legacy Tongyi format: rewrite to search/query.
        name, parsed_args = _rewrite_qwen_tool_call(name, parsed_args)
    # qwen-oss: keep name/args exactly as they appear in the source trajectory
    # (local_knowledge_base_retrieval / user_query), matching oss_client.py.

    payload = {"name": name, "arguments": parsed_args}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def _fmt_tool_response(item: Dict[str, Any]) -> str:
    """Render a function_call_output item as a <tool_response> user turn."""
    out = item.get("output", "")
    if not isinstance(out, str):
        out = json.dumps(out, ensure_ascii=False)
    return "<tool_response>\n" + out + "\n</tool_response>"


def _excerpt_to_messages(excerpt: str, template: str) -> List[Dict[str, str]]:
    """
    Walk the Responses-API items into inline chat messages.

    Rules:
      - `reasoning`           -> append text (optionally wrapped in
                                 <think>...</think> for qwen/qwen-oss)
                                 to the current assistant buffer
      - `function_call`       -> append <tool_call>...</tool_call>, flush
                                 buffer as one assistant message
      - `function_call_output`-> flush any pending assistant buffer, then:
                                   qwen-oss: role=tool, content=raw output
                                             (Qwen3 chat template adds the
                                             <tool_response> wrapper itself)
                                   qwen:     role=user, content=<tool_response>
                                   gpt-oss:  role=user, content=<tool_response>
    """
    items = _parse_excerpt_items(excerpt)
    if not items:
        return []

    messages: List[Dict[str, str]] = []
    buf: List[str] = []
    # qwen / qwen-oss: compact single-newline separator so the assistant turn
    # looks like "<think>...</think>\n<tool_call>...</tool_call>".
    # gpt-oss: blank-line separator for readability.
    sep = "\n" if template in (TEMPLATE_QWEN, TEMPLATE_QWEN_OSS) else "\n\n"

    def flush_assistant() -> None:
        if not buf:
            return
        text = sep.join(s for s in buf if s).strip()
        buf.clear()
        if text:
            messages.append({"role": "assistant", "content": text})

    for it in items:
        kind = it.get("type")
        if kind == "reasoning":
            text = _reasoning_text(it, template)
            if text:
                buf.append(text)
        elif kind == "function_call":
            buf.append(_fmt_tool_call(it, template))
            flush_assistant()
        elif kind == "function_call_output":
            flush_assistant()
            if template == TEMPLATE_QWEN_OSS:
                out = it.get("output", "")
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)
                messages.append({"role": "tool", "content": out})
            else:
                messages.append({"role": "user", "content": _fmt_tool_response(it)})
        # Unknown item types are ignored intentionally.

    flush_assistant()

    # Axolotl trains only on assistant turns; a trailing tool-result turn
    # causes the "Last turn is not trainable" warning and incorrect EOS masking.
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()

    return messages


def _coerce_excerpt(
    example: Dict[str, Any],
    source_cache: _SourceTrajectoryCache,
    template: str,
) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """
    Build messages for a (source_file, excerpt) record.

    Returns (messages | None, reason). `reason` is one of:
      "ok", "schema", "missing_source", "bad_excerpt".
    """
    if "source_file" not in example or "excerpt" not in example:
        return None, "schema"

    traj = source_cache.load(str(example["source_file"]))
    if traj is None:
        return None, "missing_source"

    prompt = _system_user_from_source(traj)
    if prompt is None:
        return None, "missing_source"

    excerpt_msgs = _excerpt_to_messages(str(example["excerpt"]), template)
    if not excerpt_msgs:
        return None, "bad_excerpt"

    has_tool_call = any(
        m["role"] == "assistant" and "<tool_call>" in m["content"]
        for m in excerpt_msgs
    )
    if not has_tool_call:
        return None, "bad_excerpt"

    return [prompt] + excerpt_msgs, "ok"


# ---------------------------------------------------------------------------
# Multi-input: subsequent-execution helpers and candidate selection
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _MultiInputSpec:
    """One entry from the --multi-input JSON config."""
    input_path: Path
    trajectory_folder: Path
    subsequent_folder: Path
    subsequent_eval_folder: Optional[Path]


@dataclasses.dataclass
class _Candidate:
    """A selected-tool-calls record with its associated metadata."""
    record: Dict[str, Any]
    source_cache: _SourceTrajectoryCache
    success: bool
    subsequent_length: float  # float so we can use inf for unknown


def _load_subsequent_folder(path: Path) -> Dict[str, int]:
    """Scan a folder of trajectory JSONs. Return {query_id: result_length}."""
    mapping: Dict[str, int] = {}
    for p in sorted(path.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        qid = data.get("query_id")
        if not qid:
            continue
        qid = str(qid).strip()
        if qid in mapping:
            print(f"[warn] duplicate query_id {qid!r} in {path}, using first seen")
            continue
        mapping[qid] = len(data.get("result", []))
    return mapping


def _load_subsequent_eval_folder(path: Path) -> Dict[str, bool]:
    """Scan eval folder for *_eval.json files. Return {query_id: correct}."""
    mapping: Dict[str, bool] = {}
    for p in sorted(path.glob("*_eval.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        qid = data.get("query_id")
        if not qid:
            continue
        qid = str(qid).strip()
        if qid in mapping:
            print(f"[warn] duplicate query_id {qid!r} in {path}, using first seen")
            continue
        correct = bool((data.get("judge_result") or {}).get("correct", False))
        mapping[qid] = correct
    return mapping


def _load_multi_input_specs(config_path: Path) -> List[_MultiInputSpec]:
    with config_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"--multi-input JSON must be a list, got {type(entries).__name__}")
    specs: List[_MultiInputSpec] = []
    for i, entry in enumerate(entries):
        input_path = Path(entry["input"])
        traj_folder = Path(entry["trajectory_folder"])
        subseq_folder = Path(entry["subsequent_folder"])
        eval_folder = Path(entry["subsequent_eval_folder"]) if entry.get("subsequent_eval_folder") else None
        specs.append(_MultiInputSpec(input_path, traj_folder, subseq_folder, eval_folder))
    return specs


def _select_mode_a(candidates: List[_Candidate]) -> Optional[_Candidate]:
    """Setting A: shortest successful run; if none succeed, shortest run overall."""
    if not candidates:
        return None
    successful = [c for c in candidates if c.success]
    pool = successful if successful else candidates
    return min(pool, key=lambda c: c.subsequent_length)


def _select_mode_b(candidates: List[_Candidate], rng: random.Random) -> List[_Candidate]:
    """Setting B: all successful runs; if none succeed, exactly one at random."""
    if not candidates:
        return []
    successful = [c for c in candidates if c.success]
    if successful:
        return successful
    return [rng.choice(candidates)]


def _select_mode_c(candidates: List[_Candidate], rng: random.Random) -> Optional[_Candidate]:
    """Setting C: shortest successful run; if none succeed, exactly one at random."""
    if not candidates:
        return None
    successful = [c for c in candidates if c.success]
    if successful:
        return min(successful, key=lambda c: c.subsequent_length)
    return rng.choice(candidates)


def _select_mode_d(candidates: List[_Candidate]) -> Optional[_Candidate]:
    """Setting D: shortest successful run; if none succeed, skip entirely (no fallback)."""
    if not candidates:
        return None
    successful = [c for c in candidates if c.success]
    if not successful:
        return None
    return min(successful, key=lambda c: c.subsequent_length)


def _build_candidates_per_query(specs: List[_MultiInputSpec]) -> Dict[str, List[_Candidate]]:
    """Load all input specs and group candidates by query_id."""
    grouped: Dict[str, List[_Candidate]] = {}
    for spec in specs:
        if not spec.input_path.is_file():
            print(f"[warn] input not found: {spec.input_path}, skipping")
            continue
        if not spec.trajectory_folder.is_dir():
            print(f"[warn] trajectory_folder not found: {spec.trajectory_folder}, skipping")
            continue
        if not spec.subsequent_folder.is_dir():
            print(f"[warn] subsequent_folder not found: {spec.subsequent_folder}, skipping")
            continue

        source_cache = _SourceTrajectoryCache(spec.trajectory_folder)
        subseq_lengths = _load_subsequent_folder(spec.subsequent_folder)

        eval_map: Dict[str, bool] = {}
        if spec.subsequent_eval_folder is not None:
            if spec.subsequent_eval_folder.is_dir():
                eval_map = _load_subsequent_eval_folder(spec.subsequent_eval_folder)
            else:
                print(f"[warn] subsequent_eval_folder not found: {spec.subsequent_eval_folder}")

        for record in _iter_jsonl(spec.input_path):
            qid = record.get("query_id")
            if qid is None or not str(qid).strip():
                src = record.get("source_file")
                if src:
                    traj = source_cache.load(str(src))
                    if traj:
                        qid = traj.get("query_id")
            if not qid or not str(qid).strip():
                continue
            qid = str(qid).strip()

            success = eval_map.get(qid, False)
            length = float(subseq_lengths.get(qid, _INF_LENGTH))

            candidate = _Candidate(
                record=record,
                source_cache=source_cache,
                success=success,
                subsequent_length=length,
            )
            grouped.setdefault(qid, []).append(candidate)

    return grouped


# ---------------------------------------------------------------------------
# Multi-input statistics
# ---------------------------------------------------------------------------

def _print_multi_input_stats(grouped: Dict[str, List[_Candidate]], mode: str) -> None:
    """Report success-count distribution and average trajectory lengths.

    Length averages reflect the actual selection logic for the given mode:
      mode a — shortest successful (>=1 success) / shortest overall (0 success)
      mode b — avg of ALL successful (>=1 success) / avg of ALL candidates (0 success,
               since the random pick has expected length = mean of the pool)
      mode c — shortest successful (>=1 success) / avg of all candidates (0 success)
      mode d — shortest successful (>=1 success) / skipped entirely (0 success)
    """
    success_counts: Counter = Counter()
    lengths_with_success: List[float] = []
    lengths_without_success: List[float] = []

    for candidates in grouped.values():
        n_success = sum(1 for c in candidates if c.success)
        success_counts[n_success] += 1

        finite = [c for c in candidates if c.subsequent_length != _INF_LENGTH]
        successful = [c for c in finite if c.success]
        if successful:
            if mode == MODE_B:  # all successful are included
                lengths_with_success.extend(c.subsequent_length for c in successful)
            else:  # mode a/c/d: shortest successful
                lengths_with_success.append(min(c.subsequent_length for c in successful))
        elif finite and mode != MODE_D:  # mode d skips queries with 0 success
            if mode == MODE_A:  # shortest overall
                lengths_without_success.append(min(c.subsequent_length for c in finite))
            else:  # mode b/c: random pick → expected length = mean of pool
                lengths_without_success.extend(c.subsequent_length for c in finite)

    max_n = max(success_counts) if success_counts else 0
    print("\n--- Multi-input statistics ---")
    print("Success count distribution (per query_id):")
    for n in range(max_n + 1):
        count = success_counts.get(n, 0)
        print(f"  {n} success: {count} queries")

    success_label = "avg of all successful" if mode == MODE_B else "shortest successful"
    if lengths_with_success:
        avg = sum(lengths_with_success) / len(lengths_with_success)
        print(
            f"\nQueries with >=1 success ({sum(1 for cs in grouped.values() if any(c.success for c in cs))} queries):\n"
            f"  avg trajectory length ({success_label}): {avg:.1f}"
        )
    else:
        print("\nQueries with >=1 success: none")

    n_fail = sum(1 for cs in grouped.values() if not any(c.success for c in cs))
    if mode == MODE_D:
        print(f"\nQueries with 0 success ({n_fail} queries): skipped (mode d)")
    elif lengths_without_success:
        avg = sum(lengths_without_success) / len(lengths_without_success)
        fail_label = "shortest overall" if mode == MODE_A else "expected (avg of all candidates)"
        print(
            f"\nQueries with 0 success ({n_fail} queries):\n"
            f"  avg trajectory length ({fail_label}): {avg:.1f}"
        )
    else:
        print("\nQueries with 0 success: none")
    print("---\n")


# ---------------------------------------------------------------------------
# I/O helpers & driver
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{line_num} could not parse: {e}")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- single-input mode args (original) ----
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a selected-tool-calls JSONL with {source_file, excerpt, ...} rows.",
    )
    parser.add_argument(
        "--trajectory-folder",
        type=Path,
        default=None,
        help=(
            "Folder containing the source trajectory JSON files referenced "
            "by each record's `source_file` field."
        ),
    )
    parser.add_argument(
        "--eval-folder",
        type=Path,
        default=None,
        help=(
            "Required with --input. Directory of *_eval.json files "
            "(judge_result.correct) used to determine trajectory success. "
            "By default only successful trajectories are kept; pass "
            "--keep-failed to include unsuccessful ones. A hard error is "
            "raised if any record's query_id is absent from this folder."
        ),
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        default=False,
        help=(
            "With --input: keep trajectories that are not successful "
            "(judge_result.correct == False). Missing query_ids still raise "
            "an error even with this flag."
        ),
    )

    # ---- multi-input mode args (new) ----
    parser.add_argument(
        "--multi-input",
        type=Path,
        default=None,
        help=(
            "Path to a JSON config file listing multiple input specs for "
            "multi-trajectory selection. Each entry must have: input, "
            "trajectory_folder, subsequent_folder, and optionally "
            "subsequent_eval_folder. Mutually exclusive with --input."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=None,
        help=(
            "Candidate selection mode for --multi-input. "
            "'a': one per query_id — shortest successful run, else shortest overall. "
            "'b': all successful runs per query_id, else one at random. "
            "'c': one per query_id — shortest successful run, else one at random. "
            "'d': one per query_id — shortest successful run; skip if none succeed."
        ),
    )

    # ---- shared args ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sft/axolotl/data"),
        help="Directory to write train.jsonl (and val.jsonl if applicable).",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=SPLIT_BCP_TRAIN530_TEST300,
        help=(
            "How to form train vs val. "
            f"'{SPLIT_BCP_TRAIN680_TEST150}' uses queries_train680.tsv / queries_test150.tsv. "
            f"'{SPLIT_BCP_TRAIN530_TEST300}' uses queries_train530.tsv / queries_test300.tsv. "
            f"'{SPLIT_RANDOM}' shuffles with --seed and holds out --val-size fraction."
        ),
    )
    parser.add_argument(
        "--queries-train-tsv",
        type=Path,
        default=None,
        help=(
            "TSV whose first column lists training query_ids. "
            f"Defaults to queries_train680.tsv for '{SPLIT_BCP_TRAIN680_TEST150}' and "
            f"queries_train530.tsv for '{SPLIT_BCP_TRAIN530_TEST300}'."
        ),
    )
    parser.add_argument(
        "--queries-test-tsv",
        type=Path,
        default=None,
        help=(
            "TSV whose first column lists held-out query_ids written to val.jsonl. "
            f"Defaults to queries_test150.tsv for '{SPLIT_BCP_TRAIN680_TEST150}' and "
            f"queries_test300.tsv for '{SPLIT_BCP_TRAIN530_TEST300}'."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help=f"With --split {SPLIT_RANDOM}: fraction held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=f"With --split {SPLIT_RANDOM}: shuffle seed. Also used for --mode b random selection.",
    )
    parser.add_argument(
        "--template",
        choices=TEMPLATE_CHOICES,
        default=TEMPLATE_QWEN_OSS,
        help=(
            "Output template. "
            "'qwen-oss' (default) targets Qwen3.5 models run via oss_client.py: "
            "keeps name=local_knowledge_base_retrieval / arg=user_query, wraps "
            "reasoning in <think>...</think>, tool responses as role=tool with raw "
            "content (the Qwen3 chat template adds <tool_response> itself). "
            "'gpt-oss' preserves the source trajectory format as-is with no "
            "reasoning wrapping. "
            "'qwen' is the legacy Tongyi/react_agent format: rewrites tool calls "
            "to name=search / arg=query with role=user <tool_response> wrapper."
        ),
    )
    args = parser.parse_args()

    # ---- validate mode choice ----
    multi_mode = args.multi_input is not None
    single_mode = args.input is not None

    if multi_mode and single_mode:
        parser.error("--multi-input and --input are mutually exclusive.")
    if not multi_mode and not single_mode:
        parser.error("Provide either --input (single-input mode) or --multi-input (multi-input mode).")

    if multi_mode and args.mode is None:
        parser.error("--mode {a,b,c,d} is required with --multi-input.")
    if single_mode and args.trajectory_folder is None:
        parser.error("--trajectory-folder is required with --input.")
    if single_mode and args.eval_folder is None:
        parser.error("--eval-folder is required with --input.")
    if multi_mode and args.eval_folder is not None:
        parser.error("--eval-folder is only valid with --input, not --multi-input.")
    if multi_mode and args.keep_failed:
        parser.error("--keep-failed is only valid with --input, not --multi-input.")

    if args.split != SPLIT_RANDOM:
        if args.queries_train_tsv is None:
            if args.split == SPLIT_BCP_TRAIN530_TEST300:
                args.queries_train_tsv = DEFAULT_BCP_QUERIES_TRAIN530_TSV
            else:
                args.queries_train_tsv = DEFAULT_BCP_QUERIES_TRAIN680_TSV
        if args.queries_test_tsv is None:
            if args.split == SPLIT_BCP_TRAIN530_TEST300:
                args.queries_test_tsv = DEFAULT_BCP_QUERIES_TEST300_TSV
            else:
                args.queries_test_tsv = DEFAULT_BCP_QUERIES_TEST150_TSV
        if not args.queries_train_tsv.is_file():
            parser.error(f"--queries-train-tsv not found: {args.queries_train_tsv}")
        if not args.queries_test_tsv.is_file():
            parser.error(f"--queries-test-tsv not found: {args.queries_test_tsv}")

    rng = random.Random(args.seed)

    # ---- collect kept examples ----
    kept: List[Dict[str, Any]] = []
    dropped_schema = 0
    dropped_missing_source = 0
    dropped_bad_excerpt = 0

    if single_mode:
        # ---- single-input path ----
        if not args.input.is_file():
            parser.error(f"--input not found: {args.input}")
        if not args.trajectory_folder.is_dir():
            parser.error(
                f"--trajectory-folder not found or not a dir: {args.trajectory_folder}"
            )
        if not args.eval_folder.is_dir():
            parser.error(f"--eval-folder not found or not a dir: {args.eval_folder}")

        eval_map = _load_subsequent_eval_folder(args.eval_folder)
        source_cache = _SourceTrajectoryCache(args.trajectory_folder)
        raw_total = 0
        dropped_not_success = 0

        for record in _iter_jsonl(args.input):
            raw_total += 1

            # Resolve query_id early — needed for eval lookup.
            qid = record.get("query_id")
            if qid is None or not str(qid).strip():
                src = record.get("source_file")
                if src:
                    traj = source_cache.load(str(src))
                    if traj:
                        qid = traj.get("query_id")
            if not qid or not str(qid).strip():
                dropped_schema += 1
                continue
            qid = str(qid).strip()

            if qid not in eval_map:
                raise SystemExit(
                    f"[error] query_id {qid!r} not found in --eval-folder {args.eval_folder}. "
                    "Ensure the eval folder contains a matching *_eval.json for every record."
                )

            if not args.keep_failed and not eval_map[qid]:
                dropped_not_success += 1
                continue

            messages, reason = _coerce_excerpt(record, source_cache, args.template)
            if messages is None:
                if reason == "schema":
                    dropped_schema += 1
                elif reason == "missing_source":
                    dropped_missing_source += 1
                else:
                    dropped_bad_excerpt += 1
                continue
            row: Dict[str, Any] = {"messages": messages}
            if args.split != SPLIT_RANDOM:
                row["_query_id"] = qid
            kept.append(row)

        print(
            f"single-input: read={raw_total} kept={len(kept)} "
            f"dropped_not_success={dropped_not_success} "
            f"dropped_schema={dropped_schema} "
            f"dropped_missing_source={dropped_missing_source} "
            f"dropped_bad_excerpt={dropped_bad_excerpt}"
        )

    else:
        # ---- multi-input path ----
        if not args.multi_input.is_file():
            parser.error(f"--multi-input not found: {args.multi_input}")

        specs = _load_multi_input_specs(args.multi_input)
        grouped = _build_candidates_per_query(specs)
        _print_multi_input_stats(grouped, args.mode)

        total_qids = len(grouped)
        selected_count = 0

        for qid, candidates in grouped.items():
            if args.mode == MODE_A:
                chosen = _select_mode_a(candidates)
                chosen_list = [chosen] if chosen is not None else []
            elif args.mode == MODE_C:
                chosen = _select_mode_c(candidates, rng)
                chosen_list = [chosen] if chosen is not None else []
            elif args.mode == MODE_D:
                chosen = _select_mode_d(candidates)
                chosen_list = [chosen] if chosen is not None else []
            else:
                chosen_list = _select_mode_b(candidates, rng)

            for candidate in chosen_list:
                messages, reason = _coerce_excerpt(
                    candidate.record, candidate.source_cache, args.template
                )
                if messages is None:
                    if reason == "schema":
                        dropped_schema += 1
                    elif reason == "missing_source":
                        dropped_missing_source += 1
                    else:
                        dropped_bad_excerpt += 1
                    continue
                row = {"messages": messages}
                if args.split != SPLIT_RANDOM:
                    row["_query_id"] = qid
                kept.append(row)
                selected_count += 1

        n_success = sum(
            1 for candidates in grouped.values() if any(c.success for c in candidates)
        )
        n_skipped_no_success = (
            sum(1 for candidates in grouped.values() if not any(c.success for c in candidates))
            if args.mode == MODE_D else 0
        )
        print(
            f"multi-input (mode={args.mode}): "
            f"total_query_ids={total_qids} "
            f"query_ids_with_success={n_success} "
            + (f"query_ids_skipped_no_success={n_skipped_no_success} " if args.mode == MODE_D else "")
            + f"selected_examples={selected_count} "
            f"dropped_schema={dropped_schema} "
            f"dropped_missing_source={dropped_missing_source} "
            f"dropped_bad_excerpt={dropped_bad_excerpt}"
        )

    if not kept:
        raise SystemExit(
            "No usable examples found. "
            f"dropped_schema={dropped_schema} "
            f"dropped_missing_source={dropped_missing_source} "
            f"dropped_bad_excerpt={dropped_bad_excerpt}"
        )

    # ---- train/val split ----
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    dropped_split = 0

    if args.split == SPLIT_RANDOM:
        rng.shuffle(kept)
        n_val = int(round(len(kept) * args.val_size)) if args.val_size > 0 else 0
        val, train = kept[:n_val], kept[n_val:]
    else:
        train_qids = _load_query_ids_from_topics_tsv(args.queries_train_tsv)
        test_qids = _load_query_ids_from_topics_tsv(args.queries_test_tsv)
        train = []
        val = []
        for row in kept:
            qid = row.pop("_query_id")
            if qid in test_qids:
                val.append(row)
            elif qid in train_qids:
                train.append(row)
            else:
                dropped_split += 1
        if dropped_split:
            print(
                f"[warn] --split {args.split}: "
                f"dropped {dropped_split} examples whose query_id is not in "
                f"{args.queries_train_tsv.name} or {args.queries_test_tsv.name}"
            )

    train_path = args.output_dir / "train.jsonl"
    _write_jsonl(train_path, train)
    print(f"wrote {len(train):>6} -> {train_path}")

    n_val = len(val)
    if n_val > 0:
        val_path = args.output_dir / "val.jsonl"
        _write_jsonl(val_path, val)
        print(f"wrote {len(val):>6} -> {val_path}")


if __name__ == "__main__":
    main()

    # Single-input example:
    # python sft/axolotl/prepare_dataset.py \
    # --input selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
    # --trajectory-folder runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    # --output-dir sft/axolotl/data/raw/data_qwen_oss \
    # --template "qwen-oss"

    # Multi-input example (mode a):
    # python sft/axolotl/prepare_dataset.py \
    # --multi-input sft/axolotl/multi_input_config.json \
    # --mode a \
    # --output-dir sft/axolotl/data/raw/data_qwen_oss \
    # --template "qwen-oss"
    #
    # multi_input_config.json:
    # [
    #   {
    #     "input": "selected_tool_calls/test150/gpt-oss-120b/seed4.jsonl",
    #     "trajectory_folder": "runs/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b/seed4",
    #     "subsequent_folder": "runs/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b/subsequent_seed4",
    #     "subsequent_eval_folder": "evals/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b/subsequent_seed4"
    #   }
    # ]
