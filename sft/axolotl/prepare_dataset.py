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

Output line shape (one JSON object per line):
    {"messages": [
        {"role": "user",      "content": "<system+question merged>"},
        {"role": "assistant", "content": "...<tool_call>...</tool_call>"},
        {"role": "user",      "content": "<tool_response>...</tool_response>"},
        ...
    ]}

Loss masking happens in Axolotl (`roles_to_train: ["assistant"]`); this
script only reshapes data.

Train/val split: by default (`--split bcp-train680-test150`) examples are
assigned using the BrowseComp-Plus fixed split in
`topics-qrels/bcp/queries_train680.tsv` and `queries_test150.tsv` (see
`scripts/split_bcp_test150.py`). Each input row needs a `query_id` field or a
resolvable `source_file` trajectory containing `query_id`. Use `--split random`
for the previous fractional holdout behavior.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# BrowseComp-Plus fixed split (830 = 680 train + 150 test); see scripts/split_bcp_test150.py.
DEFAULT_BCP_QUERIES_TRAIN_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_train680.tsv"
DEFAULT_BCP_QUERIES_TEST_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries_test150.tsv"

SPLIT_RANDOM = "random"
SPLIT_BCP_TRAIN680_TEST150 = "bcp-train680-test150"
SPLIT_CHOICES = (SPLIT_RANDOM, SPLIT_BCP_TRAIN680_TEST150)


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
# name/arguments are rewritten. `gpt-oss` keeps the source trajectory format
# (produced by gpt-oss-120b: name=local_knowledge_base_retrieval, arg=user_query,
# reasoning as plain text). `qwen` rewrites to the Tongyi/Qwen format consumed
# by `search_agent/tongyi_utils/react_agent.py`: name=search, arg=query, with
# reasoning wrapped in <think>...</think> so the assistant turn looks like
# "<think>...</think>\n<tool_call>{...}</tool_call>".
TEMPLATE_GPT_OSS = "gpt-oss"
TEMPLATE_QWEN = "qwen"
TEMPLATE_CHOICES = (TEMPLATE_GPT_OSS, TEMPLATE_QWEN)

# Map (source_name, source_arg_key) -> (target_name, target_arg_key) for the
# qwen template. Source trajectories from gpt-oss-120b use
# local_knowledge_base_retrieval/user_query; Tongyi/Qwen expects search/query.
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

    For the `qwen` template the extracted text is wrapped in
    <think>...</think> so the downstream assistant turn matches the Tongyi
    format ("<think>...</think>\\n<tool_call>...</tool_call>").
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
    if template == TEMPLATE_QWEN:
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
        name, parsed_args = _rewrite_qwen_tool_call(name, parsed_args)

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
                                 <think>...</think> for the qwen template)
                                 to the current assistant buffer
      - `function_call`       -> append <tool_call>...</tool_call>, flush
                                 buffer as one assistant message
      - `function_call_output`-> flush any pending assistant buffer, then
                                 emit a user <tool_response>...</tool_response>
    """
    items = _parse_excerpt_items(excerpt)
    if not items:
        return []

    messages: List[Dict[str, str]] = []
    buf: List[str] = []
    # Qwen/Tongyi expects "<think>...</think>\n<tool_call>...</tool_call>"
    # on the same assistant turn, so use a single newline separator. The
    # gpt-oss template keeps the original blank-line separator for readability.
    sep = "\n" if template == TEMPLATE_QWEN else "\n\n"

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
            messages.append({"role": "user", "content": _fmt_tool_response(it)})
        # Unknown item types are ignored intentionally.

    flush_assistant()
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
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a selected-tool-calls JSONL with {source_file, excerpt, ...} rows.",
    )
    parser.add_argument(
        "--trajectory-folder",
        type=Path,
        required=True,
        help=(
            "Folder containing the source trajectory JSON files referenced "
            "by each record's `source_file` field."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sft/axolotl/data"),
        help="Directory to write train.jsonl (and val.jsonl if applicable).",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=SPLIT_BCP_TRAIN680_TEST150,
        help=(
            "How to form train vs val. "
            f"'{SPLIT_BCP_TRAIN680_TEST150}' uses topics-qrels/bcp queries_train680.tsv "
            "for train and queries_test150.tsv for val (BrowseComp-Plus fixed split). "
            f"'{SPLIT_RANDOM}' shuffles with --seed and holds out --val-size fraction."
        ),
    )
    parser.add_argument(
        "--queries-train-tsv",
        type=Path,
        default=DEFAULT_BCP_QUERIES_TRAIN_TSV,
        help=(
            f"With --split {SPLIT_BCP_TRAIN680_TEST150}: TSV whose first column lists "
            "training query_ids (default: repo topics-qrels/bcp/queries_train680.tsv)."
        ),
    )
    parser.add_argument(
        "--queries-test-tsv",
        type=Path,
        default=DEFAULT_BCP_QUERIES_TEST_TSV,
        help=(
            f"With --split {SPLIT_BCP_TRAIN680_TEST150}: TSV whose first column lists "
            "held-out query_ids written to val.jsonl (default: queries_test150.tsv)."
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
        help=f"With --split {SPLIT_RANDOM}: shuffle seed.",
    )
    parser.add_argument(
        "--template",
        choices=TEMPLATE_CHOICES,
        default=TEMPLATE_GPT_OSS,
        help=(
            "Output template. 'gpt-oss' preserves the source trajectory format "
            "(name=local_knowledge_base_retrieval, arg=user_query, reasoning as "
            "plain text). 'qwen' rewrites tool calls to name=search, arg=query "
            "and wraps reasoning in <think>...</think> to match the Tongyi/Qwen "
            "ReAct format expected by search_agent/tongyi_client.py."
        ),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"--input not found: {args.input}")
    if not args.trajectory_folder.is_dir():
        parser.error(
            f"--trajectory-folder not found or not a dir: {args.trajectory_folder}"
        )

    if args.split == SPLIT_BCP_TRAIN680_TEST150:
        if not args.queries_train_tsv.is_file():
            parser.error(f"--queries-train-tsv not found: {args.queries_train_tsv}")
        if not args.queries_test_tsv.is_file():
            parser.error(f"--queries-test-tsv not found: {args.queries_test_tsv}")

    source_cache = _SourceTrajectoryCache(args.trajectory_folder)

    raw_total = 0
    kept: List[Dict[str, Any]] = []
    dropped_schema = 0
    dropped_missing_source = 0
    dropped_bad_excerpt = 0

    for record in _iter_jsonl(args.input):
        raw_total += 1
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
        if args.split == SPLIT_BCP_TRAIN680_TEST150:
            qid = _query_id_from_record(record, source_cache)
            if qid is None:
                dropped_bad_excerpt += 1
                continue
            row["_query_id"] = qid
        kept.append(row)

    if not kept:
        raise SystemExit(
            f"No usable examples found in {args.input}. "
            f"dropped_schema={dropped_schema} "
            f"dropped_missing_source={dropped_missing_source} "
            f"dropped_bad_excerpt={dropped_bad_excerpt}"
        )

    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    dropped_split = 0

    if args.split == SPLIT_RANDOM:
        rng = random.Random(args.seed)
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
                f"[warn] --split {SPLIT_BCP_TRAIN680_TEST150}: "
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

    extra = ""
    if args.split == SPLIT_BCP_TRAIN680_TEST150 and dropped_split:
        extra = f" dropped_split={dropped_split}"

    print(
        "summary: "
        f"read={raw_total} kept={len(kept)} "
        f"dropped_schema={dropped_schema} "
        f"dropped_missing_source={dropped_missing_source} "
        f"dropped_bad_excerpt={dropped_bad_excerpt}"
        f"{extra}"
    )


if __name__ == "__main__":
    main()

    # python sft/axolotl/prepare_dataset.py \
    # --input selected_tool_calls/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl \
    # --trajectory-folder runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4 \
    # --output-dir sft/axolotl/data \
    # --template "qwen"
