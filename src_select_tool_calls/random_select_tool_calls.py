#!/usr/bin/env python3
"""Random baseline: same candidate tool indices as select_useful_tool_calls.py, random subset.

Supported --format values:
  result        Use result[] field (default). Works for any model.
  gpt-oss-120b  Use original_messages in OpenAI function_call format (gpt-oss-120b).
  glm           Use original_messages in GLM role/tool_calls format (GLM, minimax, qwen3.5).
  tongyi        Use original_messages in Tongyi <tool_call> XML format.

--use-original-messages is a deprecated alias for --format gpt-oss-120b.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Set

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from select_useful_tool_calls import (  # noqa: E402
    DEFAULT_TOOL_NAMES,
    build_full_excerpt,
    build_full_excerpt_om,
    find_candidate_tool_indices,
    find_candidate_tool_indices_om,
    parse_tool_names_arg,
)
from select_useful_tool_calls_glm import (  # noqa: E402
    build_full_excerpt_glm,
    find_candidate_tool_indices_glm,
)
from select_useful_tool_calls_tongyi import (  # noqa: E402
    build_full_excerpt_tongyi,
    find_candidate_tool_indices_tongyi,
)

FORMAT_RESULT = "result"
FORMAT_GPT_OSS = "gpt-oss-120b"
FORMAT_GLM = "glm"
FORMAT_TONGYI = "tongyi"
VALID_FORMATS = (FORMAT_RESULT, FORMAT_GPT_OSS, FORMAT_GLM, FORMAT_TONGYI)


def process_trajectory(
    traj: dict,
    *,
    source_file: str,
    line_query_id: str,
    k: int,
    seed: int,
    allowed_tool_names: Set[str],
    rng: random.Random,
    fmt: str,
) -> dict[str, Any]:
    qid_traj = str(traj.get("query_id", ""))
    if line_query_id and qid_traj and line_query_id != qid_traj:
        print(
            f"[warn] query_id mismatch JSONL={line_query_id!r} trajectory={qid_traj!r} "
            f"source_file={source_file!r}",
            file=sys.stderr,
        )

    if fmt in (FORMAT_GPT_OSS, FORMAT_GLM, FORMAT_TONGYI) and not traj.get("original_messages"):
        return {
            "query_id": qid_traj or line_query_id,
            "source_file": source_file,
            "error": "missing_original_messages",
            "selected_indices": [],
            "rationale": "",
            "excerpt": "",
            "candidates": [],
            "correct_num_selected": False,
            "k_requested": k,
            "k_effective": 0,
            "candidate_count": 0,
        }

    if fmt == FORMAT_GPT_OSS:
        candidates = find_candidate_tool_indices_om(traj, allowed_tool_names)
    elif fmt == FORMAT_GLM:
        candidates = find_candidate_tool_indices_glm(traj, allowed_tool_names)
    elif fmt == FORMAT_TONGYI:
        candidates = find_candidate_tool_indices_tongyi(traj, allowed_tool_names)
    else:
        candidates = find_candidate_tool_indices(traj, allowed_tool_names)

    k_eff = min(k, len(candidates))
    if k_eff == 0:
        return {
            "query_id": qid_traj or line_query_id,
            "source_file": source_file,
            "error": "no_candidate_tool_calls",
            "selected_indices": [],
            "rationale": "",
            "excerpt": "",
            "candidates": candidates,
            "correct_num_selected": False,
            "k_requested": k,
            "k_effective": 0,
            "candidate_count": len(candidates),
        }

    if k_eff < k:
        print(
            f"[warn] {source_file}: only {len(candidates)} candidate tool calls; using k={k_eff}",
            file=sys.stderr,
        )

    chosen = sorted(rng.sample(candidates, k_eff))

    if fmt == FORMAT_GPT_OSS:
        excerpt = build_full_excerpt_om(traj, chosen)
    elif fmt == FORMAT_GLM:
        excerpt = build_full_excerpt_glm(traj, chosen)
    elif fmt == FORMAT_TONGYI:
        excerpt = build_full_excerpt_tongyi(traj, chosen)
    else:
        excerpt = build_full_excerpt(traj, chosen)

    return {
        "query_id": qid_traj or line_query_id,
        "source_file": source_file,
        "selected_indices": chosen,
        "rationale": f"random_baseline seed={seed} k={k}",
        "excerpt": excerpt,
        "candidates": candidates,
        "correct_num_selected": len(chosen) == k_eff,
        "selection_method": "random",
        "seed": seed,
        "k_requested": k,
        "k_effective": k_eff,
        "candidate_count": len(candidates),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Randomly select k tool-call indices from trajectories; same candidates as select_useful_tool_calls.py.",
    )
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Directory containing trajectory JSON files named by source_file in each line",
    )
    ap.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Default: <input-stem>_random_seed<seed>.jsonl next to input",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default=FORMAT_RESULT,
        help=(
            "Which trajectory representation to use for candidate discovery and excerpts. "
            f"Choices: {', '.join(VALID_FORMATS)}. "
            "result=result[] field (any model); "
            "gpt-oss-120b=original_messages OpenAI function_call format; "
            "glm=original_messages role/tool_calls format (GLM, minimax, qwen3.5); "
            "tongyi=original_messages <tool_call> XML format (Tongyi, default tool: search)."
        ),
    )
    ap.add_argument(
        "--use-original-messages",
        action="store_true",
        help="Deprecated alias for --format gpt-oss-120b.",
    )
    ap.add_argument(
        "--tool-names",
        action="append",
        default=None,
        help=f"Comma-separated tool filter (repeatable). Default: {','.join(DEFAULT_TOOL_NAMES)}",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output-jsonl if it already exists",
    )
    args = ap.parse_args()

    # --use-original-messages is a deprecated alias
    fmt: str = args.format
    if args.use_original_messages:
        if fmt != FORMAT_RESULT:
            print(
                f"[warn] --use-original-messages conflicts with --format {fmt}; "
                f"using --format {FORMAT_GPT_OSS}",
                file=sys.stderr,
            )
        fmt = FORMAT_GPT_OSS
        print(
            "[warn] --use-original-messages is deprecated; use --format gpt-oss-120b instead.",
            file=sys.stderr,
        )

    traj_dir: Path = args.trajectory_dir
    if not traj_dir.is_dir():
        print(f"Not a directory: {traj_dir}", file=sys.stderr)
        sys.exit(1)

    inp: Path = args.input_jsonl
    if not inp.is_file():
        print(f"Not a file: {inp}", file=sys.stderr)
        sys.exit(1)

    if args.output_jsonl is not None:
        out_path = args.output_jsonl
    else:
        out_path = inp.parent / f"{inp.stem}_random_seed{args.seed}.jsonl"

    if out_path.exists() and not args.force:
        print(
            f"Output exists: {out_path} (use --force to overwrite)",
            file=sys.stderr,
        )
        sys.exit(1)

    allowed = parse_tool_names_arg(args.tool_names)
    rng = random.Random(args.seed)
    n_ok = 0
    n_err = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with inp.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[error] line {line_no}: invalid JSON: {e}", file=sys.stderr)
                sys.exit(1)

            source_file = row.get("source_file")
            if not source_file or not isinstance(source_file, str):
                print(f"[error] line {line_no}: missing source_file", file=sys.stderr)
                sys.exit(1)

            path = traj_dir / source_file
            if not path.is_file():
                print(
                    f"[error] line {line_no}: trajectory not found: {path}\n"
                    f"  Hint: use --trajectory-dir that matches the seed used when building this JSONL.",
                    file=sys.stderr,
                )
                sys.exit(1)

            with path.open(encoding="utf-8") as tf:
                traj = json.load(tf)

            qid_line = str(row.get("query_id", "") or "")
            out = process_trajectory(
                traj,
                source_file=source_file,
                line_query_id=qid_line,
                k=args.k,
                seed=args.seed,
                allowed_tool_names=allowed,
                rng=rng,
                fmt=fmt,
            )
            if "error" in out:
                n_err += 1
            else:
                n_ok += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"Wrote {out_path} ({n_ok} ok, {n_err} with error field)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
