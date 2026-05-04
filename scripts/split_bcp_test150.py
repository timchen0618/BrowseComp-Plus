"""Sample a test150/train680 split of the BrowseComp-Plus dataset.

Samples 150 query IDs at random (with a fixed seed for reproducibility) from
the 830 BrowseComp-Plus queries, and writes matching TSV query files under
`topics-qrels/bcp/` plus JSONL ground-truth files under `data/`.

Outputs:
    topics-qrels/bcp/queries_test150.tsv
    topics-qrels/bcp/queries_train680.tsv
    data/browsecomp_plus_decrypted_test150.jsonl
    data/browsecomp_plus_decrypted_train680.jsonl

Optionally also writes the query IDs of the sampled test examples when
``--test-qids-out`` is given (one query_id per line, in the same order as
they appear in the source queries TSV).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_TSV = REPO_ROOT / "topics-qrels" / "bcp" / "queries.tsv"
DATA_JSONL = REPO_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"


def read_queries_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, _, qtext = line.partition("\t")
            rows.append((qid, qtext))
    return rows


def write_queries_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, qtext in rows:
            f.write(f"{qid}\t{qtext}\n")


def split_jsonl_by_qid(
    src: Path, test_dst: Path, train_dst: Path, test_qids: set[str]
) -> tuple[int, int]:
    """Stream src jsonl once, routing each line to test_dst or train_dst.

    Returns ``(n_test, n_train)``. Lines whose query_id is in ``test_qids`` go
    to ``test_dst``; all other (non-empty) lines go to ``train_dst``. The
    existing raw line is copied verbatim (no JSON reserialization) to preserve
    the original content byte-for-byte and keep memory usage low.
    """

    test_dst.parent.mkdir(parents=True, exist_ok=True)
    train_dst.parent.mkdir(parents=True, exist_ok=True)
    n_test = 0
    n_train = 0
    with src.open("r", encoding="utf-8") as fin, \
            test_dst.open("w", encoding="utf-8") as ftest, \
            train_dst.open("w", encoding="utf-8") as ftrain:
        for line in fin:
            if not line.strip():
                continue
            # Cheap query_id extraction without full JSON parse of huge lines.
            try:
                qid = json.loads(line).get("query_id")
            except json.JSONDecodeError:
                continue
            qid = str(qid)
            if qid in test_qids:
                ftest.write(line if line.endswith("\n") else line + "\n")
                n_test += 1
            else:
                ftrain.write(line if line.endswith("\n") else line + "\n")
                n_train += 1
    return n_test, n_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries-tsv", type=Path, default=QUERIES_TSV)
    parser.add_argument("--data-jsonl", type=Path, default=DATA_JSONL)
    parser.add_argument(
        "--test-name",
        type=str,
        default="test150",
        help="Label appended to test split filenames (e.g. test150).",
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="train680",
        help="Label appended to train split filenames (e.g. train680).",
    )
    parser.add_argument(
        "--test-qids-out",
        "--test-indices-out",
        dest="test_qids_out",
        type=Path,
        default=None,
        help=(
            "Optional path to write the query IDs of sampled test examples, "
            "one per line, in the same order they appear in the source "
            "queries TSV. Pass 'auto' (or '-') to use a default path "
            "alongside the test TSV as 'queries_{test_name}_qids.txt'."
        ),
    )
    args = parser.parse_args()

    rows = read_queries_tsv(args.queries_tsv)
    total = len(rows)
    if args.test_size > total:
        raise ValueError(
            f"--test-size={args.test_size} exceeds available queries ({total})."
        )

    rng = random.Random(args.seed)
    indices = list(range(total))
    rng.shuffle(indices)
    test_idx = set(indices[: args.test_size])
    sorted_test_idx = sorted(test_idx)

    test_rows = [rows[i] for i in range(total) if i in test_idx]
    train_rows = [rows[i] for i in range(total) if i not in test_idx]

    test_qids = {qid for qid, _ in test_rows}
    train_qids = {qid for qid, _ in train_rows}

    test_tsv = args.queries_tsv.with_name(f"queries_{args.test_name}.tsv")
    train_tsv = args.queries_tsv.with_name(f"queries_{args.train_name}.tsv")
    write_queries_tsv(test_tsv, test_rows)
    write_queries_tsv(train_tsv, train_rows)

    data_stem = args.data_jsonl.stem
    test_jsonl = args.data_jsonl.with_name(f"{data_stem}_{args.test_name}.jsonl")
    train_jsonl = args.data_jsonl.with_name(f"{data_stem}_{args.train_name}.jsonl")
    n_test, n_train = split_jsonl_by_qid(
        args.data_jsonl, test_jsonl, train_jsonl, test_qids
    )

    print(f"Total queries in {args.queries_tsv.name}: {total}")
    print(f"Sampled with seed={args.seed}: test={len(test_rows)}, train={len(train_rows)}")
    print(f"Wrote {test_tsv} ({len(test_rows)} rows)")
    print(f"Wrote {train_tsv} ({len(train_rows)} rows)")
    print(f"Wrote {test_jsonl} ({n_test} records)")
    print(f"Wrote {train_jsonl} ({n_train} records)")

    qids_out = args.test_qids_out
    if qids_out is not None:
        if str(qids_out) in ("auto", "-"):
            qids_out = args.queries_tsv.with_name(
                f"queries_{args.test_name}_qids.txt"
            )
        qids_out.parent.mkdir(parents=True, exist_ok=True)
        with qids_out.open("w", encoding="utf-8") as f:
            for i in sorted_test_idx:
                f.write(f"{rows[i][0]}\n")
        print(f"Wrote {qids_out} ({len(sorted_test_idx)} query IDs)")

    if n_test != len(test_qids):
        print(
            f"WARNING: expected {len(test_qids)} test records in "
            f"{args.data_jsonl.name}, wrote {n_test}."
        )
    if n_train != len(train_qids):
        print(
            f"WARNING: expected {len(train_qids)} train records in "
            f"{args.data_jsonl.name}, wrote {n_train}."
        )


if __name__ == "__main__":
    main()
