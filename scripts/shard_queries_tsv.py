"""Split a queries TSV into N contiguous shards.

Writes ``{out_dir}/q_0.tsv`` .. ``q_{N-1}.tsv`` preserving input row order.
Matches the existing ``topics-qrels/bcp/bcp_10_shards/`` layout.

Example:
    python scripts/shard_queries_tsv.py \\
        --input topics-qrels/bcp/queries_test150.tsv \\
        --out-dir topics-qrels/bcp/bcp_test150_3_shards \\
        --num-shards 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def shard_tsv(src: Path, out_dir: Path, num_shards: int) -> list[int]:
    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(lines)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if num_shards > n:
        raise ValueError(f"--num-shards={num_shards} exceeds row count {n}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base, rem = divmod(n, num_shards)
    counts: list[int] = []
    pos = 0
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        shard = lines[pos : pos + size]
        (out_dir / f"q_{i}.tsv").write_text("\n".join(shard) + "\n", encoding="utf-8")
        counts.append(len(shard))
        pos += size
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source queries TSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for q_0.tsv .. q_{N-1}.tsv.",
    )
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    counts = shard_tsv(args.input, args.out_dir, args.num_shards)
    print(f"Read {args.input}: {sum(counts)} rows")
    print(f"Wrote {len(counts)} shards to {args.out_dir}")
    for i, c in enumerate(counts):
        print(f"  q_{i}.tsv: {c} rows")


if __name__ == "__main__":
    main()
