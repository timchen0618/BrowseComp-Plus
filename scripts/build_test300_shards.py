"""Build test300 shard files from the user-provided 300-qid list.

Reads `selected_tool_calls/queries_test300_qids.txt` (300 qids, user-provided random
sample from BCP-830) and produces:
- topics-qrels/bcp/queries_test300_qids.txt + queries_test300.tsv (full 300-qid set)
- topics-qrels/bcp/queries_test300_shardA_qids.txt + queries_test300_shardA.tsv (lines 1-150)
- topics-qrels/bcp/queries_test300_shardB_qids.txt + queries_test300_shardB.tsv (lines 151-300)

Shard A = first 150 qids in the user-provided file (no further randomization).
Shard B = last 150 qids.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_QIDS = REPO / "selected_tool_calls" / "queries_test300_qids.txt"
QUERIES_TSV = REPO / "topics-qrels" / "bcp" / "queries.tsv"
DST_DIR = REPO / "topics-qrels" / "bcp"


def main():
    qids = SRC_QIDS.read_text().strip().splitlines()
    assert len(qids) == 300, f"expected 300 qids, got {len(qids)}"

    qid_to_text = {}
    for line in QUERIES_TSV.read_text().splitlines():
        if "\t" not in line:
            continue
        qid, text = line.split("\t", 1)
        qid_to_text[qid.strip()] = text

    missing = [q for q in qids if q not in qid_to_text]
    assert not missing, f"qids missing from queries.tsv: {missing[:10]}"

    shard_a = qids[:150]
    shard_b = qids[150:]
    assert len(shard_a) == 150 and len(shard_b) == 150
    assert set(shard_a).isdisjoint(set(shard_b)), "shards must be disjoint"

    def write_pair(name: str, qid_list: list[str]):
        (DST_DIR / f"queries_{name}_qids.txt").write_text("\n".join(qid_list) + "\n")
        tsv_lines = [f"{q}\t{qid_to_text[q]}" for q in qid_list]
        (DST_DIR / f"queries_{name}.tsv").write_text("\n".join(tsv_lines) + "\n")

    write_pair("test300", qids)
    write_pair("test300_shardA", shard_a)
    write_pair("test300_shardB", shard_b)

    print(f"wrote {len(qids)} qids to test300, {len(shard_a)} to shardA, {len(shard_b)} to shardB")
    print(f"  shard A first/last: {shard_a[0]} ... {shard_a[-1]}")
    print(f"  shard B first/last: {shard_b[0]} ... {shard_b[-1]}")
    print(f"  disjoint: {set(shard_a).isdisjoint(set(shard_b))}")


if __name__ == "__main__":
    main()
