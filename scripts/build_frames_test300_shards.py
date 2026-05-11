"""Build FRAMES test300 shard files.

Samples 300 qids from the 824-query FRAMES corpus using a fixed seed (independent
of FRAMES test150). Writes:
- topics-qrels/frames/queries_test300_qids.txt + queries_test300.tsv
- topics-qrels/frames/queries_test300_shardA_qids.txt + queries_test300_shardA.tsv
- topics-qrels/frames/queries_test300_shardB_qids.txt + queries_test300_shardB.tsv

Seed 11 chosen to match the spirit of BCP's test300 build (a fresh random sample
independent of the test150 slice, since test150 was the first 150 qids of queries.tsv).
"""

import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUERIES_TSV = REPO / "topics-qrels" / "frames" / "queries.tsv"
DST_DIR = REPO / "topics-qrels" / "frames"
SEED = 11
N = 300


def main():
    qid_to_text = {}
    for line in QUERIES_TSV.read_text().splitlines():
        if "\t" not in line:
            continue
        qid, text = line.split("\t", 1)
        qid_to_text[qid.strip()] = text
    all_qids = list(qid_to_text.keys())
    assert len(all_qids) >= N, f"only {len(all_qids)} qids available"

    rng = random.Random(SEED)
    sampled = rng.sample(all_qids, N)
    shard_a = sampled[:150]
    shard_b = sampled[150:]
    assert len(shard_a) == 150 and len(shard_b) == 150
    assert set(shard_a).isdisjoint(set(shard_b)), "shards must be disjoint"

    def write_pair(name: str, qid_list: list[str]):
        (DST_DIR / f"queries_{name}_qids.txt").write_text("\n".join(qid_list) + "\n")
        tsv_lines = [f"{q}\t{qid_to_text[q]}" for q in qid_list]
        (DST_DIR / f"queries_{name}.tsv").write_text("\n".join(tsv_lines) + "\n")

    write_pair("test300", sampled)
    write_pair("test300_shardA", shard_a)
    write_pair("test300_shardB", shard_b)

    print(f"FRAMES test300: seed={SEED}, sampled {N} from {len(all_qids)} qids")
    print(f"  shardA first/last: {shard_a[0]} / {shard_a[-1]}")
    print(f"  shardB first/last: {shard_b[0]} / {shard_b[-1]}")
    print(f"  disjoint: {set(shard_a).isdisjoint(set(shard_b))}")


if __name__ == "__main__":
    main()
