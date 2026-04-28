"""Compute recall@k for FRAMES runs from existing trajectory JSONs.

No LLM, no re-eval — just set intersection between each trajectory's
retrieved_docids and the relevant rows derived from the GT's wiki_links.

Builds qrel_evidence.txt as a side effect (idempotent — skips if exists).

Usage:
    python scripts/compute_frames_recall.py runs/frames/BGE-M3/test150/glm-4.7-flash/seed0
"""
import json
import os
import sys
from urllib.parse import quote, unquote

GT_PATH = "/scratch/afw8937/efficient-search-agents/frames/data/frames-all-gt.jsonl"
URL_MAP = "/scratch/afw8937/efficient-search-agents/frames/index/frames_wiki_url_to_rowindex.json"
QREL_OUT = "topics-qrels/frames/qrel_evidence.txt"


def build_qrels_if_missing():
    if os.path.exists(QREL_OUT) and os.path.getsize(QREL_OUT) > 0:
        return
    print("Building qrel_evidence.txt (one-time, ~1 min)...", file=sys.stderr)

    qid_to_urls = {}
    unique_urls = set()
    with open(GT_PATH) as f:
        for line in f:
            o = json.loads(line)
            qid = str(o["query_id"])
            qid_to_urls[qid] = o.get("wiki_links", [])
            unique_urls.update(qid_to_urls[qid])
    print(f"  {len(qid_to_urls)} qids, {len(unique_urls)} unique URLs", file=sys.stderr)

    # Stream-parse url_map to avoid loading 863MB into memory (login node OOM-killer hits ~1-2GB).
    # Build set of "wanted" URL variants once, then scan file for matches.
    import ijson

    # The url_map (built from a Wikipedia dump) canonicalizes titles by:
    #   - converting spaces to %20 (NOT underscores)
    #   - percent-encoding everything else except letters/digits
    # GT URLs from FRAMES use underscores for spaces. So the canonical lookup is:
    #   1. take title after /wiki/
    #   2. unquote any existing percent-escapes
    #   3. replace _ with space
    #   4. re-quote with safe='' (encode everything)
    PREFIX = "https://en.wikipedia.org/wiki/"

    def canonicalize(url):
        if not url.startswith(PREFIX):
            return url
        title = url[len(PREFIX):]
        title = unquote(title).replace("_", " ")
        return PREFIX + quote(title, safe="")

    wanted_variants = {}  # variant -> canonical (any of the qid_to_urls entries)
    for url in unique_urls:
        wanted_variants[url] = url
        wanted_variants[unquote(url)] = url
        wanted_variants[quote(url, safe=":/_.~%")] = url
        wanted_variants[canonicalize(url)] = url

    print(f"  streaming {URL_MAP} for {len(wanted_variants)} URL variants ...", file=sys.stderr)
    canonical_to_rid = {}  # canonical_url -> row_id
    n_scanned = 0
    with open(URL_MAP) as f:
        for key, value in ijson.kvitems(f, ""):
            n_scanned += 1
            if key in wanted_variants:
                canonical_to_rid[wanted_variants[key]] = value
                if len(canonical_to_rid) == len(unique_urls):
                    break  # found them all
            if n_scanned % 5_000_000 == 0:
                print(f"    scanned {n_scanned:,} entries, found {len(canonical_to_rid)}/{len(unique_urls)}", file=sys.stderr)
    print(f"  found {len(canonical_to_rid)}/{len(unique_urls)} URLs after scanning {n_scanned:,} entries", file=sys.stderr)

    def lookup(url):
        return canonical_to_rid.get(url)

    # url_map values are LISTS of row_ids (each Wikipedia article is split into many passages).
    # Expand each (qid, url) into multiple qrel lines — one per passage row.
    n_lines = 0
    n_missing = 0
    qids_with_evidence = set()
    with open(QREL_OUT, "w") as f:
        for qid, urls in qid_to_urls.items():
            for u in urls:
                rids = lookup(u)
                if rids is None:
                    n_missing += 1
                    continue
                if isinstance(rids, (list, tuple)):
                    for rid in rids:
                        f.write(f"{qid} 0 {rid} 1\n")
                        n_lines += 1
                else:
                    f.write(f"{qid} 0 {rids} 1\n")
                    n_lines += 1
                qids_with_evidence.add(qid)
    print(
        f"  wrote {n_lines} qrel lines for {len(qids_with_evidence)}/{len(qid_to_urls)} qids; missing {n_missing} URLs",
        file=sys.stderr,
    )


def load_qrels_article_level():
    """Group passage row_ids by article (consecutive runs of row_ids per qid).

    Each Wikipedia article in the BGE-M3 index is split into many passages with
    consecutive row_ids. We treat all consecutive rows for a qid as one article.
    Article-level recall = # articles where the agent retrieved ≥1 passage / # relevant articles.
    """
    qrels_articles = {}  # qid -> list of sets, each set = passage row_ids of one article
    qid_to_rows = {}
    with open(QREL_OUT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, _ = parts
            qid_to_rows.setdefault(qid, []).append(int(docid))
    for qid, rows in qid_to_rows.items():
        rows.sort()
        # Group consecutive runs (article boundaries: gap > 1 means new article)
        articles = []
        current = [rows[0]]
        for r in rows[1:]:
            if r - current[-1] == 1:
                current.append(r)
            else:
                articles.append(set(current))
                current = [r]
        articles.append(set(current))
        qrels_articles[qid] = articles
    return qrels_articles


def extract_retrieved(traj_json):
    """Return set of retrieved docids (as strings) from a trajectory JSON."""
    out = set()
    rd = traj_json.get("retrieved_docids", [])
    if isinstance(rd, list):
        for item in rd:
            if isinstance(item, dict):
                # form: [{some_query_str: [docid, docid, ...]}, ...]
                for v in item.values():
                    if isinstance(v, list):
                        out.update(str(x) for x in v)
            elif isinstance(item, (str, int)):
                out.add(str(item))
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: compute_frames_recall.py <run_dir>", file=sys.stderr)
        sys.exit(1)
    run_dir = sys.argv[1]
    build_qrels_if_missing()
    qrels_articles = load_qrels_article_level()
    print(f"loaded article-level qrels for {len(qrels_articles)} qids", file=sys.stderr)

    n_files = 0
    n_with_qrel = 0
    recalls = []
    for fn in sorted(os.listdir(run_dir)):
        if not fn.endswith(".json"):
            continue
        n_files += 1
        try:
            j = json.load(open(os.path.join(run_dir, fn)))
        except Exception:
            continue
        qid = str(j.get("query_id", ""))
        articles = qrels_articles.get(qid)
        if not articles:
            continue
        n_with_qrel += 1
        retrieved = {int(x) for x in extract_retrieved(j) if str(x).lstrip("-").isdigit()}
        hits = sum(1 for art in articles if retrieved & art)
        recall = hits / len(articles)
        recalls.append(recall)

    avg_eligible = sum(recalls) / max(len(recalls), 1) * 100
    avg_all = sum(recalls) / max(n_files, 1) * 100
    print(f"\n=== {run_dir} ===")
    print(f"files: {n_files}, with qrel evidence: {n_with_qrel}")
    print(f"article-level avg recall (over qids with evidence): {avg_eligible:.1f}%")
    print(f"article-level avg recall (over all {n_files} files, missing-qrel = 0): {avg_all:.1f}%")


if __name__ == "__main__":
    main()
