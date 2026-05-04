######################################################################
# Paired bootstrap comparison from two evaluation_summary.json files   #
#                                                                      #
# Based on paired bootstrap by Graham Neubig / Mathias Müller;         #
# this variant only resamples pre-computed per-query scores (no        #
# gold/system string evaluation).                                      #
#                                                                      #
# See: Statistical Significance Tests for Machine Translation          #
# Evaluation, Philipp Koehn — http://www.aclweb.org/anthology/W04-3250 #
######################################################################

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Any

METRIC_CORRECT = "correct"
METRIC_RECALL = "recall"

METRICS = [METRIC_CORRECT, METRIC_RECALL]


def _metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == METRIC_CORRECT:
        v = row["correct"]
        return 1.0 if v else 0.0
    if metric == METRIC_RECALL:
        return float(row["recall"])
    raise ValueError(f"Unknown metric: {metric}")


def load_query_scores(path: str, metric: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("per_query_metrics")
    if not rows:
        raise ValueError(f"{path}: missing or empty 'per_query_metrics'")
    out: dict[str, float] = {}
    for row in rows:
        qid = str(row["query_id"])
        out[qid] = _metric_value(row, metric)
    return out


def align_paired_scores(
    path_a: str,
    path_b: str,
    metric: str,
    strict: bool,
) -> tuple[list[float], list[float], list[str]]:
    da = load_query_scores(path_a, metric)
    db = load_query_scores(path_b, metric)
    ids_a, ids_b = set(da), set(db)
    common = ids_a & ids_b
    if strict:
        if ids_a != ids_b:
            only_a = sorted(ids_a - ids_b)
            only_b = sorted(ids_b - ids_a)
            raise ValueError(
                f"Query ID mismatch between summaries (strict mode).\n"
                f"  Only in A ({len(only_a)}): {only_a[:20]}{'...' if len(only_a) > 20 else ''}\n"
                f"  Only in B ({len(only_b)}): {only_b[:20]}{'...' if len(only_b) > 20 else ''}"
            )
    else:
        if ids_a != ids_b:
            print(
                f"Warning: aligning on intersection only "
                f"({len(common)} queries; "
                f"A-only {len(ids_a - ids_b)}, B-only {len(ids_b - ids_a)})."
            )
    ordered = sorted(common)
    sa = [da[i] for i in ordered]
    sb = [db[i] for i in ordered]
    return sa, sb, ordered


def paired_bootstrap_from_scores(
    scores_a: list[float],
    scores_b: list[float],
    num_samples: int = 10000,
    sample_ratio: float = 1.0,
    seed: int | None = None,
) -> None:
    assert len(scores_a) == len(scores_b)
    n = len(scores_a)
    if n == 0:
        raise ValueError("No paired scores after alignment.")

    rng = random.Random(seed)
    ids = list(range(n))

    a_scores: list[float] = []
    b_scores: list[float] = []
    wins = [0, 0, 0]

    sample_n = max(1, int(n * sample_ratio))

    mean_a_full = statistics.mean(scores_a)
    mean_b_full = statistics.mean(scores_b)
    print(
        f"Full-sample mean A={mean_a_full:.6f}, B={mean_b_full:.6f}, "
        f"diff(A-B)={mean_a_full - mean_b_full:.6f}"
    )

    for _ in range(num_samples):
        reduced_ids = rng.choices(ids, k=sample_n)
        va = statistics.mean(scores_a[i] for i in reduced_ids)
        vb = statistics.mean(scores_b[i] for i in reduced_ids)
        if va > vb:
            wins[0] += 1
        elif va < vb:
            wins[1] += 1
        else:
            wins[2] += 1
        a_scores.append(va)
        b_scores.append(vb)

    wins_frac = [x / float(num_samples) for x in wins]
    print(
        "Win ratio: A=%.3f, B=%.3f, tie=%.3f"
        % (wins_frac[0], wins_frac[1], wins_frac[2])
    )
    if wins_frac[0] > wins_frac[1]:
        print("(A is superior with p value p=%.3f)\n" % (1 - wins_frac[0]))
    elif wins_frac[1] > wins_frac[0]:
        print("(B is superior with p value p=%.3f)\n" % (1 - wins_frac[1]))

    a_scores.sort()
    b_scores.sort()
    lo = int(num_samples * 0.025)
    hi = int(num_samples * 0.975)
    print(
        "A mean=%.6f, median=%.6f, 95%% CI=[%.6f, %.6f]"
        % (
            statistics.mean(a_scores),
            statistics.median(a_scores),
            a_scores[lo],
            a_scores[hi],
        )
    )
    print(
        "B mean=%.6f, median=%.6f, 95%% CI=[%.6f, %.6f]"
        % (
            statistics.mean(b_scores),
            statistics.median(b_scores),
            b_scores[lo],
            b_scores[hi],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap on per-query metrics from two evaluation_summary.json files."
    )
    parser.add_argument("summary_a", help="First evaluation_summary.json")
    parser.add_argument("summary_b", help="Second evaluation_summary.json")
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC_CORRECT,
        choices=METRICS,
        help="Per-query field to aggregate (correct=0/1 accuracy, recall=recall %% as float)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of dataset size per bootstrap draw (1.0 = full n with replacement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible bootstrap draws",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require identical query_id sets in both files (default: use intersection)",
    )
    args = parser.parse_args()

    sa, sb, qids = align_paired_scores(
        args.summary_a, args.summary_b, args.metric, strict=args.strict
    )
    print(f"Paired rows: {len(qids)} (metric={args.metric})")
    paired_bootstrap_from_scores(
        sa,
        sb,
        num_samples=args.num_samples,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
