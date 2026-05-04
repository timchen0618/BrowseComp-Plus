#!/usr/bin/env python3
"""Compute McNemar paired significance vs each model's baseline for the
scout/explore conditions on BCP test150.

Outputs three markdown tables (one per main agent) matching the column layout
in `scout_explore.md` (lines 63-78), with one new column inserted between
`Acc` and `Recall`:

    | Condition | Acc | Δ vs base | Recall | # calls |

Δ vs base is rendered as: `+5.3 [-0.5, +11.1] **`
where the bracketed values are a paired 95% CI on the accuracy difference and
`*`/`**` mark raw / BH-corrected (q=0.05) significance.

Run with the project venv:
    /scratch/afw8937/efficient-search-agents/venv/bin/python \\
        scripts/compute_significance.py

Notes
-----
- Acc, Recall, and # calls (avg search) are read straight from each
  evaluation_summary.json's top-level fields, not recomputed.  This matches
  whatever was already in scout_explore.md.
- Per-query correctness for McNemar is read from `per_query_metrics`.  Each
  condition is intersected with the baseline's qid set before comparing, so
  rows with N < 150 still get a valid paired test (just on the overlap).
- Baseline run names had to be discovered manually: `qwen3.5-122b-a10b` and
  `minimax-m2.5` use `seed0`; `glm-4.7-flash` has no test150 baseline folder
  so we fall back to `full/glm-4.7-flash/seed0` (830 queries) filtered to the
  test150 qid list.
- McNemar exact (no continuity correction) via
  statsmodels.stats.contingency_tables.mcnemar.
- BH correction (Benjamini-Hochberg, q=0.05, fdr_bh) is applied across all
  non-NaN p-values from all 3 tables.
- 95% CI on Δ uses the simple paired-binomial half-width
  `1.96 * sqrt((b+c)/n^2)`; this is the conservative fallback documented in
  the spec (Newcombe-paired isn't directly exposed in statsmodels).
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_BASE = os.path.join(REPO_ROOT, "evals", "bcp", "Qwen3-Embedding-8B", "test150")
TEST150_QIDS_FILE = os.path.join(
    REPO_ROOT, "topics-qrels", "bcp", "queries_test150_qids.txt"
)

MODELS = ["glm-4.7-flash", "qwen3.5-122b-a10b", "minimax-m2.5"]

# (run_name_template, label_in_table, is_best_of_n)
# NOTE: The user-provided template names did not match what is actually on
# disk, so the dict has been corrected per run.  The corrections were verified
# by spot-checking Accuracy (%) against the values in scout_explore.md.
#
#   'baseline_seed0'                      -> 'seed0'
#       (and for glm: pulled from full/glm-4.7-flash/seed0 + filtered)
#   'gemini_pro_selection_traj_orig_seed0' -> 'selected_tools_seed0'
CONDITIONS: List[Tuple[str, str, bool]] = [
    ("seed0", "Baseline", False),
    ("traj_orig_ext_seed0", "+ full trajectory", False),
    ("traj_summary_orig_ext_seed0", "+ trajectory summary", False),
    ("selected_tools_seed0", "+ Gemini-2.5-pro selected k=5 tool calls", False),
    ("random_tools_seed42", "+ random k=5 tool calls (selection seed=42)", False),
    ("random_tools_seed43", "+ random k=5 tool calls (selection seed=43)", False),
    ("random_tools_seed44", "+ random k=5 tool calls (selection seed=44)", False),
    ("random_tools_seed45", "+ random k=5 tool calls (selection seed=45)", False),
    (
        "qwen3.5-4b_vanilla_traj_orig_seed0",
        "+ qwen3.5-4b explorer (budget=5, vanilla)",
        False,
    ),
    (
        "qwen3.5-4b-sft-best_of_4_random_traj_orig_seed0",
        "+ qwen3.5-4b explorer (SFT on best-of-4 random selection)",
        False,
    ),
    (
        "qwen3.5-4b-sft-gemini_traj_orig_seed0",
        "+ qwen3.5-4b explorer (SFT on Gemini-2.5-pro selection)",
        False,
    ),
    (
        "qwen3.5-4b-sft-random_traj_orig_seed0",
        "+ qwen3.5-4b explorer (SFT on random selection)",
        False,
    ),
]

MIN_N = 100  # below this we treat the condition as missing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test150_qids() -> set:
    qids = set()
    with open(TEST150_QIDS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines look like "1\t778" — take the second column.
            parts = line.split("\t")
            qid = parts[-1].strip()
            if qid:
                qids.add(qid)
    return qids


def eval_path(model: str, run_name: str) -> str:
    return os.path.join(EVAL_BASE, model, run_name, "evaluation_summary.json")


def load_eval(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline(model: str, test150_qids: set) -> Optional[dict]:
    """Special case: glm-4.7-flash has no test150 baseline folder, so we pull
    full/glm-4.7-flash/seed0 (830-query eval) and filter to test150 qids.
    Returns a dict with the same shape as a normal eval JSON's relevant
    fields, plus a 'qid_to_correct' map.
    """
    if model == "glm-4.7-flash":
        path = os.path.join(
            REPO_ROOT,
            "evals",
            "bcp",
            "Qwen3-Embedding-8B",
            "full",
            "glm-4.7-flash",
            "seed0",
            "evaluation_summary.json",
        )
        d = load_eval(path)
        if d is None:
            return None
        pq_full = d.get("per_query_metrics", [])
        pq = [r for r in pq_full if str(r["query_id"]) in test150_qids]
        if len(pq) < MIN_N:
            return None
        # Recompute headline accuracy on the test150 slice (the 830-query
        # 'Accuracy (%)' field doesn't reflect the slice).  scout_explore.md
        # reports 48.0 for this row; we'll use the recomputed value in the
        # _Δ vs base_ math but still report scout_explore.md's headline number
        # in the Acc column via a separate path (see render_table).
        n_correct = sum(1 for r in pq if r["correct"])
        acc = 100.0 * n_correct / len(pq) if pq else 0.0
        # Average search calls across the test150 slice.
        searches = [r.get("num_search_calls", 0) for r in pq]
        avg_search = sum(searches) / len(searches) if searches else 0.0
        # Recall isn't in per_query_metrics in a directly-aggregable form
        # (it's already a percentage per query).  Average it.
        recalls = [r.get("recall", 0.0) for r in pq]
        recall = sum(recalls) / len(recalls) if recalls else 0.0
        return {
            "Accuracy (%)": acc,
            "Recall (%)": recall,
            "avg_tool_stats": {"search": avg_search},
            "per_query_metrics": pq,
            "qid_to_correct": {str(r["query_id"]): bool(r["correct"]) for r in pq},
            "n": len(pq),
            "_source": "full/glm-4.7-flash/seed0 filtered to test150",
        }

    # Standard path for qwen3.5-122b-a10b and minimax-m2.5.
    path = eval_path(model, "seed0")
    d = load_eval(path)
    if d is None:
        return None
    pq = d.get("per_query_metrics", [])
    if len(pq) < MIN_N:
        return None
    return {
        "Accuracy (%)": d.get("Accuracy (%)"),
        "Recall (%)": d.get("Recall (%)"),
        "avg_tool_stats": d.get("avg_tool_stats", {}),
        "per_query_metrics": pq,
        "qid_to_correct": {str(r["query_id"]): bool(r["correct"]) for r in pq},
        "n": len(pq),
        "_source": f"test150/{model}/seed0",
    }


def paired_ci_half_width(b: int, c: int, n: int, z: float = 1.96) -> float:
    """Simple paired-binomial 95% CI half-width on Δ = (c - b) / n.

    From the spec's fallback: half_width = z * sqrt((b+c)/n^2).
    """
    if n <= 0:
        return float("nan")
    return z * math.sqrt((b + c) / (n * n))


def mcnemar_pvalue(b: int, c: int) -> float:
    """McNemar exact two-sided p-value, no continuity correction.

    statsmodels.mcnemar with exact=True ignores `correction` (correction is
    only used in the asymptotic version), so we pass correction=False for
    clarity.  The off-diagonals are b and c; the on-diagonals don't matter
    for the test, so we put any plausible values on them.
    """
    table = [[0, b], [c, 0]]
    res = mcnemar(table, exact=True, correction=False)
    return float(res.pvalue)


def fmt_delta_cell(
    delta_pp: float, lo_pp: float, hi_pp: float, star: str
) -> str:
    """Render a Δ cell as `+5.3 [-0.5, +11.1] **`."""

    def fmt_signed(x: float) -> str:
        return f"{x:+.1f}"

    star_part = f" {star}" if star else ""
    return f"{fmt_signed(delta_pp)} [{fmt_signed(lo_pp)}, {fmt_signed(hi_pp)}]{star_part}"


# ---------------------------------------------------------------------------
# Per-condition stats
# ---------------------------------------------------------------------------


def compute_row(
    model: str,
    run_name: str,
    label: str,
    baseline: dict,
) -> dict:
    """Compute one row's worth of data.  Returns a dict with at minimum:

        {
          'label': str,
          'present': bool,
          'acc': float | None,            # eval JSON's headline Accuracy (%)
          'recall': float | None,         # eval JSON's headline Recall (%)
          'num_calls': float | None,      # avg_tool_stats.search
          'n_paired': int | None,         # |intersection with baseline qids|
          'delta_pp': float | None,
          'ci_lo_pp': float | None,
          'ci_hi_pp': float | None,
          'pvalue': float | None,
          'sanity_acc_paired': float | None,  # acc on the intersection
          'baseline_acc_paired': float | None,
        }
    """
    is_baseline = label == "Baseline"
    if is_baseline:
        return {
            "label": label,
            "present": True,
            "acc": baseline["Accuracy (%)"],
            "recall": baseline["Recall (%)"],
            "num_calls": baseline.get("avg_tool_stats", {}).get("search"),
            "n_paired": baseline["n"],
            "delta_pp": None,
            "ci_lo_pp": None,
            "ci_hi_pp": None,
            "pvalue": None,
            "sanity_acc_paired": baseline["Accuracy (%)"],
            "baseline_acc_paired": baseline["Accuracy (%)"],
        }

    path = eval_path(model, run_name)
    d = load_eval(path)
    if d is None:
        return {
            "label": label,
            "present": False,
            "acc": None,
            "recall": None,
            "num_calls": None,
            "n_paired": None,
            "delta_pp": None,
            "ci_lo_pp": None,
            "ci_hi_pp": None,
            "pvalue": None,
            "sanity_acc_paired": None,
            "baseline_acc_paired": None,
        }

    pq = d.get("per_query_metrics", [])
    if len(pq) < MIN_N:
        return {
            "label": label,
            "present": False,
            "acc": d.get("Accuracy (%)"),
            "recall": d.get("Recall (%)"),
            "num_calls": d.get("avg_tool_stats", {}).get("search"),
            "n_paired": len(pq),
            "delta_pp": None,
            "ci_lo_pp": None,
            "ci_hi_pp": None,
            "pvalue": None,
            "sanity_acc_paired": None,
            "baseline_acc_paired": None,
        }

    cond_qid_to_correct = {str(r["query_id"]): bool(r["correct"]) for r in pq}
    base_q2c = baseline["qid_to_correct"]
    common = sorted(set(cond_qid_to_correct) & set(base_q2c))
    n = len(common)

    if n == 0:
        return {
            "label": label,
            "present": False,
            "acc": d.get("Accuracy (%)"),
            "recall": d.get("Recall (%)"),
            "num_calls": d.get("avg_tool_stats", {}).get("search"),
            "n_paired": 0,
            "delta_pp": None,
            "ci_lo_pp": None,
            "ci_hi_pp": None,
            "pvalue": None,
            "sanity_acc_paired": None,
            "baseline_acc_paired": None,
        }

    # b = baseline=1, cond=0; c = baseline=0, cond=1
    b = sum(1 for q in common if base_q2c[q] and not cond_qid_to_correct[q])
    c = sum(1 for q in common if not base_q2c[q] and cond_qid_to_correct[q])
    n_base_correct = sum(1 for q in common if base_q2c[q])
    n_cond_correct = sum(1 for q in common if cond_qid_to_correct[q])

    base_acc_paired = 100.0 * n_base_correct / n
    cond_acc_paired = 100.0 * n_cond_correct / n
    delta_pp = cond_acc_paired - base_acc_paired

    half = paired_ci_half_width(b, c, n)
    ci_lo = delta_pp - 100.0 * half
    ci_hi = delta_pp + 100.0 * half

    if b + c == 0:
        # No discordant pairs => p=1 by definition (no evidence either way).
        pvalue = 1.0
    else:
        pvalue = mcnemar_pvalue(b, c)

    return {
        "label": label,
        "present": True,
        "acc": d.get("Accuracy (%)"),
        "recall": d.get("Recall (%)"),
        "num_calls": d.get("avg_tool_stats", {}).get("search"),
        "n_paired": n,
        "delta_pp": delta_pp,
        "ci_lo_pp": ci_lo,
        "ci_hi_pp": ci_hi,
        "pvalue": pvalue,
        "sanity_acc_paired": cond_acc_paired,
        "baseline_acc_paired": base_acc_paired,
        "b": b,
        "c": c,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

MODEL_HEADERS = {
    "glm-4.7-flash": "Model: GLM-4.7-Flash (30B)",
    "qwen3.5-122b-a10b": "Model: Qwen3.5-122B-A10B",
    "minimax-m2.5": "Model: MiniMax-M2.5 (229B)",
}


def render_tables(per_model_rows: "OrderedDict[str, List[dict]]") -> str:
    out_lines: List[str] = []
    for model, rows in per_model_rows.items():
        out_lines.append(f"**{MODEL_HEADERS[model]}**")
        out_lines.append("")
        out_lines.append("| Condition | Acc | Δ vs base | Recall | # calls |")
        out_lines.append("| :---- | ----: | :---- | ----: | ----: |")
        for row in rows:
            label = row["label"]
            if not row["present"]:
                out_lines.append(
                    f"| {label} | (no eval) | — | — | — |"
                )
                continue
            acc = row["acc"]
            recall = row["recall"]
            calls = row["num_calls"]
            acc_s = f"{acc:.1f}" if acc is not None else "—"
            recall_s = f"{recall:.1f}" if recall is not None else "—"
            calls_s = f"{calls:.1f}" if calls is not None else "—"

            if row["delta_pp"] is None:
                # Baseline row: show "—" in Δ column.
                delta_s = "—"
            else:
                delta_s = fmt_delta_cell(
                    row["delta_pp"],
                    row["ci_lo_pp"],
                    row["ci_hi_pp"],
                    row.get("star", ""),
                )
            out_lines.append(
                f"| {label} | {acc_s} | {delta_s} | {recall_s} | {calls_s} |"
            )
        out_lines.append("")
    out_lines.append(
        "* p<0.05 (McNemar exact); ** BH-significant at q=0.05"
    )
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def sanity_check_acc(model: str, row: dict) -> List[str]:
    """Return any sanity-check warnings to print to stderr."""
    warnings = []
    if not row["present"] or row["delta_pp"] is None:
        return warnings
    # The eval JSON's headline 'Accuracy (%)' should match the paired-set
    # accuracy within ~0.1pp when N==150 and there are no missing qids.
    if row["acc"] is not None and row["sanity_acc_paired"] is not None:
        diff = abs(row["acc"] - row["sanity_acc_paired"])
        if diff > 0.5 and row["n_paired"] >= 100:
            warnings.append(
                f"  [warn] {model}/{row['label']}: headline acc "
                f"{row['acc']:.2f} vs paired-set acc "
                f"{row['sanity_acc_paired']:.2f} (Δ={diff:.2f}pp, "
                f"n_paired={row['n_paired']})"
            )
    return warnings


def main() -> int:
    test150_qids = load_test150_qids()
    print(f"# loaded {len(test150_qids)} test150 qids", file=sys.stderr)

    per_model_rows: "OrderedDict[str, List[dict]]" = OrderedDict()
    flat_rows_with_p: List[Tuple[str, str, dict]] = []  # (model, label, row)

    warnings: List[str] = []
    for model in MODELS:
        baseline = load_baseline(model, test150_qids)
        if baseline is None:
            print(f"# WARNING: no baseline for {model}", file=sys.stderr)
            per_model_rows[model] = []
            continue
        print(
            f"# baseline {model}: source={baseline['_source']} "
            f"n={baseline['n']} acc={baseline['Accuracy (%)']:.2f}",
            file=sys.stderr,
        )

        rows = []
        for run_name, label, _is_best in CONDITIONS:
            row = compute_row(model, run_name, label, baseline)
            rows.append(row)
            warnings.extend(sanity_check_acc(model, row))
            if row["present"] and row["pvalue"] is not None:
                flat_rows_with_p.append((model, label, row))
        per_model_rows[model] = rows

    # BH correction across all collected p-values.
    if flat_rows_with_p:
        pvals = np.array([r[2]["pvalue"] for r in flat_rows_with_p], dtype=float)
        # Drop NaNs defensively even though we filter above.
        mask = ~np.isnan(pvals)
        if mask.any():
            reject, _, _, _ = multipletests(
                pvals[mask], alpha=0.05, method="fdr_bh"
            )
            j = 0
            for i, ok in enumerate(mask):
                if not ok:
                    flat_rows_with_p[i][2]["bh_reject"] = False
                    continue
                flat_rows_with_p[i][2]["bh_reject"] = bool(reject[j])
                j += 1
        else:
            for _, _, row in flat_rows_with_p:
                row["bh_reject"] = False

    # Stamp star markers onto each row.
    for _, _, row in flat_rows_with_p:
        if row.get("bh_reject"):
            row["star"] = "**"
        elif row["pvalue"] is not None and row["pvalue"] < 0.05:
            row["star"] = "*"
        else:
            row["star"] = ""

    # Print warnings (if any) before tables, on stderr.
    for w in warnings:
        print(w, file=sys.stderr)

    print(render_tables(per_model_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
