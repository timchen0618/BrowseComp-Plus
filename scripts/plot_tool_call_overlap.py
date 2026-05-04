#!/usr/bin/env python3
"""
Visualize pairwise Jaccard overlap of selected_indices across selected-tool-call JSONL runs.

Usage:
    python scripts/plot_tool_call_overlap.py

Output:
    figures/tool_call_overlap_heatmap.png
    figures/tool_call_overlap_heatmap_random_seeds_seed0.png
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path("selected_tool_calls/all/gpt-oss-120b")

# 4 random (one per agent seed) vs 4 actual Gemini selections
RUNS_MAIN = [
    ("rand_s0",   BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s1",   BASE / "seed1/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s2",   BASE / "seed2/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s3",   BASE / "seed3/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("actual_s0", BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_more_chars.jsonl"),
    ("actual_s1", BASE / "seed1/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
    ("actual_s2", BASE / "seed2/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
    ("actual_s3", BASE / "seed3/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
]

# 5 random baselines from seed0 trajectories with different random seeds
RUNS_RANDOM_SEED0 = [
    ("rand42",        BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand0",         BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed0.jsonl"),
    ("rand1",         BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed1.jsonl"),
    ("rand2",         BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed2.jsonl"),
    ("rand3",         BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed3.jsonl"),
    ("gemini_2.5-pro-1", BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_more_chars.jsonl"),
    ("gemini_2.5-pro-0", BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
    ("gemini_3.1-pro",   BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_gemini_3.1-pro-preview.jsonl"),
]

HEATMAPS = [
    (RUNS_MAIN,         "figures/tool_call_overlap_heatmap.png",
     "Mean Jaccard Overlap of Selected Tool Call Indices\n(over shared query IDs per pair)"),
    (RUNS_RANDOM_SEED0, "figures/tool_call_overlap_heatmap_random_seeds_seed0.png",
     "Mean Jaccard Overlap — Random Baselines, seed0 Trajectories\n(5 random seeds on same candidate pool)"),
]


def load_run(path: Path) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            data[rec["query_id"]] = set(rec.get("selected_indices") or [])
    return data


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def pairwise_jaccard(run_a: dict, run_b: dict) -> tuple:
    shared = set(run_a) & set(run_b)
    only_a = len(run_a) - len(shared)
    only_b = len(run_b) - len(shared)
    mean = float(np.mean([jaccard(run_a[q], run_b[q]) for q in shared])) if shared else 0.0
    return mean, len(shared), only_a, only_b


def build_matrix(runs: dict, labels: list) -> np.ndarray:
    n = len(labels)
    matrix = np.eye(n)
    print("\nPairwise Jaccard (mean | shared | only_A | only_B):")
    for i, j in combinations(range(n), 2):
        la, lb = labels[i], labels[j]
        mean, n_shared, only_a, only_b = pairwise_jaccard(runs[la], runs[lb])
        matrix[i, j] = matrix[j, i] = mean
        mismatch = f"  *** MISMATCH: only_{la}={only_a}, only_{lb}={only_b}" if (only_a or only_b) else ""
        print(f"  {la:12s} vs {lb:12s}: {mean:.3f}  (shared={n_shared}){mismatch}")
    return matrix


def plot_heatmap(matrix: np.ndarray, labels: list, title: str, out: Path) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n + 2), max(5, n + 1)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    # Load all unique paths once
    all_paths = {label: path for run_set, _, _ in HEATMAPS for label, path in run_set}
    loaded = {}
    for label, path in all_paths.items():
        if label not in loaded:
            if not path.exists():
                print(f"WARNING: missing {path}")
                loaded[label] = {}
            else:
                loaded[label] = load_run(path)
                print(f"Loaded {label}: {len(loaded[label])} queries")

    for run_set, out_path, title in HEATMAPS:
        print(f"\n--- {out_path} ---")
        labels = [r[0] for r in run_set]
        matrix = build_matrix(loaded, labels)
        plot_heatmap(matrix, labels, title, Path(out_path))


if __name__ == "__main__":
    main()
