"""Plot histograms of search-call counts across trajectory folders.

Example:
    python scripts/plot_search_call_distribution.py \
        --pattern "runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_*" \
        --output figures/search_call_distribution_gpt-oss-120b.png
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt


BIN_LABELS = ["0", "1-5", "6-10", "11-15", "16-20", "21-25", "26-30", ">30"]


def assign_bin(n: int) -> int:
    if n <= 0:
        return 0
    if 1 <= n <= 5:
        return 1
    if 6 <= n <= 10:
        return 2
    if 11 <= n <= 15:
        return 3
    if 16 <= n <= 20:
        return 4
    if 21 <= n <= 25:
        return 5
    if 26 <= n <= 30:
        return 6
    return 7  # > 30


FOLDER_LABEL_MAP = {
    "traj_summary_ext_selected_tools":      "Selected Tools (Tags)",
    "traj_summary_orig_ext_selected_tools": "Selected Tools (Original Messages)",
    "traj_summary_ext":                     "Summary (Tags)",
    "traj_summary_orig_ext":                "Summary (Original Messages)",
    "traj_orig_ext":                        "Full Trajectory (Original Messages)",
    "traj_ext":                             "Full Trajectory (Tags)",
}

# Desired subplot order for --sort label. Top row = Original Messages,
# bottom row = Tags; within each row: Full Trajectory → Selected Tools → Summary.
LABEL_ORDER = [
    "Full Trajectory (Original Messages)",
    "Selected Tools (Original Messages)",
    "Summary (Original Messages)",
    "Full Trajectory (Tags)",
    "Selected Tools (Tags)",
    "Summary (Tags)",
]


def folder_label(folder_name: str) -> str:
    """Pick the longest key in FOLDER_LABEL_MAP that prefixes the folder name."""
    for key in sorted(FOLDER_LABEL_MAP, key=len, reverse=True):
        if folder_name.startswith(key):
            return FOLDER_LABEL_MAP[key]
    return folder_name


def load_search_counts(folder: str):
    counts = []
    n_missing = 0
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    for fp in files:
        try:
            with open(fp) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            n_missing += 1
            continue
        tcc = d.get("tool_call_counts") or {}
        n = tcc.get("search", 0) if isinstance(tcc, dict) else 0
        if not isinstance(n, int):
            n_missing += 1
            continue
        counts.append(n)
    return counts, n_missing, len(files)


def bin_counts(counts):
    binned = Counter()
    for c in counts:
        binned[assign_bin(c)] += 1
    return [binned.get(i, 0) for i in range(len(BIN_LABELS))]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pattern",
        required=True,
        help='Glob pattern matching trajectory folders, e.g. "runs/bcp/.../traj_*"',
    )
    ap.add_argument("--output", required=True, help="Path to save the output PNG.")
    ap.add_argument("--cols", type=int, default=3, help="Number of subplot columns.")
    ap.add_argument("--title", default=None, help="Figure suptitle.")
    ap.add_argument(
        "--sort",
        default="label",
        choices=["label", "name"],
        help="Subplot order: 'label' uses the FOLDER_LABEL_MAP order; 'name' sorts by folder name.",
    )
    return ap.parse_args()


def order_folders(folders, mode):
    if mode == "name":
        return sorted(folders)
    # 'label' mode: use the explicit LABEL_ORDER list. Labels not present in
    # LABEL_ORDER (including unmapped folders) sort after, alphabetically.
    order_index = {label: i for i, label in enumerate(LABEL_ORDER)}
    def sort_key(path):
        name = os.path.basename(path.rstrip("/"))
        label = folder_label(name)
        if label in order_index:
            return (0, order_index[label], name)
        return (1, 0, label if label != name else name)
    return sorted(folders, key=sort_key)


def main():
    args = parse_args()

    folders = [p for p in glob.glob(args.pattern) if os.path.isdir(p)]
    if not folders:
        print(f"No folders matched pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)
    folders = order_folders(folders, args.sort)

    print(f"Matched {len(folders)} folder(s) for pattern: {args.pattern}")
    print()

    per_folder = []
    header = f"{'folder':<55} {'label':<38} {'n_files':>8} {'plotted':>8} {'missing':>8}"
    print(header)
    print("-" * len(header))
    for folder in folders:
        name = os.path.basename(folder.rstrip("/"))
        label = folder_label(name)
        counts, n_missing, n_files = load_search_counts(folder)
        per_folder.append((folder, name, label, counts, n_missing, n_files))
        print(f"{name:<55} {label:<38} {n_files:>8} {len(counts):>8} {n_missing:>8}")
    print()

    n = len(folders)
    cols = max(1, min(args.cols, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)

    x = list(range(len(BIN_LABELS)))
    ymax = 0
    for ax_idx, (folder, name, label, counts, n_missing, n_files) in enumerate(per_folder):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        binned = bin_counts(counts)
        ymax = max(ymax, max(binned) if binned else 0)
        bars = ax.bar(x, binned, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=0, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("number of search calls")
        ax.set_ylabel("number of trajectories")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for b, v in zip(bars, binned):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                        ha="center", va="bottom", fontsize=7)

    # hide unused axes
    for idx in range(len(per_folder), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    # unify y-limits for easier visual comparison
    pad = max(1, int(ymax * 0.08))
    for ax_row in axes:
        for ax in ax_row:
            if ax.has_data():
                ax.set_ylim(0, ymax + pad)

    if args.title:
        fig.suptitle(args.title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
