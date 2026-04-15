"""Plot histogram of candidate positions for selected tool calls."""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bin_position(pos_1indexed: int) -> str:
    """Map a 1-indexed candidate position to a histogram bin label."""
    if pos_1indexed <= 5:
        return str(pos_1indexed)
    elif pos_1indexed <= 10:
        return "6-10"
    elif pos_1indexed <= 15:
        return "11-15"
    elif pos_1indexed <= 20:
        return "16-20"
    else:
        return ">20"


BIN_ORDER = ["1", "2", "3", "4", "5", "6-10", "11-15", "16-20", ">20"]


def collect_positions(jsonl_path: str) -> list[int]:
    """Return list of 1-indexed candidate positions for all selected indices."""
    positions = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if "candidates" not in rec or "selected_indices" not in rec:
                continue
            candidates = rec["candidates"]
            selected = rec["selected_indices"]
            cand_to_pos = {c: i for i, c in enumerate(candidates)}
            for s in selected:
                if s in cand_to_pos:
                    positions.append(cand_to_pos[s] + 1)  # 1-indexed
                else:
                    print(f"WARNING: selected index {s} not in candidates for query_id={rec.get('query_id')}")
    return positions


def make_histogram_data(positions: list[int]) -> tuple[list[str], list[int], list[float]]:
    binned = Counter(bin_position(p) for p in positions)
    counts = [binned.get(b, 0) for b in BIN_ORDER]
    total = sum(counts)
    pcts = [100 * c / total if total > 0 else 0 for c in counts]
    return BIN_ORDER, counts, pcts


def main():
    base = Path("selected_tool_calls")
    files = {
        "selected_tool_calls [Tags]": base / "selected_tool_calls.jsonl",
        "selected_tool_calls [Original Messages]": base / "selected_tool_calls_gpt-oss-120b_use_original_messages_fixed.repaired.jsonl",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle("Distribution of Selected Candidate Positions", fontsize=15, fontweight="bold", y=1.02)

    colors = ["#4C72B0", "#DD8452"]

    for ax, (label, path), color in zip(axes, files.items(), colors):
        positions = collect_positions(str(path))
        bins, counts, pcts = make_histogram_data(positions)

        bars = ax.bar(bins, counts, color=color, edgecolor="white", linewidth=0.8, alpha=0.85)
        for bar, count, pct in zip(bars, counts, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xlabel("Candidate Position (1 = first candidate)", fontsize=11)
        ax.set_ylabel("Number of Selections", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        total_sel = len(positions)
        n_queries = sum(1 for _ in open(str(path)))
        ax.text(
            0.97, 0.95,
            f"Total selections: {total_sel}\nQueries: {n_queries}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    plt.tight_layout()
    out_path = "figures/selected_candidate_position_distribution.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    for label, path in files.items():
        positions = collect_positions(str(path))
        bins, counts, pcts = make_histogram_data(positions)
        print(f"\n--- {label} ---")
        print(f"  Total selections: {len(positions)}")
        for b, c, p in zip(bins, counts, pcts):
            print(f"  Position {b:>5s}: {c:>5d}  ({p:5.1f}%)")
        if positions:
            print(f"  Mean position: {np.mean(positions):.2f}")
            print(f"  Median position: {np.median(positions):.1f}")


if __name__ == "__main__":
    main()
