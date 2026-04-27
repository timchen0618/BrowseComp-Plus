# Tool Call Overlap Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Visualize pairwise Jaccard overlap of `selected_indices` across 8 selected-tool-call JSONL runs (4 random baselines × seeds 0–3, 4 actual Gemini selections × seeds 0–3) as an 8×8 annotated heatmap.

**Architecture:** A self-contained Python script reads 8 JSONL files, computes mean per-instance Jaccard similarity for each pair over shared `query_id`s, prints a mismatch report, and saves a seaborn heatmap to `figures/`. A companion skill (`~/.claude/skills/plot-tool-call-overlap/SKILL.md`) documents when/how to invoke or extend the script.

**Tech Stack:** Python 3, numpy, matplotlib, seaborn (all available in the conda env).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/plot_tool_call_overlap.py` | Load 8 runs, compute pairwise Jaccard, save heatmap |
| Create | `~/.claude/skills/plot-tool-call-overlap/SKILL.md` | Skill: when/how to use this visualization |
| Modify | `CLAUDE.md` | Add row to `scripts/` table |
| Write | `~/.claude/projects/.../memory/tool_call_overlap_skill.md` | Memory entry for the skill |
| Modify | `~/.claude/projects/.../memory/MEMORY.md` | Index pointer to new memory entry |

---

## Task 1: Write `scripts/plot_tool_call_overlap.py`

**Files:**
- Create: `scripts/plot_tool_call_overlap.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Visualize pairwise Jaccard overlap of selected_indices across selected-tool-call JSONL runs.

Usage:
    python scripts/plot_tool_call_overlap.py

Output:
    figures/tool_call_overlap_heatmap.png
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path("selected_tool_calls/all/gpt-oss-120b")

RUNS = [
    ("rand_s0",   BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s1",   BASE / "seed1/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s2",   BASE / "seed2/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("rand_s3",   BASE / "seed3/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed42.jsonl"),
    ("actual_s0", BASE / "seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_more_chars.jsonl"),
    ("actual_s1", BASE / "seed1/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
    ("actual_s2", BASE / "seed2/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
    ("actual_s3", BASE / "seed3/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl"),
]


def load_run(path: Path) -> dict[str, set]:
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            data[rec["query_id"]] = set(rec.get("selected_indices") or [])
    return data


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def pairwise_jaccard(run_a: dict, run_b: dict) -> tuple[float, int, int, int]:
    shared = set(run_a) & set(run_b)
    only_a = len(run_a) - len(shared)
    only_b = len(run_b) - len(shared)
    mean = float(np.mean([jaccard(run_a[q], run_b[q]) for q in shared])) if shared else 0.0
    return mean, len(shared), only_a, only_b


def main():
    runs = {}
    for label, path in RUNS:
        if not path.exists():
            print(f"WARNING: missing {path}")
            runs[label] = {}
        else:
            runs[label] = load_run(path)
            print(f"Loaded {label}: {len(runs[label])} queries")

    labels = [r[0] for r in RUNS]
    n = len(labels)
    matrix = np.eye(n)

    print("\nPairwise Jaccard (mean | shared | only_A | only_B):")
    for i, j in combinations(range(n), 2):
        la, lb = labels[i], labels[j]
        mean, n_shared, only_a, only_b = pairwise_jaccard(runs[la], runs[lb])
        matrix[i, j] = matrix[j, i] = mean
        mismatch = f"  *** MISMATCH: only_{la}={only_a}, only_{lb}={only_b}" if (only_a or only_b) else ""
        print(f"  {la:12s} vs {lb:12s}: {mean:.3f}  (shared={n_shared}){mismatch}")

    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_title(
        "Mean Jaccard Overlap of Selected Tool Call Indices\n(over shared query IDs per pair)",
        fontsize=13,
    )
    plt.tight_layout()

    out = Path("figures/tool_call_overlap_heatmap.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

```bash
cd /scratch/hc3337/projects/BrowseComp-Plus
python scripts/plot_tool_call_overlap.py
```

Expected: 8 "Loaded …" lines, pairwise table printed, `figures/tool_call_overlap_heatmap.png` saved with no errors.

---

## Task 2: Create the skill

**Files:**
- Create: `~/.claude/skills/plot-tool-call-overlap/SKILL.md`

- [ ] **Step 1: Write skill**

```markdown
---
name: plot-tool-call-overlap
description: >
  Use when the user asks to visualize, compare, or compute pairwise overlap
  between selected_tool_calls JSONL files (Jaccard similarity of selected_indices).
  Triggers on phrases like "overlap between runs", "how similar are the selections",
  "pairwise overlap heatmap", or when pointing at selected_tool_calls/ JSONL files.
---

# Plot Tool Call Overlap Heatmap

## What this skill does

Runs `scripts/plot_tool_call_overlap.py` to produce an 8×8 annotated heatmap of
mean Jaccard similarity between `selected_indices` sets across selected-tool-call JSONL runs.
Saves output to `figures/tool_call_overlap_heatmap.png`.

---

## Step-by-step

### 1. Verify files exist

The script expects these 8 files under `selected_tool_calls/all/gpt-oss-120b/`:

| Label | File |
|-------|------|
| `rand_s0` | `seed0/…random_seed42.jsonl` |
| `rand_s1` | `seed1/…random_seed42.jsonl` |
| `rand_s2` | `seed2/…random_seed42.jsonl` |
| `rand_s3` | `seed3/…random_seed42.jsonl` |
| `actual_s0` | `seed0/…more_chars.jsonl` |
| `actual_s1` | `seed1/…use_original_messages.jsonl` |
| `actual_s2` | `seed2/…use_original_messages.jsonl` |
| `actual_s3` | `seed3/…use_original_messages.jsonl` |

If any are missing, the script prints a WARNING and uses an empty dict for that run (all overlaps = 0).

### 2. Run the script

```bash
cd /scratch/hc3337/projects/BrowseComp-Plus
python scripts/plot_tool_call_overlap.py
```

### 3. Interpret output

- **Diagonal** = 1.00 (self-overlap, always perfect).
- **MISMATCH lines** = the two runs operated on different query sets; only shared IDs contribute to the score.
- **rand vs rand** similarity reflects how random the random baseline is.
- **actual vs actual** similarity shows Gemini selection consistency across agent seeds.
- **rand vs actual** similarity is the key comparison: how different is the learned selection from random?

### 4. Customise RUNS list

To add or swap runs, edit the `RUNS` list at the top of `scripts/plot_tool_call_overlap.py`.
Each entry is `(label, Path)` — label appears on both axes of the heatmap.

---

## Overlap metric

**Jaccard similarity** per instance: `|A ∩ B| / |A ∪ B|` over `selected_indices` sets.
Averaged over all `query_id`s shared by both runs.
Empty ∩ empty → 1.0 (both selected nothing).

---

## Edge cases

- Missing file → WARNING printed, empty dict used (overlap = 0 vs all others).
- Query ID mismatch → flagged with `*** MISMATCH` in stdout; only intersection used.
- Both sets empty for a query → Jaccard = 1.0 (neither selected anything).
```

---

## Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (scripts/ table)

- [ ] **Step 1: Add row to scripts/ table**

Find the scripts table and add:

```
| `plot_tool_call_overlap.py` | Pairwise Jaccard heatmap of selected_indices across JSONL runs |
```

---

## Task 4: Update memory

**Files:**
- Write: `~/.claude/projects/-scratch-hc3337-projects-BrowseComp-Plus/memory/project_tool_call_overlap_skill.md`
- Modify: `~/.claude/projects/-scratch-hc3337-projects-BrowseComp-Plus/memory/MEMORY.md`

- [ ] **Step 1: Write memory file**

```markdown
---
name: tool-call-overlap-visualization
description: script and skill for pairwise Jaccard heatmap of selected tool call runs
type: project
---

Script `scripts/plot_tool_call_overlap.py` computes and visualizes mean Jaccard similarity
of `selected_indices` between 8 selected-tool-call JSONL runs (4 random × seeds 0–3, 4 actual × seeds 0–3).
Output: `figures/tool_call_overlap_heatmap.png`.

**Why:** Requested to compare consistency of Gemini-selected vs random baseline tool call selections across agent seeds.
**How to apply:** Use skill `plot-tool-call-overlap` when asked about overlap/similarity between selected_tool_calls runs. To change which runs are compared, edit the `RUNS` list in the script.
```

- [ ] **Step 2: Add pointer to MEMORY.md**

```
- [Tool call overlap visualization](project_tool_call_overlap_skill.md) — Jaccard heatmap script + skill for comparing selected_indices across runs
```
