#!/usr/bin/env python3
"""
Submit SLURM array jobs for missing shards only.
Reads run_qwen3_planning.SBATCH as a template and patches
--array, MODEL_NAME, mode, and seed for each incomplete run.

Usage:
    python submit_missing.py          # dry-run: prints sbatch commands
    python submit_missing.py --submit # actually submits
"""

import argparse
import re
import subprocess
import time
import tempfile
import os

TEMPLATE_PATH = "run_qwen3_planning.SBATCH"
TEMPLATE_PATH_FIRST50 = "run_qwen3_first50.SBATCH"
TEMPLATE_PATH_TEST150 = "run_qwen3_test150.SBATCH"
TEMPLATE_PATH_TRAIN680 = "run_qwen3_train680.SBATCH"

# Missing shards per leaf folder for full split (from missing_1.txt analysis)
MISSING = {
    # "tongyi_seed3":                                                                  list(range(10)),
    # "gpt-oss-120b_planning_retrospective_seed0":                                     list(range(10)),
    # "gpt-oss-120b_planning_retrospective_reinject_every_5_seed0":                     list(range(10)),
    # # "gpt-oss-120b_planning_v0.5_start_ext_gemini_2.5_pro_seed0":                     list(range(10)),
    # # "gpt-oss-120b_planning_v0.5_start_ext_gemini_2.5_pro_reinject_every_5_seed0":    list(range(10)),
    # "gpt-oss-120b_traj_summary_ext_selected_tools_gpt-oss-120b_seed0":              list(range(10)),
    # "gpt-oss-120b_planning_v4_seed0":                                                list(range(10)),
    # "gpt-oss-120b_planning_v4_reinject_every_5_seed0":                               list(range(10)),
    # "gpt-oss-120b_planning_v3_start_ext_gemini_2.5_pro_seed0":                       [1, 2, 4, 5, 6],
    # "gpt-oss-120b_planning_v3_start_ext_gemini_2.5_pro_reinject_every_5_seed0":      list(range(10)),
    # "gpt-oss-120b_planning_v4_start_ext_gemini_2.5_pro_seed0":                       list(range(10)),
    # "gpt-oss-120b_planning_v4_start_ext_gemini_2.5_pro_reinject_every_5_seed0":      [1, 8],
    # "gpt-oss-120b_seed7":      list(range(10)),
    # "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0":                                     list(range(10)),
    # "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0":      [1,3,6,9],
    # "gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0":      [3,4,5],
    # "gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0":      [4,5],
    # "gpt-oss-120b_traj_summary_ext_gpt-oss-120b_seed0":                             [3,5,7],
    # "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0":      list(range(10)),  # complete
}

# Missing runs for first50 split (no shards — each entry is a full re-run).
# Value is None to auto-parse model/mode/seed from the run name,
# or an explicit (model, mode, seed) tuple for names that can't be parsed.
MISSING_FIRST50 = {
    # "tongyi_seed3":                                                                  None,
    # "gpt-oss-120b_planning_v4_start_ext_gemini_2.5_pro_seed0":                   None,
    # "gpt-oss-120b_planning_v4_start_ext_gemini_2.5_pro_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_traj_summary_ext_selected_tools_gpt-oss-120b_seed0":          None,
    # "gpt-oss-120b_planning_v4_start_ext_gemini_2.5_pro_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v1_start_ext_gemini_2.5_pro_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v2_start_ext_gemini_2.5_pro_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v3_start_ext_gemini_2.5_pro_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v1_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v2_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v3_revise_every_5_seed0":    None,
    # "gpt-oss-120b_planning_v4_revise_every_5_seed0":    None,
    # "gpt-oss-120b_seed4":      None,
    # "gpt-oss-120b_seed5":      None,
    # "gpt-oss-120b_seed6":      None,
    # "gpt-oss-120b_seed7":      None,
    # "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0":                                     None,
    # "gpt-oss-120b_traj_summary_ext_gpt-oss-120b_seed0":                             None,
    # "gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0":      None,  # complete
    # "gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0":      None,
    # "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0":      None,  # complete
}

# Missing runs for frames/first50 split — gpt-oss-120b model.
# Saves to runs/frames/Qwen3-Embedding-0.6B/first50/gpt-oss-120b/
MISSING_FRAMES_FIRST50 = {
    # "gpt-oss-120b_planning_v0.5_start_ext_seed0":                 None,
    # "gpt-oss-120b_planning_v0.5_start_ext_reinject_every_5_seed0": None,
    # "gpt-oss-120b_planning_v1_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v1_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v2_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v2_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v3_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v3_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v4_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v4_start_ext_reinject_every_5_seed0":  None,
}

MISSING_MUSIQUE_FIRST50 = {
    # "gpt-oss-120b_planning_v0.5_start_ext_seed0":                 None,
    # "gpt-oss-120b_planning_v0.5_start_ext_reinject_every_5_seed0": None,
    # "gpt-oss-120b_planning_v1_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v1_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v2_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v2_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v3_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v3_start_ext_reinject_every_5_seed0":  None,
    # "gpt-oss-120b_planning_v4_start_ext_seed0":                   None,
    # "gpt-oss-120b_planning_v4_start_ext_reinject_every_5_seed0":  None,
}

# Missing shards for bcp/test150 split (3 shards: 0-2).
MISSING_TEST150 = {
    # gpt-oss-120b — all modes
    # "gpt-oss-120b_traj_budget_orig_ext_gpt-oss-120b_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_traj_budget_orig_ext_tongyi_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_traj_budget_orig_ext_qwen3.6-35b-a3b_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_traj_budget_orig_ext_qwen3.5-4b_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_traj_budget_orig_ext_qwen3.5-4b-sft_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_seed0":                                                             [2],  # complete
    # "gpt-oss-120b_seed0":                                                           list(range(3)),
    # "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0":                                      [2],  # 16 missing: all in shard 2
    # "gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0":                                 [2],  # 16 missing: all in shard 2
    # "gpt-oss-120b_traj_summary_ext_gpt-oss-120b_seed0":                              [2],  # 16 missing: all in shard 2
    # "gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0":                         [2],  # 16 missing: all in shard 2
    # "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0":                                    list(range(3)),  # complete
    # "gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0":                               list(range(3)),  # complete
    # "gpt-oss-120b_traj_summary_ext_gpt-oss-120b_seed0":                            list(range(3)),  # complete
    # "gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0":                       [1],  # complete
    # "gpt-oss-120b_traj_summary_ext_selected_tools_gpt-oss-120b_seed0":             list(range(3)),  # complete
    # "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed1":        list(range(3)),  # complete
    # "gpt-oss-120b_budget_seed0":                                                    list(range(3)),
    # "gpt-oss-120b_budget_seed0":                                                   [0, 1, 2],  # 100 incomplete: shard1=50, shard2=50
    # "gpt-oss-120b_seed0":                                                           [0, 2],  # 60 incomplete: shard0=10, shard2=50
    # "gpt-oss-120b_traj_summary_ext_selected_tools_gpt-oss-120b_seed0":             [2],     # 10 incomplete: shard2=10
    # qwen3.5-4b — budget only
    # "qwen3.5-4b_budget_seed0":                                                       list(range(3)),
    "qwen3.5-4b-sft_budget_seed0":                                                       list(range(3)),
    # qwen3.6-35b-a3b — seed0 and budget only
    # "qwen3.6-35b-a3b_seed0":                                                        list(range(3)),
    # "qwen3.6-35b-a3b_budget_seed0":                                                 list(range(3)),
    # "qwen3.6-35b-a3b_seed0":                                                        [2],  # complete
    # "qwen3.6-35b-a3b_budget_seed1":                                                [0],  # complete
    # tongyi — all modes
    # "tongyi_seed0":                                                                  list(range(3)),
    # "tongyi_seed0":                                                                  [0],  # complete
    # "tongyi_budget5_seed0":                                                          [2],  # complete
    # "tongyi_traj_ext_tongyi_seed0":                                                  list(range(3)),
    # "tongyi_traj_orig_ext_tongyi_seed0":                                             list(range(3)),
    # "tongyi_traj_summary_ext_tongyi_seed0":                                          [1],
    # "tongyi_traj_summary_orig_ext_tongyi_seed0":                                     list(range(3)),
    # "tongyi_budget_seed0":                                                           list(range(3)),
    "tongyi_traj_budget_orig_ext_gpt-oss-120b_seed0":                               list(range(3)),  # complete
    "tongyi_traj_budget_orig_ext_tongyi_seed0":                               list(range(3)),  # complete
    "tongyi_traj_budget_orig_ext_qwen3.6-35b-a3b_seed0":                               list(range(3)),  # complete
    "tongyi_traj_budget_orig_ext_qwen3.5-4b_seed0":                               list(range(3)),  # complete
    "tongyi_traj_budget_orig_ext_qwen3.5-4b-sft_seed0":                               list(range(3)),  # complete
}

# Missing shards for bcp/train680 split (8 shards: 0-7).
MISSING_TRAIN680 = {
    # gpt-oss-120b — all modes
    # "gpt-oss-120b_seed0":                                                           list(range(8)),
    # "gpt-oss-120b_traj_ext_gpt-oss-120b_seed0":                                    list(range(8)),
    # "gpt-oss-120b_traj_orig_ext_gpt-oss-120b_seed0":                               list(range(8)),
    # "gpt-oss-120b_traj_summary_ext_gpt-oss-120b_seed0":                            list(range(8)),
    # "gpt-oss-120b_traj_summary_orig_ext_gpt-oss-120b_seed0":                       list(range(8)),
    # "gpt-oss-120b_traj_summary_ext_selected_tools_gpt-oss-120b_seed0":             list(range(8)),
    # "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0":        list(range(8)),
    # "gpt-oss-120b_budget_seed0":                                                    list(range(8)),
    # qwen3.6-35b-a3b — seed0 and budget only
    # "qwen3.6-35b-a3b_seed0":                                                        list(range(8)),
    # "qwen3.6-35b-a3b_budget_seed0":                                                 list(range(8)),
    # tongyi — all modes
    # "tongyi_seed0":                                                                  list(range(8)),
    # "tongyi_traj_ext_tongyi_seed0":                                                  list(range(8)),
    # "tongyi_traj_orig_ext_tongyi_seed0":                                             list(range(8)),
    # "tongyi_traj_summary_ext_tongyi_seed0":                                          list(range(8)),
    # "tongyi_traj_summary_orig_ext_tongyi_seed0":                                     list(range(8)),
    # "tongyi_budget_seed0":                                                           list(range(8)),
}


def parse_run_name(name):
    """Extract MODEL_NAME, mode, seed from a leaf folder name."""
    # Determine model
    if name.startswith("gpt-oss-120b"):
        model = "gpt-oss-120b"
        rest = name[len("gpt-oss-120b_"):]
    elif name.startswith("tongyi"):
        model = "tongyi"
        rest = name[len("tongyi_"):]
    elif name.startswith("qwen3.6-35b-a3b"):
        model = "qwen3.6-35b-a3b"
        rest = name[len("qwen3.6-35b-a3b_"):]
    elif name.startswith("qwen3.5-4b-sft"):
        model = "qwen3.5-4b-sft"
        rest = name[len("qwen3.5-4b-sft_"):]
    elif name.startswith("qwen3.5-4b"):
        model = "qwen3.5-4b"
        rest = name[len("qwen3.5-4b_"):]
    else:
        raise ValueError(f"Unknown model in: {name}")

    # Extract seed (always at end; may be "_seedN" or just "seedN" when mode is org)
    seed_match = re.search(r"_?seed(\d+)$", rest)
    if seed_match:
        seed = int(seed_match.group(1))
        rest = rest[:seed_match.start()]
    else:
        seed = 0  # no seed suffix — default to 0

    # Strip known plan-model names embedded in start_ext output dirs
    # e.g. "planning_v0.5_start_ext_gemini_2.5_pro" → "planning_v0.5_start_ext"
    rest = re.sub(r"(_start_ext)_gemini_2\.5_pro", r"\1", rest)

    # Handle traj_ext_{traj_model} and traj_summary_ext_{traj_model} patterns.
    # Extract TRAJ_MODEL before stripping so patch_sbatch can override it when needed.
    traj_model = None
    traj_model_patterns = [
        (r"^traj_budget_orig_ext_(.*)", "traj_budget_orig_ext"),
        (r"^traj_orig_ext_(.*)", "traj_orig_ext"),
        (r"^traj_summary_orig_ext_selected_tools_(.*)", "traj_summary_orig_ext_selected_tools"),
        (r"^traj_summary_orig_ext_(?!selected_tools)(.*)", "traj_summary_orig_ext"),
        (r"^traj_summary_ext_selected_tools_(.*)", "traj_summary_ext_selected_tools"),
        (r"^traj_summary_ext_(?!selected_tools)(.*)", "traj_summary_ext"),
        (r"^traj_ext_(.*)", "traj_ext"),
    ]
    for pattern, normalized_mode in traj_model_patterns:
        m = re.match(pattern, rest)
        if m:
            traj_model = m.group(1) or None
            rest = normalized_mode
            break

    # Aliases only — everything else passes through as-is
    aliases = {
        "":                    "org",
    }
    mode = aliases.get(rest, rest)

    return model, mode, seed, traj_model


def patch_sbatch(template: str, run_name: str, model: str, mode: str, seed: int,
                 shards: list = None, dataset: str = "bcp", split: str = "full",
                 traj_model: str = None) -> str:
    content = template

    # Patch SLURM directives
    if shards is not None:
        array_str = ",".join(str(s) for s in sorted(shards))
        content = re.sub(r"#SBATCH --array=.*", f"#SBATCH --array={array_str}", content)
    split_tag = f"_{split}" if split != "full" else ""
    content = re.sub(r"#SBATCH --job-name=.*",   f"#SBATCH --job-name={run_name}{split_tag}_missing", content)
    content = re.sub(r"#SBATCH --output=.*",     f"#SBATCH --output=sbatch_outputs/{run_name}{split_tag}_missing.out", content)

    # Patch shell variables
    content = re.sub(r'^MODEL_NAME=".*?"', f'MODEL_NAME="{model}"', content, flags=re.MULTILINE)
    content = re.sub(r'^mode=".*?"',       f'mode="{mode}"',        content, flags=re.MULTILINE)
    content = re.sub(r'^seed=\d+',         f'seed={seed}',          content, flags=re.MULTILINE)
    content = re.sub(r'^dataset=".*?"',    f'dataset="{dataset}"',  content, flags=re.MULTILINE)
    # Override TRAJ_MODEL when the trajectory source differs from the running model.
    # Templates default to TRAJ_MODEL="${MODEL_NAME}", which is wrong for cross-model traj modes.
    if traj_model is not None and traj_model != model:
        content = re.sub(r'^TRAJ_MODEL=.*', f'TRAJ_MODEL="{traj_model}"', content, flags=re.MULTILINE)

    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs (default: dry-run)")
    args = parser.parse_args()

    with open(TEMPLATE_PATH) as f:
        template_full = f.read()
    with open(TEMPLATE_PATH_FIRST50) as f:
        template_first50 = f.read()
    with open(TEMPLATE_PATH_TEST150) as f:
        template_test150 = f.read()
    with open(TEMPLATE_PATH_TRAIN680) as f:
        template_train680 = f.read()

    os.makedirs("sbatch_outputs", exist_ok=True)

    jobs = (
        [(run_name, shards, template_full,    "full",    "bcp")    for run_name, shards in MISSING.items()] +
        [(run_name, value,  template_first50, "first50", "bcp")    for run_name, value  in MISSING_FIRST50.items()] +
        [(run_name, value,  template_first50, "first50", "frames") for run_name, value  in MISSING_FRAMES_FIRST50.items()] +
        [(run_name, value,  template_first50, "first50", "musique") for run_name, value  in MISSING_MUSIQUE_FIRST50.items()] +
        [(run_name, shards, template_test150, "test150", "bcp")    for run_name, shards in MISSING_TEST150.items()] +
        [(run_name, shards, template_train680,"train680","bcp")    for run_name, shards in MISSING_TRAIN680.items()]
    )

    for run_name, shards, template, split, dataset in jobs:
        model, mode, seed, traj_model = parse_run_name(run_name)
        content = patch_sbatch(template, run_name, model, mode, seed, shards, dataset=dataset, split=split, traj_model=traj_model)

        print(f"\n{'='*60}")
        print(f"Run:    {run_name}")
        traj_suffix = f" | TrajModel: {traj_model}" if traj_model and traj_model != model else ""
        print(f"Split:  {split} | Model: {model} | Mode: {mode} | Seed: {seed}{traj_suffix}")
        if shards is not None:
            print(f"Shards: {','.join(str(s) for s in sorted(shards))}")

        if args.submit:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".SBATCH", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
            os.unlink(tmp_path)
            if result.returncode == 0:
                print(f"Submitted: {result.stdout.strip()}")
            else:
                print(f"ERROR: {result.stderr.strip()}")
            print("Sleeping 200s before next submission...")
            time.sleep(200)
        else:
            print("(dry-run — use --submit to actually submit)")


if __name__ == "__main__":
    main()
