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

# Missing shards per leaf folder (from missing_1.txt analysis)
MISSING = {
    "tongyi_planning_new_prompt_start_ext_seed0": [0, 1, 2, 9],
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
    else:
        raise ValueError(f"Unknown model in: {name}")

    # Extract seed (always at end; may be "_seedN" or just "seedN" when mode is org)
    seed_match = re.search(r"_?seed(\d+)$", rest)
    if not seed_match:
        raise ValueError(f"No seed found in: {name}")
    seed = int(seed_match.group(1))
    rest = rest[:seed_match.start()]

    # Map remainder to mode
    mode_map = {
        "planning_after_steps_5":          "planning_after_steps_5",
        "planning_start_and_after_steps_5": "planning_start_and_after_steps_5",
        "planning_start_ext":               "planning_start_ext",
        "planning_new_prompt_start_ext":    "planning_start_ext",
        "planning":                         "planning",
        "planning_new_prompt":              "planning",
        "":                                 "org",
    }
    mode = mode_map.get(rest)
    if mode is None:
        raise ValueError(f"Unknown mode fragment '{rest}' in: {name}")

    return model, mode, seed


def patch_sbatch(template: str, run_name: str, model: str, mode: str, seed: int, shards: list) -> str:
    array_str = ",".join(str(s) for s in sorted(shards))
    content = template

    # Patch SLURM directives
    content = re.sub(r"#SBATCH --array=.*",      f"#SBATCH --array={array_str}", content)
    content = re.sub(r"#SBATCH --job-name=.*",   f"#SBATCH --job-name={run_name}_missing", content)
    content = re.sub(r"#SBATCH --output=.*",     f"#SBATCH --output=sbatch_outputs/{run_name}_missing.out", content)

    # Patch shell variables
    content = re.sub(r'^MODEL_NAME=".*?"', f'MODEL_NAME="{model}"', content, flags=re.MULTILINE)
    content = re.sub(r'^mode=".*?"',       f'mode="{mode}"',        content, flags=re.MULTILINE)
    content = re.sub(r'^seed=\d+',         f'seed={seed}',          content, flags=re.MULTILINE)

    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs (default: dry-run)")
    args = parser.parse_args()

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    os.makedirs("sbatch_outputs", exist_ok=True)

    for run_name, shards in MISSING.items():
        model, mode, seed = parse_run_name(run_name)
        content = patch_sbatch(template, run_name, model, mode, seed, shards)

        array_str = ",".join(str(s) for s in sorted(shards))
        print(f"\n{'='*60}")
        print(f"Run:    {run_name}")
        print(f"Model:  {model} | Mode: {mode} | Seed: {seed}")
        print(f"Shards: {array_str}")

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
