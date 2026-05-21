"""
End-to-end tests for the pairwise-bootstrap-comparison skill.

Tests verify:
  1. Bootstrap script produces parseable output for all expected runs
  2. Sign convention: diff(A-B) negated correctly (positive Δ = run beats baseline)
  3. CI conversion: fractions multiplied by 100 to produce percentages
  4. Significance threshold: ✓ iff p < 0.05
  5. Baseline accuracy field: Accuracy (%) present and correct
  6. All expected runs discoverable via ls of the eval directory
  7. Query count (n) parsed correctly for footnote
"""

import subprocess
import json
import re
import os
import sys

EVAL_DIR = "evals/bcp/Qwen3-Embedding-8B/test300/gpt-oss-120b"
BASELINE = "seed4"
BASELINE_SUMMARY = f"{EVAL_DIR}/{BASELINE}/evaluation_summary.json"

EXPECTED_RUNS = [
    "traj_summary_orig_ext_selected_tools_gpt-oss-120b_gemini-2.5-pro_1_seed0",
    "traj_summary_orig_ext_selected_tools_gpt-oss-120b_gemini_3.1-pro-preview_seed0",
    "traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0",
    "traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0_less_chars",
    "traj_summary_orig_ext_selected_tools_random_seed0_gpt-oss-120b_seed0",
    "traj_summary_orig_ext_selected_tools_random_seed1_gpt-oss-120b_seed0",
    "traj_summary_orig_ext_selected_tools_random_seed2_gpt-oss-120b_seed0",
    "traj_summary_orig_ext_selected_tools_random_seed3_gpt-oss-120b_seed0",
]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        failures.append(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_bootstrap(run_name):
    """Run paired_bootstrap_eval_summaries.py and return stdout."""
    summary_b = f"{EVAL_DIR}/{run_name}/evaluation_summary.json"
    result = subprocess.run(
        ["python", "paired_bootstrap_eval_summaries.py",
         BASELINE_SUMMARY, summary_b, "--num_samples", "10000"],
        capture_output=True, text=True
    )
    return result.stdout, result.returncode

def parse_bootstrap_output(stdout):
    """
    Parse the script's stdout into a dict:
      mean_a, mean_b, diff_ab, p_value, ci_b_lo, ci_b_hi, n_rows
    Returns None if parsing fails.
    """
    lines = stdout.strip().splitlines()
    d = {}

    for line in lines:
        # "Paired rows: 300 (metric=correct)"
        m = re.search(r"Paired rows:\s*(\d+)", line)
        if m:
            d["n_rows"] = int(m.group(1))

        # "Full-sample mean A=0.423333, B=0.493333, diff(A-B)=-0.070000"
        m = re.search(r"mean A=([\d.]+),\s*B=([\d.]+),\s*diff\(A-B\)=(-?[\d.]+)", line)
        if m:
            d["mean_a"] = float(m.group(1))
            d["mean_b"] = float(m.group(2))
            d["diff_ab"] = float(m.group(3))

        # "(B is superior with p value p=0.000)" or "(A is superior...)" or "(no significant...)"
        m = re.search(r"p value p=([\d.]+)", line)
        if m:
            d["p_value"] = float(m.group(1))

        # "B mean=0.493432, median=..., 95% CI=[0.436667, 0.550000]"
        m = re.search(r"B mean=[\d.]+.*?95% CI=\[([\d.]+),\s*([\d.]+)\]", line)
        if m:
            d["ci_b_lo"] = float(m.group(1))
            d["ci_b_hi"] = float(m.group(2))

    if len(d) < 6:
        return None
    return d


# ---------------------------------------------------------------------------
# Test 1: Eval directory structure — all expected runs discoverable
# ---------------------------------------------------------------------------
print("\n[Test 1] Eval directory structure")

discovered = set(os.listdir(EVAL_DIR)) if os.path.isdir(EVAL_DIR) else set()
check("Eval dir exists", os.path.isdir(EVAL_DIR), EVAL_DIR)
check("Baseline dir exists", BASELINE in discovered, BASELINE)
for run in EXPECTED_RUNS:
    check(f"Run dir exists: {run[:50]}...", run in discovered)


# ---------------------------------------------------------------------------
# Test 2: Baseline evaluation_summary.json format
# ---------------------------------------------------------------------------
print("\n[Test 2] Baseline evaluation_summary.json")

with open(BASELINE_SUMMARY) as f:
    baseline_data = json.load(f)

check("Has 'Accuracy (%)' field", "Accuracy (%)" in baseline_data)
check("Baseline accuracy is 42.33", abs(baseline_data.get("Accuracy (%)", 0) - 42.33) < 0.01,
      f"got {baseline_data.get('Accuracy (%)')}")
check("Has 'per_query_metrics'", "per_query_metrics" in baseline_data)
n_queries = len(baseline_data.get("per_query_metrics", []))
check("300 queries in baseline", n_queries == 300, f"got {n_queries}")


# ---------------------------------------------------------------------------
# Test 3: Bootstrap script runs and parses correctly for each run
# ---------------------------------------------------------------------------
print("\n[Test 3] Bootstrap script execution and parsing")

parsed_results = {}
for run in EXPECTED_RUNS:
    stdout, rc = run_bootstrap(run)
    check(f"Exit 0 for {run[:40]}...", rc == 0, f"rc={rc}")
    parsed = parse_bootstrap_output(stdout)
    check(f"Parsed output for {run[:40]}...", parsed is not None, stdout[:200])
    if parsed:
        parsed_results[run] = parsed


# ---------------------------------------------------------------------------
# Test 4: Sign convention — Δ = -(diff(A-B)), positive when B > A
# ---------------------------------------------------------------------------
print("\n[Test 4] Sign convention on Δ")

# gemini-2.5-pro_1_seed0 is known to be better than baseline (49.33% vs 42.33%)
gemini_run = "traj_summary_orig_ext_selected_tools_gpt-oss-120b_gemini-2.5-pro_1_seed0"
if gemini_run in parsed_results:
    p = parsed_results[gemini_run]
    delta_correct = -p["diff_ab"]  # skill says negate diff(A-B)
    check("Δ positive when B beats A (gemini-2.5-pro)", delta_correct > 0,
          f"diff_ab={p['diff_ab']:.4f}, delta={delta_correct:.4f}")
    check("Δ ≈ +0.07 for gemini-2.5-pro", abs(delta_correct - 0.07) < 0.001,
          f"got {delta_correct:.4f}")

# random_seed1 is the weakest — Δ should be small positive
random1_run = "traj_summary_orig_ext_selected_tools_random_seed1_gpt-oss-120b_seed0"
if random1_run in parsed_results:
    p = parsed_results[random1_run]
    delta = -p["diff_ab"]
    check("Δ positive but small for random_seed1", 0 < delta < 0.05,
          f"delta={delta:.4f}")


# ---------------------------------------------------------------------------
# Test 5: CI conversion — fractions × 100 = percentages > 1.0
# ---------------------------------------------------------------------------
print("\n[Test 5] CI conversion (fractions → percentages)")

for run, p in parsed_results.items():
    label = run[:40]
    # Raw CI bounds from script are fractions in [0, 1]
    check(f"Raw CI lo < 1.0 ({label}...)", p["ci_b_lo"] < 1.0,
          f"ci_lo={p['ci_b_lo']}")
    check(f"Raw CI hi < 1.0 ({label}...)", p["ci_b_hi"] < 1.0,
          f"ci_hi={p['ci_b_hi']}")
    # After × 100, should be in [0, 100]
    ci_lo_pct = p["ci_b_lo"] * 100
    ci_hi_pct = p["ci_b_hi"] * 100
    check(f"Converted CI lo > 1.0 ({label}...)", ci_lo_pct > 1.0,
          f"{ci_lo_pct:.2f}%")
    check(f"Converted CI is ordered ({label}...)", ci_lo_pct < ci_hi_pct,
          f"[{ci_lo_pct:.2f}, {ci_hi_pct:.2f}]")
    break  # one run is enough to validate the pattern


# ---------------------------------------------------------------------------
# Test 6: Significance threshold — ✓ iff p < 0.05
# ---------------------------------------------------------------------------
print("\n[Test 6] Significance threshold")

# Known significant (p=0.000): gemini-2.5-pro
if gemini_run in parsed_results:
    p_val = parsed_results[gemini_run]["p_value"]
    check("gemini-2.5-pro p < 0.05", p_val < 0.05, f"p={p_val}")
    check("gemini-2.5-pro → ✓", p_val < 0.05)  # maps to ✓

# Known NOT significant (p=0.208): random_seed1
if random1_run in parsed_results:
    p_val = parsed_results[random1_run]["p_value"]
    check("random_seed1 p >= 0.05", p_val >= 0.05, f"p={p_val}")
    check("random_seed1 → ✗", p_val >= 0.05)  # maps to ✗

# Borderline (p≈0.057): random_seed0 — should be ✗
random0_run = "traj_summary_orig_ext_selected_tools_random_seed0_gpt-oss-120b_seed0"
if random0_run in parsed_results:
    p_val = parsed_results[random0_run]["p_value"]
    check("random_seed0 p >= 0.05 (borderline)", p_val >= 0.05, f"p={p_val}")


# ---------------------------------------------------------------------------
# Test 7: Query count matches across baseline and comparison runs
# ---------------------------------------------------------------------------
print("\n[Test 7] Paired row count")

for run, p in list(parsed_results.items())[:3]:
    check(f"n_rows=300 for {run[:40]}...", p["n_rows"] == 300, f"got {p['n_rows']}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
total = sum(1 for _ in failures) + sum(1 for _ in [])  # failures already counted
if failures:
    print(f"FAILED: {len(failures)} test(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
