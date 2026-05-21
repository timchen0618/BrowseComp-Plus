"""Tests for new features in sft/axolotl/prepare_dataset.py:
  - mode d (multi-input): shortest successful, skip entirely if none succeed
  - single-input eval filtering: --eval-folder required, --keep-failed flag
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test via path (it lives outside the tests/ package)
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "sft" / "axolotl" / "prepare_dataset.py"
)
_spec = importlib.util.spec_from_file_location("prepare_dataset", _MODULE_PATH)
_pd = importlib.util.module_from_spec(_spec)
sys.modules["prepare_dataset"] = _pd
_spec.loader.exec_module(_pd)

_Candidate = _pd._Candidate
_select_mode_c = _pd._select_mode_c
_select_mode_d = _pd._select_mode_d
_print_multi_input_stats = _pd._print_multi_input_stats
MODE_A = _pd.MODE_A
MODE_B = _pd.MODE_B
MODE_C = _pd.MODE_C
MODE_D = _pd.MODE_D
_INF_LENGTH = _pd._INF_LENGTH
main = _pd.main


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cand(success: bool, length: float) -> _Candidate:
    return _Candidate(record={}, source_cache=None, success=success, subsequent_length=length)


# A minimal excerpt that passes _coerce_excerpt validation:
# one function_call item → one assistant message with <tool_call>.
_EXCERPT = json.dumps({
    "type": "function_call",
    "name": "search",
    "arguments": json.dumps({"query": "test"}),
})


def _write_source_traj(folder: Path, filename: str, qid: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(json.dumps({
        "query_id": qid,
        "original_messages": [
            {"role": "user", "content": f"System prompt.\nUser: What is {qid}?"}
        ],
        "result": [],
    }))


def _write_eval(folder: Path, qid: str, correct: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{qid}_eval.json").write_text(
        json.dumps({"query_id": qid, "judge_result": {"correct": correct}})
    )


def _write_subsequent_traj(folder: Path, qid: str, length: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{qid}.json").write_text(
        json.dumps({"query_id": qid, "result": list(range(length))})
    )


def _write_input_jsonl(path: Path, records: list) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _run_main(argv: list) -> None:
    old = sys.argv
    sys.argv = ["prepare_dataset.py"] + argv
    try:
        main()
    finally:
        sys.argv = old


def _read_train(out_dir: Path) -> list:
    return [json.loads(l) for l in (out_dir / "train.jsonl").read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Unit: _select_mode_d
# ---------------------------------------------------------------------------

def test_mode_d_empty_returns_none():
    assert _select_mode_d([]) is None


def test_mode_d_all_fail_returns_none():
    candidates = [_cand(False, 5), _cand(False, 3)]
    assert _select_mode_d(candidates) is None


def test_mode_d_single_failure_returns_none():
    assert _select_mode_d([_cand(False, 3)]) is None


def test_mode_d_single_success_returns_it():
    c = _cand(True, 3)
    assert _select_mode_d([c]) is c


def test_mode_d_all_success_picks_shortest():
    c1, c2, c3 = _cand(True, 10), _cand(True, 4), _cand(True, 7)
    assert _select_mode_d([c1, c2, c3]) is c2


def test_mode_d_mix_picks_shortest_successful_not_shortest_overall():
    # c_fail has length 1 (shortest overall) but is not successful
    c_fail = _cand(False, 1)
    c_success_long = _cand(True, 8)
    c_success_short = _cand(True, 5)
    result = _select_mode_d([c_fail, c_success_long, c_success_short])
    assert result is c_success_short


# ---------------------------------------------------------------------------
# Unit: mode D vs mode C contrast
# ---------------------------------------------------------------------------

def test_mode_d_and_c_agree_when_success_exists():
    rng = random.Random(0)
    candidates = [_cand(True, 5), _cand(False, 2)]
    assert _select_mode_d(candidates) is _select_mode_c(candidates, rng)


def test_mode_d_returns_none_when_no_success_but_c_picks_randomly():
    rng = random.Random(0)
    candidates = [_cand(False, 5), _cand(False, 2)]
    assert _select_mode_d(candidates) is None
    assert _select_mode_c(candidates, rng) is not None


# ---------------------------------------------------------------------------
# Unit: _print_multi_input_stats with mode D
# ---------------------------------------------------------------------------

def test_stats_mode_d_zero_success_shows_skipped_label(capsys):
    grouped = {
        "Q001": [_cand(False, 4)],   # 0 success
        "Q002": [_cand(True, 6)],    # 1 success
    }
    _print_multi_input_stats(grouped, MODE_D)
    out = capsys.readouterr().out
    assert "skipped (mode d)" in out
    assert "0 success: 1 queries" in out
    assert "1 success: 1 queries" in out


def test_stats_mode_c_zero_success_shows_avg_not_skipped(capsys):
    grouped = {
        "Q001": [_cand(False, 4)],
        "Q002": [_cand(True, 6)],
    }
    _print_multi_input_stats(grouped, MODE_C)
    out = capsys.readouterr().out
    assert "skipped (mode d)" not in out
    assert "avg trajectory length" in out


def test_stats_mode_d_all_success_skipped_count_is_zero(capsys):
    grouped = {"Q001": [_cand(True, 5)], "Q002": [_cand(True, 3)]}
    _print_multi_input_stats(grouped, MODE_D)
    out = capsys.readouterr().out
    # All queries succeed: the skipped line still appears but with count 0
    assert "Queries with 0 success (0 queries): skipped (mode d)" in out
    assert "1 success: 2 queries" in out


# ---------------------------------------------------------------------------
# E2E: single-input argument validation
# ---------------------------------------------------------------------------

def test_single_input_eval_folder_required(tmp_path):
    inp = tmp_path / "input.jsonl"
    inp.write_text("")
    traj = tmp_path / "trajs"
    traj.mkdir()
    with pytest.raises(SystemExit):
        _run_main([
            "--input", str(inp),
            "--trajectory-folder", str(traj),
            "--output-dir", str(tmp_path / "out"),
            "--split", "random",
        ])


def test_eval_folder_rejected_with_multi_input(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("[]")
    with pytest.raises(SystemExit):
        _run_main([
            "--multi-input", str(cfg),
            "--mode", "a",
            "--eval-folder", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--split", "random",
        ])


def test_keep_failed_rejected_with_multi_input(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("[]")
    with pytest.raises(SystemExit):
        _run_main([
            "--multi-input", str(cfg),
            "--mode", "a",
            "--keep-failed",
            "--output-dir", str(tmp_path / "out"),
            "--split", "random",
        ])


# ---------------------------------------------------------------------------
# E2E: single-input success filtering
# ---------------------------------------------------------------------------

def _setup_single_input(tmp_path, qid_success_map: dict):
    """Create trajectory + eval + input JSONL for single-input E2E tests."""
    traj_dir = tmp_path / "trajs"
    eval_dir = tmp_path / "evals"
    inp = tmp_path / "input.jsonl"
    records = []
    for qid, correct in qid_success_map.items():
        fname = f"traj_{qid}.json"
        _write_source_traj(traj_dir, fname, qid)
        _write_eval(eval_dir, qid, correct)
        records.append({"query_id": qid, "source_file": fname, "excerpt": _EXCERPT})
    _write_input_jsonl(inp, records)
    return traj_dir, eval_dir, inp


def test_single_input_default_keeps_only_successful(tmp_path):
    traj_dir, eval_dir, inp = _setup_single_input(
        tmp_path, {"Q001": True, "Q002": False}
    )
    out_dir = tmp_path / "out"
    _run_main([
        "--input", str(inp),
        "--trajectory-folder", str(traj_dir),
        "--eval-folder", str(eval_dir),
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
    ])
    train = _read_train(out_dir)
    assert len(train) == 1  # only Q001 (successful)


def test_single_input_keep_failed_includes_all(tmp_path):
    traj_dir, eval_dir, inp = _setup_single_input(
        tmp_path, {"Q001": True, "Q002": False}
    )
    out_dir = tmp_path / "out"
    _run_main([
        "--input", str(inp),
        "--trajectory-folder", str(traj_dir),
        "--eval-folder", str(eval_dir),
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
        "--keep-failed",
    ])
    train = _read_train(out_dir)
    assert len(train) == 2  # both Q001 and Q002


def test_single_input_all_failed_no_output(tmp_path):
    traj_dir, eval_dir, inp = _setup_single_input(
        tmp_path, {"Q001": False, "Q002": False}
    )
    with pytest.raises(SystemExit):
        _run_main([
            "--input", str(inp),
            "--trajectory-folder", str(traj_dir),
            "--eval-folder", str(eval_dir),
            "--output-dir", str(tmp_path / "out"),
            "--split", "random", "--val-size", "0",
        ])


def test_single_input_missing_qid_in_eval_raises_error(tmp_path):
    traj_dir = tmp_path / "trajs"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    inp = tmp_path / "input.jsonl"
    _write_source_traj(traj_dir, "traj_Q001.json", "Q001")
    # No eval file for Q001
    _write_input_jsonl(inp, [
        {"query_id": "Q001", "source_file": "traj_Q001.json", "excerpt": _EXCERPT}
    ])
    with pytest.raises(SystemExit) as exc_info:
        _run_main([
            "--input", str(inp),
            "--trajectory-folder", str(traj_dir),
            "--eval-folder", str(eval_dir),
            "--output-dir", str(tmp_path / "out"),
            "--split", "random",
        ])
    assert "not found in --eval-folder" in str(exc_info.value.code)


def test_single_input_missing_qid_raises_even_with_keep_failed(tmp_path):
    traj_dir = tmp_path / "trajs"
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    inp = tmp_path / "input.jsonl"
    _write_source_traj(traj_dir, "traj_Q001.json", "Q001")
    _write_input_jsonl(inp, [
        {"query_id": "Q001", "source_file": "traj_Q001.json", "excerpt": _EXCERPT}
    ])
    with pytest.raises(SystemExit) as exc_info:
        _run_main([
            "--input", str(inp),
            "--trajectory-folder", str(traj_dir),
            "--eval-folder", str(eval_dir),
            "--output-dir", str(tmp_path / "out"),
            "--split", "random",
            "--keep-failed",
        ])
    assert "not found in --eval-folder" in str(exc_info.value.code)


# ---------------------------------------------------------------------------
# E2E: multi-input mode D
# ---------------------------------------------------------------------------

def _setup_multi_input_spec(base: Path, records_meta: list) -> dict:
    """
    Build one multi-input spec entry under `base`.
    records_meta: list of (qid, correct, subseq_length).
    Returns the spec dict for the config JSON.
    """
    traj_dir = base / "trajs"
    subseq_dir = base / "subseq"
    eval_dir = base / "eval"
    inp = base / "input.jsonl"
    records = []
    for qid, correct, length in records_meta:
        fname = f"traj_{qid}.json"
        _write_source_traj(traj_dir, fname, qid)
        _write_subsequent_traj(subseq_dir, qid, length)
        _write_eval(eval_dir, qid, correct)
        records.append({"query_id": qid, "source_file": fname, "excerpt": _EXCERPT})
    _write_input_jsonl(inp, records)
    return {
        "input": str(inp),
        "trajectory_folder": str(traj_dir),
        "subsequent_folder": str(subseq_dir),
        "subsequent_eval_folder": str(eval_dir),
    }


def _write_config(path: Path, specs: list) -> None:
    path.write_text(json.dumps(specs))


def test_mode_d_skips_query_with_no_success(tmp_path):
    """Q001 succeeds, Q002 fails — mode D outputs only Q001."""
    spec = _setup_multi_input_spec(tmp_path / "spec", [
        ("Q001", True, 5),
        ("Q002", False, 3),
    ])
    cfg = tmp_path / "config.json"
    _write_config(cfg, [spec])
    out_dir = tmp_path / "out"
    _run_main([
        "--multi-input", str(cfg), "--mode", "d",
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
    ])
    assert len(_read_train(out_dir)) == 1


def test_mode_d_all_queries_successful(tmp_path):
    """All queries succeed — mode D outputs all of them."""
    spec = _setup_multi_input_spec(tmp_path / "spec", [
        ("Q001", True, 5),
        ("Q002", True, 3),
    ])
    cfg = tmp_path / "config.json"
    _write_config(cfg, [spec])
    out_dir = tmp_path / "out"
    _run_main([
        "--multi-input", str(cfg), "--mode", "d",
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
    ])
    assert len(_read_train(out_dir)) == 2


def test_mode_d_no_successes_raises_no_usable_examples(tmp_path):
    """All queries fail — mode D skips all → SystemExit 'No usable examples'."""
    spec = _setup_multi_input_spec(tmp_path / "spec", [("Q001", False, 5)])
    cfg = tmp_path / "config.json"
    _write_config(cfg, [spec])
    with pytest.raises(SystemExit) as exc_info:
        _run_main([
            "--multi-input", str(cfg), "--mode", "d",
            "--output-dir", str(tmp_path / "out"),
            "--split", "random", "--val-size", "0",
        ])
    assert "No usable examples" in str(exc_info.value.code)


def test_mode_d_vs_c_on_all_fail_queries(tmp_path):
    """With all-fail candidates, mode C picks one; mode D skips (SystemExit)."""
    spec_d = _setup_multi_input_spec(tmp_path / "d", [("Q001", False, 5)])
    spec_c = _setup_multi_input_spec(tmp_path / "c", [("Q001", False, 5)])

    # mode D: no output
    cfg_d = tmp_path / "d" / "config.json"
    _write_config(cfg_d, [spec_d])
    with pytest.raises(SystemExit):
        _run_main([
            "--multi-input", str(cfg_d), "--mode", "d",
            "--output-dir", str(tmp_path / "d" / "out"),
            "--split", "random", "--val-size", "0",
        ])

    # mode C: one example kept (random fallback)
    cfg_c = tmp_path / "c" / "config.json"
    _write_config(cfg_c, [spec_c])
    out_c = tmp_path / "c" / "out"
    _run_main([
        "--multi-input", str(cfg_c), "--mode", "c",
        "--output-dir", str(out_c),
        "--split", "random", "--val-size", "0",
    ])
    assert len(_read_train(out_c)) == 1


def test_mode_d_picks_shortest_successful_across_two_specs(tmp_path):
    """Two specs for Q001, both succeed: lengths 8 and 3. Mode D picks the shorter."""
    traj_dir = tmp_path / "trajs"
    subseq1 = tmp_path / "subseq1"
    subseq2 = tmp_path / "subseq2"
    eval1 = tmp_path / "eval1"
    eval2 = tmp_path / "eval2"
    inp1 = tmp_path / "input1.jsonl"
    inp2 = tmp_path / "input2.jsonl"

    fname = "traj_Q001.json"
    _write_source_traj(traj_dir, fname, "Q001")
    _write_subsequent_traj(subseq1, "Q001", 8)
    _write_subsequent_traj(subseq2, "Q001", 3)
    _write_eval(eval1, "Q001", True)
    _write_eval(eval2, "Q001", True)

    rec = {"query_id": "Q001", "source_file": fname, "excerpt": _EXCERPT}
    _write_input_jsonl(inp1, [rec])
    _write_input_jsonl(inp2, [rec])

    cfg = tmp_path / "config.json"
    _write_config(cfg, [
        {"input": str(inp1), "trajectory_folder": str(traj_dir),
         "subsequent_folder": str(subseq1), "subsequent_eval_folder": str(eval1)},
        {"input": str(inp2), "trajectory_folder": str(traj_dir),
         "subsequent_folder": str(subseq2), "subsequent_eval_folder": str(eval2)},
    ])
    out_dir = tmp_path / "out"
    _run_main([
        "--multi-input", str(cfg), "--mode", "d",
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
    ])
    # Both candidates are successful, mode D picks one (the shortest) → 1 example
    assert len(_read_train(out_dir)) == 1


def test_mode_d_summary_print_includes_skipped_count(tmp_path, capsys):
    """Verify the summary line reports query_ids_skipped_no_success for mode D."""
    spec = _setup_multi_input_spec(tmp_path / "spec", [
        ("Q001", True, 5),   # success
        ("Q002", False, 3),  # fail → skipped by mode D
    ])
    cfg = tmp_path / "config.json"
    _write_config(cfg, [spec])
    out_dir = tmp_path / "out"
    _run_main([
        "--multi-input", str(cfg), "--mode", "d",
        "--output-dir", str(out_dir),
        "--split", "random", "--val-size", "0",
    ])
    out = capsys.readouterr().out
    assert "query_ids_skipped_no_success=1" in out
    assert "selected_examples=1" in out
