import auto_pipeline


def test_module_imports():
    assert hasattr(auto_pipeline, "Target")
    assert hasattr(auto_pipeline, "TargetState")
    assert hasattr(auto_pipeline, "PipelineState")


def test_parse_run_name_random_selected_tools_baseline():
    import submit_missing as sm

    m, mode, seed, tm = sm.parse_run_name(
        "gpt-oss-120b_traj_summary_orig_ext_selected_tools_random_seed42_gpt-oss-120b_seed0"
    )
    assert m == "gpt-oss-120b"
    assert mode == "traj_summary_orig_ext_selected_tools_random_seed42"
    assert seed == 0
    assert tm == "gpt-oss-120b"

    m2, mode2, seed2, tm2 = sm.parse_run_name(
        "gpt-oss-120b_traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0"
    )
    assert mode2 == "traj_summary_orig_ext_selected_tools"
    assert tm2 == "gpt-oss-120b"
    assert seed2 == 0


def test_resolve_run_subdir_random_selected_tools():
    t = auto_pipeline.Target(
        run_name="gpt-oss-120b_traj_summary_orig_ext_selected_tools_random_seed42_gpt-oss-120b_seed0",
        dataset="bcp",
        split="test150",
        template_path="run_qwen3_test150.SBATCH",
        declared_shards=[0],
        model="gpt-oss-120b",
        mode="traj_summary_orig_ext_selected_tools_random_seed42",
        seed=0,
        traj_model="gpt-oss-120b",
    )
    assert auto_pipeline._resolve_run_subdir(t) == (
        "traj_summary_orig_ext_selected_tools_random_seed42_gpt-oss-120b_seed0"
    )


def test_collect_targets_reads_missing_dicts(monkeypatch):
    import submit_missing
    monkeypatch.setattr(submit_missing, "MISSING", {"tongyi_seed3": [0, 1]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FRAMES_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_MUSIQUE_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TEST150",
                        {"gpt-oss-120b_seed0": [2]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TRAIN680", {}, raising=False)

    targets = auto_pipeline.collect_targets()
    assert len(targets) == 2
    names = {t.run_name for t in targets}
    assert names == {"tongyi_seed3", "gpt-oss-120b_seed0"}
    tongyi = next(t for t in targets if t.run_name == "tongyi_seed3")
    assert tongyi.split == "full"
    assert tongyi.dataset == "bcp"
    assert tongyi.model == "tongyi"
    assert tongyi.seed == 3
    assert tongyi.declared_shards == [0, 1]
    test150 = next(t for t in targets if t.run_name == "gpt-oss-120b_seed0")
    assert test150.split == "test150"
    assert test150.declared_shards == [2]


def test_preflight_detects_missing_template(tmp_project, monkeypatch):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="does_not_exist.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    errors = auto_pipeline.preflight([t], run_sbatch_check=False)
    assert len(errors) == 1
    assert "does_not_exist.SBATCH" in errors[0].reason


def test_preflight_runs_sbatch_test_only(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=test\n"
        "#SBATCH --output=sbatch_outputs/test.out\n"
        "#SBATCH --array=0\n"
        'MODEL_NAME="tongyi"\n'
        'mode="org"\n'
        'seed=0\n'
        'dataset="bcp"\n'
        "echo ok\n"
    )
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, stdout="sbatch: Job 123 would be submitted\n", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    errors = auto_pipeline.preflight([t], run_sbatch_check=True)
    assert errors == []
    assert any("--test-only" in c for c in calls[0])


def test_compute_actual_missing_reads_run_dir(tmp_project):
    # Disk subdir for mode=org,seed=0 is "seed0" (not the run_name).
    run_dir = tmp_project / "runs" / "bcp" / "Qwen3-Embedding-8B" / "test150" / "gpt-oss-120b" / "seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_q1.json").write_text('{"query_id": "q1", "result": [1]}')
    (run_dir / "run_q3.json").write_text('{"query_id": "q3", "result": [1]}')

    t = auto_pipeline.Target(
        run_name="gpt-oss-120b_seed0", dataset="bcp", split="test150",
        template_path="run_qwen3_test150.SBATCH", declared_shards=[0, 1, 2],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    missing_shards, missing_qids = auto_pipeline.compute_actual_missing(
        t, retriever="Qwen3-Embedding-8B", agent_model="gpt-oss-120b"
    )
    assert set(missing_shards) == {0, 1, 2}
    assert set(missing_qids) == {"q2", "q4", "q5", "q6"}


def _mini_sbatch_text():
    return (
        "#!/bin/bash\n"
        "#SBATCH --job-name=x\n"
        "#SBATCH --output=sbatch_outputs/x.out\n"
        "#SBATCH --array=0\n"
        'MODEL_NAME="tongyi"\nmode="org"\nseed=0\ndataset="bcp"\necho ok\n'
    )


def test_submit_target_dry_run_returns_none(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(_mini_sbatch_text())
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0, 1],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    jid = auto_pipeline.submit_target(t, shards=[0, 1], submit=False)
    assert jid is None


def test_submit_target_submits_and_parses_jobid(tmp_project, monkeypatch):
    template = tmp_project / "mini.SBATCH"
    template.write_text(_mini_sbatch_text())
    t = auto_pipeline.Target(
        run_name="tongyi_seed0", dataset="bcp", split="test150",
        template_path=str(template), declared_shards=[0, 1],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, stdout="Submitted batch job 987654\n", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    jid = auto_pipeline.submit_target(t, shards=[0, 1], submit=True)
    assert jid == 987654


def test_poll_jobs_parses_squeue(monkeypatch):
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        out = "987654 RUNNING\n987655 PENDING\n"
        return sp.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    states = auto_pipeline.poll_jobs([987654, 987655, 987656])
    assert states[987654] == "RUNNING"
    assert states[987655] == "PENDING"
    assert states[987656] == "DONE"


def test_poll_jobs_empty_list():
    assert auto_pipeline.poll_jobs([]) == {}


def _mk_target_state(name, missing, last, stuck):
    t = auto_pipeline.Target(
        run_name=name, dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    return auto_pipeline.TargetState(
        target=t, missing_qids=missing, last_missing_qids=last, stuck_cycles=stuck,
    )


def test_detect_stuck_targets_flags_unchanged():
    states = [
        _mk_target_state("A", ["q1", "q2"], ["q1", "q2"], 2),
        _mk_target_state("B", ["q3"], ["q4", "q3"], 1),
        _mk_target_state("C", [], [], 0),
    ]
    stuck = auto_pipeline.detect_stuck_targets(states, threshold=3)
    assert [s.target.run_name for s in stuck] == ["A"]
    assert states[1].stuck_cycles == 0


def test_diagnose_slurm_out_finds_oom(tmp_project, sample_slurm_out):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    text = auto_pipeline.diagnose_slurm_out(
        t, slurm_glob="slurm-*.out", sbatch_glob="sbatch_outputs/*.out"
    )
    assert "CUDA out of memory" in text
    assert "slurm-9999.out" in text


def test_save_and_load_state_roundtrip(tmp_project):
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0, 1],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    ts = auto_pipeline.TargetState(
        target=t, missing_qids=["q1"], last_missing_qids=[], stuck_cycles=0,
        submitted_job_ids=[42], status="running",
    )
    state = auto_pipeline.PipelineState(
        pid=123, started_at="2026-04-20T10:00:00", phase="monitoring",
        interval_seconds=600, stuck_threshold=3, targets=[ts],
    )
    path = tmp_project / "auto_pipeline_state.json"
    auto_pipeline.save_state(state, path)
    loaded = auto_pipeline.load_state(path)
    assert loaded.pid == 123
    assert loaded.phase == "monitoring"
    assert len(loaded.targets) == 1
    assert loaded.targets[0].target.run_name == "x_seed0"
    assert loaded.targets[0].missing_qids == ["q1"]
    assert loaded.targets[0].submitted_job_ids == [42]


def test_build_eval_sbatch_generates_one_exec_per_target(tmp_project):
    header_path = tmp_project / "eval.SBATCH"
    header_path.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=eval\n"
        "#SBATCH --output=sbatch_outputs/eval.out\n"
        "SINGULARITY_IMAGE=/img.sif\n"
        "OVERLAY_FILE=/ov.ext3\n"
        'BASE="runs/bcp/Qwen3-Embedding-8B"\n'
        'CMD_PREFIX="python eval.py"\n'
        "# --- autogenerated eval lines below ---\n"
    )
    t1 = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    t2 = auto_pipeline.Target(
        run_name="y_seed0", dataset="bcp", split="full",
        template_path="y.SBATCH", declared_shards=[0],
        model="tongyi", mode="org", seed=0, traj_model=None,
    )
    out = tmp_project / "eval_auto.SBATCH"
    auto_pipeline.build_eval_sbatch(
        [t1, t2], eval_template=header_path, out_path=out,
        retriever="Qwen3-Embedding-8B",
    )
    content = out.read_text()
    assert "x_seed0" in content
    assert "y_seed0" in content
    assert "test150/gpt-oss-120b/x_seed0" in content
    assert "full/tongyi/y_seed0" in content
    assert "singularity exec" in content


def test_write_summary_from_eval_out(tmp_project):
    eval_out = tmp_project / "sbatch_outputs" / "eval_auto.out"
    eval_out.write_text(
        "Processed 150 evaluations\n"
        "Accuracy: 65.3%\n"
        "Recall: 72.1%\n"
        "Average Tool Calls: {'search': 8.4}\n"
        "Summary saved to evals/bcp/Qwen3-Embedding-8B/test150/gpt-oss-120b/x_seed0/evaluation_summary.json\n"
    )
    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="x.SBATCH", declared_shards=[0],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    state = auto_pipeline.PipelineState(
        pid=1, started_at="2026-04-20T10:00:00", last_check_at="2026-04-20T12:00:00",
        cycle_count=1, phase="done", interval_seconds=7200, stuck_threshold=3,
        targets=[auto_pipeline.TargetState(target=t, status="complete")],
    )
    out = tmp_project / "pipeline_summary.md"
    auto_pipeline.write_summary(state, eval_out_path=eval_out, path=out)
    text = out.read_text()
    assert "65.3" in text
    assert "x_seed0" in text
    assert "cycle_count" in text.lower() or "cycles" in text.lower()


def test_main_dry_run_does_not_submit(tmp_project, monkeypatch):
    """Smoke test: dry-run flow completes without calling sbatch."""
    import submit_missing
    from pathlib import Path as _P
    import sys as _sys
    for name in ("run_qwen3_planning.SBATCH", "run_qwen3_first50.SBATCH",
                 "run_qwen3_test150.SBATCH", "run_qwen3_train680.SBATCH"):
        src = _P("/scratch/hc3337/projects/BrowseComp-Plus") / name
        (tmp_project / name).write_text(src.read_text())
    monkeypatch.setattr(submit_missing, "MISSING", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_FRAMES_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_MUSIQUE_FIRST50", {}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TEST150",
                        {"gpt-oss-120b_seed0": [0]}, raising=False)
    monkeypatch.setattr(submit_missing, "MISSING_TRAIN680", {}, raising=False)
    # Pretend everything's already done so the loop exits immediately.
    run_dir = tmp_project / "runs" / "bcp" / "Qwen3-Embedding-8B" / "test150" / "gpt-oss-120b" / "gpt-oss-120b_seed0"
    run_dir.mkdir(parents=True)
    for qid in ["q1", "q2", "q3", "q4", "q5", "q6"]:
        (run_dir / f"run_{qid}.json").write_text(f'{{"query_id": "{qid}"}}')
    sbatch_calls = []
    def fake_run(cmd, *a, **kw):
        import subprocess as sp
        sbatch_calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_run)
    (tmp_project / "eval.SBATCH").write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=eval\n#SBATCH --output=sbatch_outputs/eval.out\n"
        'BASE="runs/bcp/Qwen3-Embedding-8B"\n'
        'CMD_PREFIX="python eval.py"\n'
    )
    monkeypatch.setattr(_sys, "argv",
                        ["auto_pipeline.py", "--skip-preflight", "--skip-eval"])
    rc = auto_pipeline.main()
    assert rc == 0
    assert not any(c and c[0] == "sbatch" for c in sbatch_calls)
