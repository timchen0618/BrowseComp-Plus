import auto_pipeline


def test_module_imports():
    assert hasattr(auto_pipeline, "Target")
    assert hasattr(auto_pipeline, "TargetState")
    assert hasattr(auto_pipeline, "PipelineState")


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
    run_dir = tmp_project / "runs" / "bcp" / "Qwen3-Embedding-8B" / "test150" / "gpt-oss-120b" / "x_seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_q1.json").write_text('{"query_id": "q1", "result": [1]}')
    (run_dir / "run_q3.json").write_text('{"query_id": "q3", "result": [1]}')

    t = auto_pipeline.Target(
        run_name="x_seed0", dataset="bcp", split="test150",
        template_path="run_qwen3_test150.SBATCH", declared_shards=[0, 1, 2],
        model="gpt-oss-120b", mode="org", seed=0, traj_model=None,
    )
    missing_shards, missing_qids = auto_pipeline.compute_actual_missing(
        t, retriever="Qwen3-Embedding-8B", agent_model="gpt-oss-120b"
    )
    assert set(missing_shards) == {0, 1, 2}
    assert set(missing_qids) == {"q2", "q4", "q5", "q6"}
