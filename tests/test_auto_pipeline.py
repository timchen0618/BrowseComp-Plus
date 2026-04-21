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
