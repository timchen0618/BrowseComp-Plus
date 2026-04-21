import auto_pipeline


def test_module_imports():
    assert hasattr(auto_pipeline, "Target")
    assert hasattr(auto_pipeline, "TargetState")
    assert hasattr(auto_pipeline, "PipelineState")
