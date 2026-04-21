import json
from pathlib import Path
import pytest


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch):
    """Temp project dir with minimal subdirs the pipeline writes to."""
    (tmp_path / "sbatch_outputs").mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "evals").mkdir()
    (tmp_path / "topics-qrels" / "bcp" / "bcp_test150_3_shards").mkdir(parents=True)
    # 3 shards, 2 qids each
    for i, qids in enumerate([["q1", "q2"], ["q3", "q4"], ["q5", "q6"]]):
        (tmp_path / "topics-qrels" / "bcp" / "bcp_test150_3_shards" / f"q_{i}.tsv").write_text(
            "\n".join(f"{q}\tsome text" for q in qids) + "\n"
        )
    (tmp_path / "topics-qrels" / "bcp" / "queries_test150.tsv").write_text(
        "\n".join(f"{q}\tsome text" for q in ["q1", "q2", "q3", "q4", "q5", "q6"]) + "\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sample_slurm_out(tmp_project: Path) -> Path:
    """An example SLURM output with a common error pattern."""
    path = tmp_project / "slurm-9999.out"
    path.write_text(
        "Starting job ...\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        "Traceback (most recent call last):\n"
        '  File "/app/oss_client.py", line 42, in <module>\n'
        "    main()\n"
        "RuntimeError: CUDA error\n"
    )
    return path
