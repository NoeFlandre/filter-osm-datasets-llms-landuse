from pathlib import Path

SCRIPT = (Path(__file__).parents[2] / "scripts" / "node_job.sh").read_text()


def test_cuda_home_is_exported_before_sglang_starts():
    """Regression (job 4165500): DeepEP asserts CUDA_HOME at SGLang import time."""
    assert SCRIPT.index("export CUDA_HOME") < SCRIPT.index('exec "$venv/bin/luf"')
    assert "module load" in SCRIPT


def test_env_is_built_once_under_a_lock():
    assert "flock 9" in SCRIPT
    assert ".ready" in SCRIPT


def test_no_credentials_are_used_on_nodes():
    assert "HF_TOKEN" not in SCRIPT
