from pathlib import Path

SCRIPT = (Path(__file__).parents[2] / "scripts" / "node_job.sh").read_text()


def test_cuda_home_is_exported_before_sglang_starts():
    """Regression (job 4165500): DeepEP asserts CUDA_HOME at SGLang import time."""
    assert SCRIPT.index("export CUDA_HOME") < SCRIPT.index('exec "$venv/bin/luf"')
    assert "module load" in SCRIPT


def test_env_is_built_once_under_a_lock():
    assert "flock 9" in SCRIPT
    assert ".ready" in SCRIPT


def test_token_is_read_from_a_private_file_never_echoed():
    assert 'HF_TOKEN="$(<"$HOME/luf/hf_token")"' in SCRIPT
    assert "echo $HF_TOKEN" not in SCRIPT
    assert "set -x" not in SCRIPT


def test_scratch_is_node_local_and_removed():
    assert 'LUF_SCRATCH="/tmp/' in SCRIPT
    assert '"$LUF_SCRATCH"\' EXIT' in SCRIPT


def test_venv_bin_is_on_path_for_flashinfer_jit():
    """Regression (job 4165509): FlashInfer shells out to the venv's ninja."""
    assert SCRIPT.index('export PATH="$venv/bin:$PATH"') < SCRIPT.index('exec "$venv/bin/luf"')
