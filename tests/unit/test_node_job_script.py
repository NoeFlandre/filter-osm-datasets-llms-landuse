from pathlib import Path

SCRIPT = (Path(__file__).parents[2] / "scripts" / "node_job.sh").read_text()


def test_cuda_home_is_exported_before_sglang_starts():
    """Regression (job 4165500): DeepEP asserts CUDA_HOME at SGLang import time."""
    assert SCRIPT.index("export CUDA_HOME") < SCRIPT.index("luf node run")
    assert "module load" in SCRIPT


def test_missing_site_env_is_built_on_node_local_disk_not_nfs():
    """Regression (Grenoble job 3122433): copying the venv onto slow NFS took the whole job."""
    assert 'venv="/tmp/$USER-venv-' in SCRIPT
    assert ".ready" in SCRIPT
    assert "flock" not in SCRIPT


def test_token_is_read_from_a_private_file_never_echoed():
    assert 'HF_TOKEN="$(<"$HOME/luf/hf_token")"' in SCRIPT
    assert "echo $HF_TOKEN" not in SCRIPT
    assert "set -x" not in SCRIPT


def test_scratch_is_node_local_and_removed():
    assert 'LUF_SCRATCH="/tmp/' in SCRIPT
    assert '"$LUF_SCRATCH"\' EXIT' in SCRIPT


def test_venv_bin_is_on_path_for_flashinfer_jit():
    """Regression (job 4165509): FlashInfer shells out to the venv's ninja."""
    assert SCRIPT.index('export PATH="$venv/bin:$PATH"') < SCRIPT.index("luf node run")


def test_job_runs_its_own_code_not_a_stale_installed_copy():
    """Regression (Lyon job 2070636): the shared site venv had an old package installed."""
    assert 'export PYTHONPATH="$CODE/src' in SCRIPT
    assert "--no-install-project" in SCRIPT
    assert '"$venv/bin/luf"' not in SCRIPT


def test_relative_code_path_points_python_to_deployed_source(tmp_path):
    """A site-style relative CODE path must still resolve after the script changes directory."""
    import hashlib
    import json
    import os
    import subprocess

    home = tmp_path / "home"
    code = home / "luf" / "code" / "abc123"
    code_src = code / "src"
    code_src.mkdir(parents=True)
    lock_contents = b"test lock\n"
    (code / "uv.lock").write_bytes(lock_contents)
    lock_sha = hashlib.sha256(lock_contents).hexdigest()[:12]

    venv = home / "luf" / "cache" / f"venv-{lock_sha}"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    (venv / ".ready").touch()
    stub_python = venv_bin / "python"
    stub_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        "print(json.dumps({'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH']}))\n"
    )
    stub_python.chmod(0o755)

    tools = home / "tools"
    tools.mkdir()
    sha256sum = tools / "sha256sum"
    sha256sum.write_text(f"#!/bin/sh\nprintf '%s  %s\\n' '{lock_sha}' \"$1\"\n")
    sha256sum.chmod(0o755)

    env = {
        "HOME": str(home),
        "OAR_JOB_ID": f"test-{os.getpid()}",
        "PATH": f"{tools}:{os.environ['PATH']}",
        "PYTHONPATH": "",
        "USER": f"luf-test-{os.getpid()}",
    }
    script = Path(__file__).parents[2] / "scripts" / "node_job.sh"
    result = subprocess.run(
        ["bash", str(script), "luf/code/abc123", "plan", "dataset", "revision"],
        cwd=home,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    observed = json.loads(result.stdout.splitlines()[-1])

    assert observed == {"cwd": str(code), "pythonpath": str(code_src)}
