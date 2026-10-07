from pathlib import Path

SCRIPT = (Path(__file__).parents[2] / "scripts" / "node_job.sh").read_text()


def test_cuda_home_is_exported_before_sglang_starts():
    """Regression (job 4165500): DeepEP asserts CUDA_HOME at SGLang import time."""
    assert SCRIPT.index("export CUDA_HOME") < SCRIPT.index("luf node run")
    assert "module load" in SCRIPT


def test_env_is_always_built_on_node_local_disk_not_nfs():
    """Regressions: slow NFS copies (Grenoble job 3122433) and site home quotas exceeded."""
    assert 'venv="/tmp/$USER-venv-' in SCRIPT
    assert ".ready" not in SCRIPT
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
    """Regression (#48): a site-style relative CODE path must still resolve after cd."""
    import json
    import os
    import subprocess

    home = tmp_path / "home"
    code = home / "luf" / "code" / "abc123"
    (code / "src").mkdir(parents=True)
    (code / "uv.lock").write_text("lock\n")
    tools = home / "tools"
    tools.mkdir()
    # A fake `uv sync` that creates the venv with a stub python reporting cwd + PYTHONPATH.
    fake_uv = tools / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"\n'
        "cat > \"$UV_PROJECT_ENVIRONMENT/bin/python\" <<'PY'\n"
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        "print(json.dumps({'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH']}))\n"
        "PY\n"
        'chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"\n'
    )
    fake_uv.chmod(0o755)
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
    assert observed == {"cwd": str(code), "pythonpath": str(code / "src")}


def test_cpu_jobs_do_not_install_the_gpu_stack():
    """Regression (Lyon jobs 2070709/2070898): planning installed sglang, torch and CUDA
    into node-local /tmp, filled the disk and crashed with "No space left on device"."""
    assert '"${EXTRAS[@]}"' in SCRIPT
    assert "run | calibrate) EXTRAS=(--extra gpu)" in SCRIPT
    assert "*) EXTRAS=(--extra tokenize)" in SCRIPT
    assert SCRIPT.count("--extra gpu") == 1  # only the GPU modes


def test_numpy_stays_below_the_cpu_baseline_bump():
    """Regression (Lyon planning jobs 2070945/2070978): numpy>=2.4 died with SIGILL on
    older Grid'5000 CPU nodes; the working GPU stack ran numpy 2.3.5."""
    pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    lock = (Path(__file__).parents[2] / "uv.lock").read_text()
    assert '"numpy>=2.0,<2.4"' in pyproject
    assert 'name = "numpy"\nversion = "2.3.5"' in lock


def test_repair_mode_needs_no_gpu_stack():
    """`repair` only rewrites parquet tables: it must take the light CPU environment."""
    assert '"$MODE" == "repair"' in SCRIPT
    assert "publish | repair | replan) EXTRAS=(--extra tokenize --extra map)" in SCRIPT


def test_replan_mode_builds_the_tokeniser_and_h3_environment():
    """`replan` tokenises texts and bins them into H3 cells, never loads SGLang."""
    assert '"${1:-}" == "replan"' in SCRIPT
    assert "publish | repair | replan) EXTRAS=(--extra tokenize --extra map)" in SCRIPT


def test_publish_jobs_export_their_deadline_before_running():
    """A publish job learns its end time (oarstat) so it can stop starting files 6 minutes early."""
    assert "LUF_JOB_DEADLINE_EPOCH" in SCRIPT
    assert SCRIPT.index("LUF_JOB_DEADLINE_EPOCH") < SCRIPT.index('luf node "$MODE"')
    assert "oarstat -j" in SCRIPT


def test_checkpoint_signal_is_forwarded_to_the_child_and_its_exit_code_returned(tmp_path):
    """Regression: OAR's SIGUSR2 hit bash only (exit 12); python never flushed gracefully."""
    import re
    import signal
    import subprocess
    import sys
    import time

    block = re.search(r"# >>> run_forwarding\n(.*?)# <<< run_forwarding", SCRIPT, re.S)
    assert block, "run_forwarding block missing"
    luf = re.search(r"^luf\(\) \{.*\}$", SCRIPT, re.M)
    assert luf, "luf definition missing"
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    child = tmp_path / "child.py"
    ready = tmp_path / "ready"
    child.write_text(
        "import signal, sys, time, pathlib\n"
        "def h(n, f):\n"
        "    time.sleep(0.5)\n"
        "    print('child got usr2', flush=True)\n"
        "    sys.exit(7)\n"
        "signal.signal(signal.SIGUSR2, h)\n"
        f"pathlib.Path({str(ready)!r}).write_text('x')\n"
        "time.sleep(30)\n"
    )
    # The fake venv python ignores `luf`'s arguments and runs the child instead.
    fake_python.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{child}'\n")
    fake_python.chmod(0o755)
    harness = tmp_path / "h.sh"
    harness.write_text(
        "set -euo pipefail\n"
        f"venv='{tmp_path / 'venv'}'\n"
        + luf.group(0)
        + "\n"
        + block.group(1)
        + "rc=0\nrun_forwarding luf node run || rc=$?\n"
        'echo "wrapper rc=$rc"\nexit "$rc"\n'
    )
    proc = subprocess.Popen(
        ["bash", str(harness)], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    deadline = time.time() + 10
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    proc.send_signal(signal.SIGUSR2)  # only the wrapper, like OAR
    out, _ = proc.communicate(timeout=15)
    assert "child got usr2" in out
    assert "wrapper rc=7" in out
    assert proc.returncode == 7


def test_every_mode_runs_its_child_through_the_forwarding_wrapper():
    assert "run_forwarding luf node run --assignment" in SCRIPT
    for line in SCRIPT.splitlines():
        if line.lstrip().startswith("luf node"):
            raise AssertionError(f"unwrapped child: {line}")
