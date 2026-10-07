"""Behavioural tests for scripts/node_job.sh.

The script runs under bash with a fake `uv` and a fake venv python on PATH. Both record
their argv and environment as JSON lines in a shared log, so each test asserts what the
script actually did: commands invoked, environment handed to the child, files left behind.
"""

import json
import re
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "node_job.sh"
# The script keeps its venv, uv cache and scratch on node-local /tmp by design.
NODE_LOCAL = Path("/tmp")  # noqa: S108

# Stub used as the venv's python. Records every call; `python -c <code> node ...` (the `luf`
# function) has argv, while the deadline helper `python -c <code>` has none and reads stdin.
PY_STUB = """#!__PYTHON__
import json, os, sys
def record(**kw):
    with open(os.environ["NODE_JOB_LOG"], "a") as f:
        f.write(json.dumps(kw) + "\\n")
KEPT_ENV = {
    "CUDA_HOME", "HF_TOKEN", "HF_HOME", "LD_LIBRARY_PATH", "LUF_JOB_DEADLINE_EPOCH",
    "LUF_SCRATCH", "PATH", "PYTHONPATH", "PYTHONUNBUFFERED",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
}
if os.path.basename(sys.argv[0]) == "luf":
    # The installed entry point of the venv; the script must never run it.
    record(role="installed-luf")
    sys.exit(99)
# Invoked as `<venv>/bin/python -c CODE [args]`: the deadline helper is the one whose CODE
# reads oarstat's JSON from stdin; the `luf` function passes `node ...` as args.
if sys.argv[1:2] == ["-c"] and "deadline_from_oarstat" in sys.argv[2]:
    stdin = sys.stdin.read()
    record(role="deadline-helper", stdin=stdin)
    print(os.environ.get("NODE_JOB_DEADLINE_STUB", ""))
    sys.exit(0)
record(
    role="child",
    argv=sys.argv[3:],
    executable=sys.argv[0],
    cwd=os.getcwd(),
    env={k: v for k, v in os.environ.items() if k in KEPT_ENV},
    which_ninja=__import__("shutil").which("ninja"),
    event="ready",
)
scratch = os.environ["LUF_SCRATCH"]
os.makedirs(scratch, exist_ok=True)
open(os.path.join(scratch, "chunk.parquet"), "w").close()
sys.exit(int(os.environ.get("NODE_JOB_CHILD_RC", "0")))
"""

# A fake `uv sync` builds the venv: a stub python, a stub ninja, and an installed `luf`
# entry point that must never run. It records argv and the UV_* variables it was given.
UV_STUB = """#!__PYTHON__
import json, os, sys
PY_STUB_TEXT = __PY_STUB__
venv = os.environ["UV_PROJECT_ENVIRONMENT"]
cache = os.environ["UV_CACHE_DIR"]
with open(os.environ["NODE_JOB_LOG"], "a") as f:
    f.write(json.dumps(dict(role="uv", argv=sys.argv[1:], venv=venv, cache=cache)) + "\\n")
os.makedirs(os.path.join(cache, "wheels"), exist_ok=True)
open(os.path.join(cache, "wheels", "marker"), "w").close()
bin_dir = os.path.join(venv, "bin")
os.makedirs(bin_dir, exist_ok=True)
def write(name, text):
    path = os.path.join(bin_dir, name)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)
write("python", PY_STUB_TEXT)
write("luf", PY_STUB_TEXT)
write("ninja", "#!/bin/sh\\nexit 0\\n")
"""

OARSTAT_STUB = """#!/bin/sh
printf '{"role": "oarstat", "argv": "%s"}\\n' "$*" >> "$NODE_JOB_LOG"
printf '{"deadline": 1900000000}\\n'
"""


@dataclass
class Sandbox:
    tmp: Path
    home: Path = field(init=False)
    code_rel: str = "luf/code/abc123"
    log: Path = field(init=False)
    tools: Path = field(init=False)
    job_id: str = field(default_factory=lambda: f"t{uuid.uuid4().hex[:10]}")
    user: str = field(default_factory=lambda: f"luf-test-{uuid.uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        self.home = self.tmp / "home"
        self.log = self.tmp / "calls.jsonl"
        self.tools = self.tmp / "tools"
        (self.home / "luf" / "code" / "abc123" / "src").mkdir(parents=True)
        (self.home / "luf" / "code" / "abc123" / "uv.lock").write_text("lock\n")
        self.tools.mkdir()
        self.log.touch()
        py_stub = PY_STUB.replace("__PYTHON__", sys.executable)
        uv_stub = UV_STUB.replace("__PYTHON__", sys.executable).replace(
            "__PY_STUB__", repr(py_stub)
        )
        self._write_exec("uv", uv_stub)
        self._write_exec("oarstat", OARSTAT_STUB)
        self.cuda_root = self.tmp / "cuda"
        (self.cuda_root / "bin").mkdir(parents=True)
        nvcc = self.cuda_root / "bin" / "nvcc"
        nvcc.write_text("#!/bin/sh\nexit 0\n")
        nvcc.chmod(0o755)

    def _write_exec(self, name: str, text: str) -> None:
        path = self.tools / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)

    def env(self, *, gpu: bool, extra: dict[str, str] | None = None) -> dict[str, str]:
        path = [str(self.tools)]
        if gpu:
            path.append(str(self.cuda_root / "bin"))
        path += ["/usr/bin", "/bin"]
        env = {
            "HOME": str(self.home),
            "PATH": ":".join(path),
            "OAR_JOB_ID": self.job_id,
            "USER": self.user,
            "NODE_JOB_LOG": str(self.log),
            "PYTHONPATH": "",
        }
        env.update(extra or {})
        return env

    def run(
        self, *args: str, gpu: bool = True, extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), "luf/code/abc123", *args],
            cwd=self.home,
            env=self.env(gpu=gpu, extra=extra),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def records(self, role: str) -> list[dict]:
        lines = self.log.read_text().splitlines()
        return [r for r in map(json.loads, lines) if r.get("role") == role]

    def child(self) -> dict:
        ready = [r for r in self.records("child") if r.get("event") == "ready"]
        assert len(ready) == 1, f"expected one child start, got {ready}"
        return ready[0]

    def venv_dir(self) -> Path:
        return NODE_LOCAL / f"{self.user}-venv-{self.job_id}"


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    box = Sandbox(tmp_path)
    yield box
    # The script removes its own /tmp venv and scratch on exit; clean up anything left by a
    # failing test so /tmp does not fill up.
    for leftover in (box.venv_dir(), NODE_LOCAL / f"{box.user}-luf-{box.job_id}"):
        subprocess.run(["rm", "-rf", str(leftover)], check=False)


MODES = [
    # (script args after the code dir, uv extras, child argv)
    pytest.param(
        ["7"],
        ["--extra", "gpu"],
        ["node", "run", "--assignment", "7"],
        id="run",
    ),
    pytest.param(
        ["calibrate", "chunk-1"],
        ["--extra", "gpu"],
        ["node", "calibrate", "--chunk", "chunk-1"],
        id="calibrate",
    ),
    pytest.param(
        ["plan", "ds", "rev"],
        ["--extra", "tokenize"],
        ["node", "plan", "--dataset", "ds", "--revision", "rev"],
        id="plan",
    ),
    pytest.param(
        ["card", "ds", "rev"],
        ["--extra", "tokenize", "--extra", "map"],
        ["node", "publish", "--card-only", "--dataset", "ds", "--revision", "rev"],
        id="card",
    ),
    pytest.param(
        ["publish", "ds", "rev"],
        ["--extra", "tokenize", "--extra", "map"],
        ["node", "publish", "--dataset", "ds", "--revision", "rev"],
        id="publish",
    ),
    pytest.param(
        ["repair", "ds", "rev"],
        ["--extra", "tokenize", "--extra", "map"],
        ["node", "repair", "--dataset", "ds"],
        id="repair",
    ),
    pytest.param(
        ["replan", "ds", "rev"],
        ["--extra", "tokenize", "--extra", "map"],
        ["node", "replan", "--dataset", "ds", "--revision", "rev"],
        id="replan",
    ),
]


@pytest.mark.parametrize(("args", "extras", "child_argv"), MODES)
def test_each_mode_installs_only_its_extras_and_runs_its_subcommand(
    sandbox: Sandbox, args: list[str], extras: list[str], child_argv: list[str]
):
    result = sandbox.run(*args)

    assert result.returncode == 0, result.stderr
    (uv,) = sandbox.records("uv")
    assert uv["argv"] == [
        "sync",
        "--frozen",
        "--no-dev",
        "--no-install-project",
        *extras,
        "--python",
        "3.12",
    ]
    assert sandbox.child()["argv"] == child_argv


def test_cpu_job_never_gets_the_gpu_extra_even_with_a_cuda_toolkit_present(sandbox: Sandbox):
    """Regression (Lyon planning jobs): a CPU job on a GPU node must not pull the GPU stack."""
    sandbox.run("plan", "ds", "rev", gpu=True)

    (uv,) = sandbox.records("uv")
    assert "gpu" not in " ".join(uv["argv"])


def test_gpu_job_without_nvcc_fails_before_building_or_running(sandbox: Sandbox):
    result = sandbox.run("7", gpu=False)

    assert result.returncode == 6
    assert "nvcc" in result.stderr
    assert sandbox.records("uv") == []
    assert sandbox.records("child") == []


def test_cuda_home_and_toolkit_are_set_for_the_gpu_child(sandbox: Sandbox):
    """Regression (job 4165500): DeepEP needs CUDA_HOME when SGLang imports."""
    result = sandbox.run("7")

    assert result.returncode == 0, result.stderr
    env = sandbox.child()["env"]
    assert env["CUDA_HOME"] == str(sandbox.cuda_root)
    assert env["LD_LIBRARY_PATH"].startswith(f"{sandbox.cuda_root}/lib64:")
    assert env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_venv_bin_is_first_on_path_so_flashinfer_finds_its_ninja(sandbox: Sandbox):
    """Regression (job 4165509): FlashInfer shells out to the venv's ninja."""
    sandbox.run("plan", "ds", "rev")

    child = sandbox.child()
    assert child["env"]["PATH"].split(":")[0] == str(sandbox.venv_dir() / "bin")
    assert child["which_ninja"] == str(sandbox.venv_dir() / "bin" / "ninja")


def test_job_runs_its_own_code_not_an_installed_copy(sandbox: Sandbox):
    """Regression (Lyon job 2070636): a shared venv ran a stale installed package."""
    result = sandbox.run("plan", "ds", "rev")

    assert result.returncode == 0, result.stderr
    code = sandbox.home / "luf" / "code" / "abc123"
    child = sandbox.child()
    assert child["env"]["PYTHONPATH"] == str(code / "src")
    assert child["cwd"] == str(code)
    assert sandbox.records("installed-luf") == []


def test_env_is_built_fresh_on_node_local_disk_even_when_a_cached_env_exists(sandbox: Sandbox):
    """Regression (Grenoble job 3122433, quota): never reuse a cached venv from $HOME."""
    cached = sandbox.home / "luf" / "cache" / "venv-deadbeef"
    (cached / "bin").mkdir(parents=True)
    (cached / ".ready").touch()

    result = sandbox.run("plan", "ds", "rev")

    assert result.returncode == 0, result.stderr
    (uv,) = sandbox.records("uv")
    assert uv["venv"] == str(sandbox.venv_dir())
    assert sandbox.child()["executable"] == str(sandbox.venv_dir() / "bin" / "python")
    assert not str(sandbox.venv_dir()).startswith(str(sandbox.home))


def test_private_venv_and_scratch_are_removed_on_exit(sandbox: Sandbox):
    result = sandbox.run("plan", "ds", "rev")

    assert result.returncode == 0, result.stderr
    scratch = Path(sandbox.child()["env"]["LUF_SCRATCH"])
    assert scratch == NODE_LOCAL / f"{sandbox.user}-luf-{sandbox.job_id}"
    # The child wrote a file into scratch; the script's exit trap must have removed it.
    assert not (scratch / "chunk.parquet").exists()
    assert not sandbox.venv_dir().exists()
    assert not Path(sandbox.records("uv")[0]["cache"]).exists()


def test_token_reaches_the_child_and_is_never_printed(sandbox: Sandbox):
    token = f"hf_test_{uuid.uuid4().hex}"
    token_file = sandbox.home / "luf" / "hf_token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)

    result = sandbox.run("plan", "ds", "rev")

    assert result.returncode == 0, result.stderr
    assert sandbox.child()["env"]["HF_TOKEN"] == token
    assert token not in result.stdout
    assert token not in result.stderr


def test_no_token_file_means_no_token_in_the_child(sandbox: Sandbox):
    result = sandbox.run("plan", "ds", "rev")

    assert result.returncode == 0, result.stderr
    assert "HF_TOKEN" not in sandbox.child()["env"]


def test_publish_exports_its_deadline_from_walltime_before_running(sandbox: Sandbox):
    """A publish job stops starting files 6 minutes before its end, so it must know it."""
    before = int(time.time())
    result = sandbox.run("publish", "ds", "rev", extra={"OAR_JOB_WALLTIME_SECONDS": "3600"})
    after = int(time.time())

    assert result.returncode == 0, result.stderr
    deadline = int(sandbox.child()["env"]["LUF_JOB_DEADLINE_EPOCH"])
    assert before + 3600 <= deadline <= after + 3600
    assert sandbox.records("oarstat") == []


def test_publish_deadline_comes_from_oarstat_when_walltime_is_unknown(sandbox: Sandbox):
    result = sandbox.run("publish", "ds", "rev", extra={"NODE_JOB_DEADLINE_STUB": "1900000001"})

    assert result.returncode == 0, result.stderr
    assert sandbox.child()["env"]["LUF_JOB_DEADLINE_EPOCH"] == "1900000001"
    (oarstat,) = sandbox.records("oarstat")
    assert oarstat["argv"] == f"-j {sandbox.job_id} -J"
    (helper,) = sandbox.records("deadline-helper")
    assert json.loads(helper["stdin"]) == {"deadline": 1900000000}


def test_no_deadline_is_exported_when_oarstat_gives_none(sandbox: Sandbox):
    result = sandbox.run("publish", "ds", "rev", extra={"NODE_JOB_DEADLINE_STUB": ""})

    assert result.returncode == 0, result.stderr
    assert "LUF_JOB_DEADLINE_EPOCH" not in sandbox.child()["env"]


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


@pytest.mark.parametrize(("args", "extras", "child_argv"), MODES)
def test_each_mode_returns_the_child_exit_code(
    sandbox: Sandbox, args: list[str], extras: list[str], child_argv: list[str]
):
    result = sandbox.run(*args, extra={"NODE_JOB_CHILD_RC": "7"})

    assert result.returncode == 7, result.stderr
    assert sandbox.child()["argv"] == child_argv


def test_checkpoint_signal_is_forwarded_to_the_child_and_its_exit_code_returned(tmp_path):
    """Regression: OAR's SIGUSR2 hit bash only (exit 12); python never flushed gracefully."""
    script = SCRIPT.read_text()
    block = re.search(r"# >>> run_forwarding\n(.*?)# <<< run_forwarding", script, re.S)
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


def test_locked_numpy_stays_below_the_cpu_baseline_bump():
    """Regression (Lyon planning jobs 2070945/2070978): numpy>=2.4 died with SIGILL on
    older Grid'5000 CPU nodes. Any lock refresh must stay below 2.4, not equal one version."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    numpy_versions = [p["version"] for p in lock["package"] if p["name"] == "numpy"]
    assert len(numpy_versions) == 1
    assert Version(numpy_versions[0]) < Version("2.4")
