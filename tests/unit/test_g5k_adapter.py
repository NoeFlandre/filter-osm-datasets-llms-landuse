import json
import subprocess

import pytest

from landuse_filter.adapters import g5k


class Run:
    """Fake subprocess.run: returns canned (returncode, stdout, stderr) per call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        code, out, err = self.responses.pop(0)
        if isinstance(code, Exception):
            raise code
        return subprocess.CompletedProcess(args, code, out.encode(), err.encode())


def fake(monkeypatch, *responses):
    run = Run(*responses)
    monkeypatch.setattr(g5k.subprocess, "run", run)
    return run


def test_ssh_returns_stdout_and_raises_on_failure_or_timeout(monkeypatch):
    fake(monkeypatch, (0, "ok\n", ""))
    assert g5k.ssh("nancy", "true") == "ok\n"
    fake(monkeypatch, (255, "", "Connection refused"))
    with pytest.raises(g5k.RemoteError, match="Connection refused"):
        g5k.ssh("nancy", "true")
    fake(monkeypatch, (subprocess.TimeoutExpired("ssh", 1), "", ""))
    with pytest.raises(g5k.RemoteError, match="timed out"):
        g5k.ssh("nancy", "sleep 9")


def test_our_jobs_only_returns_luf_jobs(monkeypatch):
    payload = {
        "1": {"name": "luf-abc", "state": "Running", "queue": "p3", "scheduled_start": 5},
        "2": {"name": "lrb-bench", "state": "Running", "queue": "default"},
        "3": {"name": None, "state": "Waiting", "queue": "default"},
    }
    fake(monkeypatch, (0, json.dumps(payload), ""))
    jobs = g5k.our_jobs("lille")
    assert [(j.job_id, j.name, j.scheduled_start) for j in jobs] == [("1", "luf-abc", 5)]


def test_submit_parses_the_job_id_and_quotes_arguments(monkeypatch):
    run = fake(monkeypatch, (0, "[ADMISSION RULE] ...\nOAR_JOB_ID=4165999\n", ""))
    assert g5k.submit("rennes", ["-p", "cluster='x'", "cmd with space"]) == "4165999"
    assert "'cmd with space'" in run.calls[0][-1]
    fake(monkeypatch, (0, "nothing useful", ""))
    with pytest.raises(g5k.RemoteError, match="no job id"):
        g5k.submit("rennes", ["x"])


def test_scheduled_start_and_cancel(monkeypatch):
    fake(monkeypatch, (0, json.dumps({"7": {"state": "Waiting", "scheduled_start": 42}}), ""))
    assert g5k.scheduled_start("lyon", "7") == ("Waiting", 42)
    run = fake(monkeypatch, (0, "", ""))
    g5k.cancel("lyon", "7")
    assert run.calls[0][-1] == "oardel 7"


def test_project_listing_parses_paths_and_ages(monkeypatch):
    fake(monkeypatch, (0, "luf/code/abc 1.50\nluf/logs/1.out 8.00\n\n", ""))
    assert g5k.project_listing("grenoble") == [("luf/code/abc", 1.5), ("luf/logs/1.out", 8.0)]


@pytest.mark.parametrize("bad", ["../etc", "geoparser-venv", "luf/../x"])
def test_remove_refuses_anything_outside_the_project_tree(monkeypatch, bad):
    run = fake(monkeypatch)
    with pytest.raises(g5k.RemoteError, match="outside"):
        g5k.remove("lille", ["luf/code/old", bad])
    assert run.calls == []


def test_remove_deletes_in_batches(monkeypatch):
    run = fake(monkeypatch, (0, "", ""), (0, "", ""))
    g5k.remove("lille", [f"luf/logs/{i}.out" for i in range(250)])
    assert len(run.calls) == 2


def test_rsync_tolerates_vanished_files_only(monkeypatch):
    fake(monkeypatch, (24, "", ""))
    g5k.rsync("a", "b")
    fake(monkeypatch, (23, "", "some files not transferred"))
    with pytest.raises(g5k.RemoteError, match="rsync"):
        g5k.rsync("a", "b")


def test_concurrent_deploys_of_the_same_commit_all_succeed(tmp_path):
    """Regression (Rennes): parallel controllers raced rm -rf / tar on one target dir."""
    import io
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for i in range(50):
            data = b"x" * 1000
            info = tarfile.TarInfo(f"tests/unit/f{i}.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()
    command = g5k.deploy_command("luf/code/abc")

    def deploy(_):
        return subprocess.run(
            ["bash", "-c", command], input=archive, cwd=tmp_path, capture_output=True, check=False
        )

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(deploy, range(8)))
    assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
    target = tmp_path / "luf" / "code" / "abc"
    assert (target / ".complete").exists()
    assert len(list((target / "tests" / "unit").iterdir())) == 50
    assert not list((tmp_path / "luf" / "code").glob(".deploy.*"))


def test_our_jobs_reads_the_submission_time(monkeypatch):
    payload = {"1": {"name": "luf-abc", "state": "Waiting", "submissionTime": 1700000000}}
    fake(monkeypatch, (0, json.dumps(payload), ""))
    assert g5k.our_jobs("lille")[0].submitted == 1700000000
