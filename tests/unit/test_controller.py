from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import controller as ctl_mod
from landuse_filter.application.assignment import CycleReport
from landuse_filter.application.controller import Controller, Settings

NOW = datetime(2026, 9, 28, 22, 0, tzinfo=ZoneInfo("Europe/Paris"))
GRES = {
    "site": "nancy",
    "cluster": "gres",
    "gpu": "L40S",
    "memory_mib": 46068,
    "compute_capability": [8, 9],
    "gpus_per_node": 2,
    "nodes": 2,
    "queues": ["abaca", "production"],
    "exotic": False,
}
FREE = {
    "gres-1.nancy.grid5000.fr": {
        "hard": "alive",
        "free_slots": 48,
        "freeable_slots": 0,
        "busy_slots": 0,
    }
}


class FakeG5K:
    def __init__(self, refuse=None):
        self.jobs: dict[str, list[g5k.Job]] = {"nancy": []}
        self.submitted: list[list[str]] = []
        self.refuse = refuse
        self.policy_checks = 0
        self.late = False
        self.cancelled = []

    def our_jobs(self, site):
        return list(self.jobs.get(site, []))

    def submit(self, site, args):
        if self.refuse:
            raise g5k.RemoteError(self.refuse)
        self.submitted.append(args)
        job_id = str(100 + len(self.submitted))
        self.jobs[site].append(
            g5k.Job(site, job_id, args[args.index("-n") + 1], "Waiting", "abaca")
        )
        return job_id

    def policy_check(self, site):
        self.policy_checks += 1

    def scheduled_start(self, site, job_id):
        return ("Running", None) if not self.late else ("Waiting", 4_000_000_000)

    def cancel(self, site, job_id):
        self.jobs[site] = [j for j in self.jobs[site] if j.job_id != job_id]
        self.cancelled.append(job_id)


@pytest.fixture
def world(tmp_path, monkeypatch):
    fake = FakeG5K()
    monkeypatch.setattr(ctl_mod, "POST_SUBMIT_WAIT", 0.0)
    for name in ("our_jobs", "submit", "policy_check", "scheduled_start", "cancel"):
        monkeypatch.setattr(g5k, name, getattr(fake, name))
    monkeypatch.setattr(g5k, "site_status", lambda site: {"nodes": FREE})
    monkeypatch.setattr(g5k, "rsync", lambda *a, **k: None)
    monkeypatch.setattr(g5k, "ssh", lambda *a, **k: "")
    monkeypatch.setattr(g5k, "deploy_code", lambda site, commit, archive: "luf/code/x")
    monkeypatch.setattr(ctl_mod, "git_archive", lambda ref: b"")
    monkeypatch.setattr(ctl_mod, "commit", lambda: "abc")
    store = WorkStore(tmp_path)
    store.write_json("inventory.json", [GRES])
    c = Controller(
        store,
        Settings(datasets=["benchmark"], sites=["nancy"], gpu_models=["l40s"]),
        log=lambda m: None,
        remote=DirRemote(tmp_path / "bucket"),
    )
    for i in range(3):
        cid = f"c{i}"
        store.write_chunk(
            cid,
            pa.table({"text_sha256": [f"{cid}s"], "text": ["t"], "input_ids": [[1]]}, schema=CHUNK),
        )
        store.append_jsonl(
            f"plans/benchmark/{c.fp}/chunks.jsonl",
            [{"chunk_id": cid, "order": [0, 0], "size": 3000}],
        )
    return c, fake


def test_submits_where_free_with_policy_checks(world):
    c, fake = world
    report = c.cycle(NOW)
    assert len(report.submitted) == 2  # two free GPUs -> two one-GPU jobs
    assert fake.policy_checks == 4  # before and after each submission
    assert fake.submitted[0][:2] == ["-q", "abaca"]
    chunks = [set(a.chunks) for a in c.live()]
    assert not chunks[0] & chunks[1]  # disjoint assignments


def test_second_cycle_does_not_duplicate(world):
    c, fake = world
    c.cycle(NOW)
    c.cycle(NOW)
    assert all(len(a.chunks) for a in c.live())
    taken = [ch for a in c.live() for ch in a.chunks]
    assert len(taken) == len(set(taken))


def test_ended_jobs_release_their_chunks(world):
    c, fake = world
    c.cycle(NOW)
    fake.jobs["nancy"] = []  # jobs finished without results
    c.reconcile()
    assert c.live() == []
    assert {a.state for a in c.ledger()} == {"ended"}


def test_crash_after_oarsub_is_adopted(world):
    c, fake = world
    c.cycle(NOW)
    a = c.live()[0]
    a.state, a.job_id = "submitting", None  # as if we died before recording the id
    c.save(a)
    c.reconcile()
    adopted = c.store.read_json(f"assignments/{a.id}.json")
    assert adopted["state"] == "submitted"
    assert adopted["job_id"]


def test_besteffort_only_cluster_is_remembered(world, monkeypatch):
    c, fake = world
    fake.refuse = "# You can only access the required resources in besteffort."
    report = c.cycle(NOW)
    assert report.submitted == []
    assert c.store.exists("access/nancy_gres.json")
    fake.refuse = None
    assert c.cycle(NOW).submitted == []  # not retried


def test_paused_submits_nothing(world):
    c, fake = world
    c.settings.paused = True
    assert c.cycle(NOW).submitted == []


def test_late_start_is_cancelled_and_cluster_backed_off(world):
    c, fake = world
    fake.late = True
    assert c.cycle(NOW).submitted == []
    assert fake.cancelled
    assert c.live() == []
    fake.late = False
    assert c.cycle(NOW).submitted == []  # still backed off


def test_waiting_reservation_blocks_node():
    from landuse_filter.domain.capacity import Cluster, free_gpus

    gres = Cluster("nancy", "gres", "L40S", 46068, (8, 9), 2, 1, ("abaca",), exotic=False)
    node = {
        "gres-1.x": {
            "hard": "alive",
            "free_slots": 48,
            "busy_slots": 0,
            "reservations": [{"state": "waiting", "scheduled_at": 1000}],
        }
    }
    assert free_gpus(gres, node, besteffort_counts=False, now=0, walltime_s=3600) == 0
    assert free_gpus(gres, node, besteffort_counts=False, now=0, walltime_s=500) == 2


def test_waiting_job_whose_start_drifts_is_cancelled(world):
    c, fake = world
    c.cycle(NOW)
    job = fake.jobs["nancy"][0]
    fake.jobs["nancy"][0] = g5k.Job(
        job.site, job.job_id, job.name, "Waiting", "abaca", int(NOW.timestamp()) + 7200
    )
    c.now = NOW
    c.reconcile()
    assert job.job_id in fake.cancelled
    assert c.store.exists("backoff/nancy_gres.json")
    assert job.name not in {a.name for a in c.live()}


def test_candidate_namespace_and_window(world):
    c, fake = world
    c.settings.namespace, c.settings.window = "w128", 128
    c.cycle(NOW)
    a = c.live()[0]
    assert a.fp == f"{c.fp}-w128"
    assert a.window == 128
    assert a.engine_kwargs["max_running_requests"] == 128
    assert a.provenance["config_fingerprint"] == c.fp  # same generation identity


def test_regular_access_counts_besteffort_held_gpus(world, monkeypatch):
    c, fake = world
    held = {
        "gres-1.nancy.grid5000.fr": {
            "hard": "alive",
            "free_slots": 0,
            "freeable_slots": 48,
            "busy_slots": 0,
        }
    }
    monkeypatch.setattr(g5k, "site_status", lambda site: {"nodes": held})
    assert len(c.cycle(NOW).submitted) == 2  # regular jobs preempt besteffort ones
    assert "besteffort" not in fake.submitted[0]


def test_besteffort_only_cluster_gets_besteffort_jobs_on_free_gpus(world):
    c, fake = world
    c.store.write_json("access/nancy_gres.json", {"besteffort_only": True})
    c.settings.besteffort = True
    c.cycle(NOW)
    assert fake.submitted[0][:2] == ["-t", "besteffort"]


def test_parallel_controllers_do_not_release_each_others_work(world):
    c, fake = world
    c.cycle(NOW)
    theirs = c.live()
    other = Controller(
        c.store,
        Settings(datasets=["benchmark"], sites=["lyon"], namespace="gpu-x"),
        log=lambda m: None,
        remote=DirRemote(c.store.root / "bucket"),
    )
    other.reconcile()  # sees none of nancy's jobs
    assert [a.id for a in c.live()] == [a.id for a in theirs]  # untouched
    assert other.live() == []


def test_loop_survives_a_failing_cycle(world):
    """Regression: one git error (vanished cwd) killed every controller."""
    from landuse_filter.application.controller import run_loop

    c, fake = world
    calls = {"n": 0}
    real = c.cycle

    def flaky(now=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("git rev-parse failed")
        return CycleReport(0, 0)

    c.cycle = flaky
    logs, sleeps, out = [], [], []
    c.log = logs.append
    run_loop(c, interval=5, sleep=sleeps.append, emit=out.append)
    assert calls["n"] == 2
    assert sleeps == [5]
    assert logs[0] == "cycle failed; retrying in 5s (RuntimeError: git rev-parse failed)"
    assert any(line.startswith("  | ") and "flaky" in line for line in logs[1:])  # traceback
    assert out
    assert '"pending_chunks": 0' in out[0]
    c.cycle = real


def test_loop_once_returns_after_one_failed_cycle(world):
    from landuse_filter.application.controller import run_loop

    c, fake = world
    c.cycle = lambda now=None: (_ for _ in ()).throw(OSError("boom"))
    c.log = lambda m: None
    run_loop(
        c, interval=5, once=True, sleep=lambda s: (_ for _ in ()).throw(AssertionError("no sleep"))
    )


def test_shared_site_cache_queries_each_site_once_per_cycle(world, monkeypatch):
    from landuse_filter.application.site_cache import SiteCache

    c, fake = world
    calls = {"jobs": 0, "status": 0}
    orig_jobs = g5k.our_jobs

    def counting_jobs(site):
        calls["jobs"] += 1
        return orig_jobs(site)

    monkeypatch.setattr(g5k, "our_jobs", counting_jobs)
    monkeypatch.setattr(
        g5k,
        "site_status",
        lambda site: calls.__setitem__("status", calls["status"] + 1) or {"nodes": FREE},
    )
    cache = SiteCache()
    a = Controller(
        c.store,
        Settings(datasets=["benchmark"], sites=["nancy"], gpu_models=["l40s"], namespace="gpu-a"),
        log=lambda m: None,
        sites=cache,
        remote=DirRemote(c.store.root / "bucket"),
    )
    b = Controller(
        c.store,
        Settings(datasets=["benchmark"], sites=["nancy"], gpu_models=["l40s"], namespace="gpu-b"),
        log=lambda m: None,
        sites=cache,
        remote=DirRemote(c.store.root / "bucket"),
    )
    cache.reset()
    a.cycle(NOW)
    b.cycle(NOW)
    assert calls["status"] == 1  # one status call per site per cycle
    assert calls["jobs"] <= 3  # initial + one refresh per submission, not one per controller
    assert {x.fp for x in a.live()} == {f"{a.fp}-gpu-a"}
    assert {x.fp for x in b.live()} == {f"{b.fp}-gpu-b"}


def test_run_many_drops_finished_namespaces(world):
    from landuse_filter.application.controller import run_many
    from landuse_filter.application.site_cache import SiteCache

    c, fake = world
    done = Controller(
        c.store,
        Settings(datasets=["none"], sites=["nancy"], namespace="gpu-z"),
        log=lambda m: None,
        remote=DirRemote(c.store.root / "bucket"),
    )
    out, sleeps = [], []
    run_many([done], SiteCache(), interval=1, sleep=sleeps.append, emit=out.append)
    assert len(out) == 1
    assert sleeps == []


BUSY = {
    "gres-1.nancy.grid5000.fr": {
        "hard": "alive",
        "free_slots": 0,
        "freeable_slots": 0,
        "busy_slots": 48,
    }
}


def test_no_queueing_by_default_when_nothing_is_free(world, monkeypatch):
    c, fake = world
    monkeypatch.setattr(g5k, "site_status", lambda site: {"nodes": BUSY})
    assert c.cycle(NOW).submitted == []


def test_bounded_queue_submits_one_waiting_job_with_a_longer_tolerance(world, monkeypatch):
    c, fake = world
    monkeypatch.setattr(g5k, "site_status", lambda site: {"nodes": BUSY})
    c.settings.max_queued_per_site = 1
    fake.late = False
    submitted = c.cycle(NOW).submitted
    assert len(submitted) == 1
    a = c.live()[0]
    assert a.late_after_s == 7200
    # the job waits, predicted to start in 1 h: within its tolerance, so it is kept
    job = fake.jobs["nancy"][0]
    fake.jobs["nancy"][0] = g5k.Job(
        job.site, job.job_id, job.name, "Waiting", "abaca", int(NOW.timestamp()) + 3600
    )
    c.now = NOW
    c.reconcile()
    assert job.job_id not in fake.cancelled
    # the site's queue is full: no second waiting job
    assert c.cycle(NOW).submitted == []


def test_free_slots_outrank_queued_ones():
    from landuse_filter.domain.scheduling import Slot, rank_slots

    free = Slot("a", "c", "g", 1, 1, timedelta(0), timedelta(hours=1), None, 1.0)
    queued = Slot(
        "b", "c", "g", 1, 1, timedelta(hours=1), timedelta(hours=1), None, 1.0, queued=True
    )
    assert rank_slots([queued, free], timedelta(minutes=8))[0] is free


def test_run_command_passes_the_night_walltime_to_the_controller(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = {}

    def fake_controller(work, settings):
        seen["settings"] = settings
        return object()

    monkeypatch.setattr(cli, "_controller", fake_controller)
    monkeypatch.setattr(ctl_mod, "run_loop", lambda *a, **k: None)
    result = CliRunner().invoke(
        cli.g5k_app,
        ["run", "--datasets", "d", "--work", str(tmp_path), "--night-walltime-minutes", "30"],
    )
    assert result.exit_code == 0, result.output
    assert seen["settings"].night_walltime == timedelta(minutes=30)


def test_run_admission_passes_both_walltimes(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = {}
    monkeypatch.setattr(ctl_mod, "run_many", lambda ctls, *a, **k: seen.update(ctls=ctls))
    result = CliRunner().invoke(
        cli.g5k_app,
        [
            "run-admission",
            "--gpus",
            "l4",
            "--work",
            str(tmp_path),
            "--walltime-minutes",
            "20",
            "--night-walltime-minutes",
            "30",
        ],
    )
    assert result.exit_code == 0, result.output
    settings = seen["ctls"][0].settings
    assert (settings.walltime, settings.night_walltime) == (
        timedelta(minutes=20),
        timedelta(minutes=30),
    )


def test_cpu_jobs_avoid_the_sagittaire_cluster(monkeypatch, tmp_path):
    """Regression (Lyon jobs 2070898..2071001): every run on sagittaire-5 (ancient CPU)
    died with 'Illegal instruction' or a full disk; the same code ran fine on taurus."""
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    submitted = []
    monkeypatch.setattr(g5k, "deploy_code", lambda *a, **k: "luf/code/x")
    monkeypatch.setattr(g5k, "ssh", lambda *a, **k: "")
    monkeypatch.setattr(g5k, "policy_check", lambda site: None)
    monkeypatch.setattr(g5k, "submit", lambda site, args: submitted.append(args) or "1")
    monkeypatch.setattr(ctl_mod, "commit", lambda: "abc")
    monkeypatch.setattr(ctl_mod, "git_archive", lambda ref: b"")
    result = CliRunner().invoke(
        cli.g5k_app,
        ["cpu-job", "plan", "--site", "lyon", "--dataset", "d", "--revision", "r"],
    )
    assert result.exit_code == 0, result.output
    prop = submitted[0][submitted[0].index("-p") + 1]
    assert prop == "gpu_count = 0 AND cluster != 'sagittaire'"


def test_run_admission_passes_the_queue_depth(monkeypatch, tmp_path):
    """Admission used the default depth of 1 waiting job per site, starving busy clusters."""
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = {}
    monkeypatch.setattr(ctl_mod, "run_many", lambda ctls, *a, **k: seen.update(ctls=ctls))
    result = CliRunner().invoke(
        cli.g5k_app,
        ["run-admission", "--gpus", "l4", "--work", str(tmp_path), "--max-queued-per-site", "3"],
    )
    assert result.exit_code == 0, result.output
    assert seen["ctls"][0].settings.max_queued_per_site == 3


def test_git_archive_is_read_once_per_commit_and_retried_on_a_faulting_drive(monkeypatch):
    """Regression: `git archive` died with SIGBUS on the external drive under load, and every
    controller cycle re-read the repository for a commit that was already deployed."""
    import subprocess

    calls = []

    def flaky(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(-10, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"tar-bytes")

    monkeypatch.setattr(ctl_mod.subprocess, "run", flaky)
    monkeypatch.setattr(ctl_mod, "_ARCHIVES", ctl_mod.ArchiveCache(sleep=lambda s: None))
    assert ctl_mod.git_archive("abc") == b"tar-bytes"
    assert ctl_mod.git_archive("abc") == b"tar-bytes"  # cached
    assert len(calls) == 3


def test_git_archive_gives_up_after_three_attempts(monkeypatch):
    import subprocess

    def broken(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(ctl_mod.subprocess, "run", broken)
    monkeypatch.setattr(ctl_mod, "_ARCHIVES", ctl_mod.ArchiveCache(sleep=lambda s: None))
    with pytest.raises(subprocess.CalledProcessError):
        ctl_mod.git_archive("nope")


def test_run_many_logs_the_traceback_of_a_failing_cycle(world):
    from landuse_filter.application.controller import run_many
    from landuse_filter.application.site_cache import SiteCache

    c, _ = world
    logs: list[str] = []
    c.log = logs.append

    def boom(now=None):
        raise RuntimeError("hub down")

    c.cycle = boom

    # the failing namespace never finishes; end the loop at its first sleep
    def sleeper(seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_many([c], SiteCache(), interval=1, sleep=sleeper, emit=lambda _: None)
    assert f"{c.work_fp}: cycle failed (RuntimeError: hub down)" == logs[0]
    assert any(line.startswith("  | ") and "boom" in line for line in logs[1:])


def _failing(times: int, payload: bytes = b"tar"):
    import subprocess

    calls: list[str] = []

    def read(ref: str) -> bytes:
        calls.append(ref)
        if len(calls) <= times:
            raise subprocess.CalledProcessError(135, "git archive")
        return payload

    return read, calls


def test_archive_cache_reads_each_ref_once():
    read, calls = _failing(0)
    archive = ctl_mod.ArchiveCache(read, sleep=lambda s: None)
    assert archive("abc") == archive("abc") == b"tar"
    assert calls == ["abc"]
    archive("def")
    assert calls == ["abc", "def"]


def test_archive_retries_transient_faults_without_sleeping_for_real():
    read, calls = _failing(2)
    sleeps: list[float] = []
    archive = ctl_mod.ArchiveCache(read, sleep=sleeps.append, delay=5.0)
    assert archive("abc") == b"tar"
    assert len(calls) == 3
    assert sleeps == [5.0, 5.0]


def test_archive_raises_after_the_last_attempt_and_does_not_cache_failure():
    import subprocess

    read, calls = _failing(99)
    sleeps: list[float] = []
    archive = ctl_mod.ArchiveCache(read, sleep=sleeps.append, attempts=3)
    with pytest.raises(subprocess.CalledProcessError):
        archive("abc")
    assert len(calls) == 3
    assert len(sleeps) == 2  # no sleep after the final failure
    assert archive.cache == {}


def test_commit_retries_a_ref_read_that_races_with_another_fetch(monkeypatch):
    """Regression: `git rev-parse origin/main` failed (exit 128) while another process was
    updating the ref, and the whole admission cycle was lost."""
    import subprocess

    calls = []

    def flaky(cmd, **kwargs):
        calls.append(cmd)
        if cmd[3] == "rev-parse" and calls.count(cmd) < 2:
            raise subprocess.CalledProcessError(128, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n")

    monkeypatch.setattr(ctl_mod.subprocess, "run", flaky)
    monkeypatch.setattr(ctl_mod.time, "sleep", lambda s: None)
    assert ctl_mod.commit("origin/main") == "abc123"


def test_commit_gives_up_when_the_ref_stays_unreadable(monkeypatch):
    import subprocess

    def broken(cmd, **kwargs):
        if cmd[3] == "fetch":
            return subprocess.CompletedProcess(cmd, 1)
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(ctl_mod.subprocess, "run", broken)
    monkeypatch.setattr(ctl_mod.time, "sleep", lambda s: None)
    with pytest.raises(subprocess.CalledProcessError):
        ctl_mod.commit("origin/main")
