from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import CorruptJSONLError, WorkStore
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
        self.fail_checks: set[int] = set()  # 1-based indexes of failing policy checks
        self.events: list[str] = []
        self.late = False
        self.cancelled = []

    def our_jobs(self, site):
        return list(self.jobs.get(site, []))

    def submit(self, site, args):
        self.events.append(f"submit:{site}")
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
        self.events.append(f"check:{site}")
        if self.policy_checks in self.fail_checks:
            raise g5k.RemoteError(f"{site}: exit 1: usage policy violated")

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


def batch_world(world):
    c, fake = world
    c.settings.policy_check = "per-batch"
    c.policy.mode = "per-batch"
    return c, fake


def test_per_batch_checks_once_before_and_once_after_all_jobs(world):
    c, fake = batch_world(world)
    assert len(c.cycle(NOW).submitted) == 2
    assert fake.events == [
        "check:nancy",
        "submit:nancy",
        "submit:nancy",
        "check:nancy",
    ]


def test_per_batch_failing_pre_check_blocks_the_site_for_the_cycle(world):
    c, fake = batch_world(world)
    fake.fail_checks = {1}
    assert c.cycle(NOW).submitted == []
    assert fake.events == ["check:nancy"]  # no oarsub, no retry, no post-check
    assert [a.state for a in c.ledger()] == ["failed_submit"]
    fake.fail_checks = set()
    assert len(c.cycle(NOW).submitted) == 2  # the next cycle checks again and proceeds


def test_per_batch_violation_after_the_batch_is_logged_loudly(world):
    c, fake = batch_world(world)
    logs = []
    c.log = c.policy.log = logs.append
    fake.fail_checks = {2}
    assert len(c.cycle(NOW).submitted) == 2  # the jobs exist; the violation is surfaced
    assert any(m.startswith("POLICY VIOLATION: nancy") for m in logs)
    assert fake.policy_checks == 2


def test_per_batch_refused_slot_is_still_checked_before_and_after(world):
    c, fake = batch_world(world)
    fake.refuse = "oarsub: quota"
    assert c.cycle(NOW).submitted == []
    assert fake.events == ["check:nancy", "submit:nancy", "check:nancy"]


def test_per_batch_without_submissions_runs_no_check(world):
    c, fake = batch_world(world)
    c.settings.paused = True
    assert c.cycle(NOW).submitted == []
    assert fake.policy_checks == 0
    c.settings.paused = False
    fake.jobs["nancy"] = [g5k.Job("nancy", "9", "other", "Running", "abaca")]
    c.settings.max_jobs_per_site = 1  # site full: nothing is submitted
    assert c.cycle(NOW).submitted == []
    assert fake.policy_checks == 0


def test_per_job_failing_pre_check_refuses_the_slot_as_before(world):
    c, fake = world
    fake.fail_checks = {1}
    assert c.cycle(NOW).submitted == []
    assert fake.policy_checks == 1


def test_second_cycle_does_not_duplicate(world):
    c, fake = world
    c.cycle(NOW)
    c.cycle(NOW)
    assert all(len(a.chunks) for a in c.live())
    taken = [ch for a in c.live() for ch in a.chunks]
    assert len(taken) == len(set(taken))


def test_cycle_fails_closed_when_a_corrupt_plan_row_hides_unfinished_work(world):
    c, fake = world
    c.store.append_jsonl(c.complete_log, [{"chunk_id": "c0"}, {"chunk_id": "c2"}])
    plan = c.store.path(f"plans/benchmark/{c.fp}/chunks.jsonl")
    content = plan.read_bytes()
    marker = b'"chunk_id": "c1"'
    plan.write_bytes(content.replace(marker, b'"chunk_id": "\xff1"', 1))

    with pytest.raises(CorruptJSONLError, match="line 2"):
        c.cycle(NOW)

    assert fake.submitted == []


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
    assert out == ['{"pending_chunks": 0, "live_jobs": 0, "submitted": []}']
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


def test_results_are_ingested_again_during_a_long_run_of_submissions(world, monkeypatch):
    """Regression: a cycle submitting dozens of jobs (20 s wait + deploy each) ingested results
    only at its start, so counts lagged by tens of minutes (23 minutes on 2026-10-01)."""
    c, _ = world
    pulls = []
    monkeypatch.setattr(c, "pull", lambda: pulls.append(1))
    monkeypatch.setattr(ctl_mod, "PULL_INTERVAL", 0.0)
    report = c.cycle(NOW)
    assert len(report.submitted) == 2
    assert len(pulls) == 1 + 2 + 1  # start, before each submission, end


def test_no_extra_ingest_when_submissions_are_quick(world, monkeypatch):
    c, _ = world
    pulls = []
    monkeypatch.setattr(c, "pull", lambda: pulls.append(1))
    monkeypatch.setattr(ctl_mod, "PULL_INTERVAL", 3600.0)
    c.cycle(NOW)
    assert len(pulls) == 2  # start and end of the cycle


# --- long night jobs with a shorter fallback (ADR-0016) -------------------------------


def _night_world(world, long=120):
    """The world with a non-production cluster: night jobs may exceed the day walltime."""
    c, fake = world
    c.store.write_json("inventory.json", [{**GRES, "queues": ["default"]}])
    c.settings.night_walltime = timedelta(minutes=long)
    c.settings.immediate_in_night = False
    c.settings.night_fallback_walltime = timedelta(minutes=30)
    return c, fake


def _minutes(fake):
    import re

    found = (re.search(r"walltime=(\d+):(\d+)", " ".join(args)).groups() for args in fake.submitted)
    return [int(h) * 60 + int(m) for h, m in found]


def _late_when_long(fake, monkeypatch):
    def start(site, job_id):
        long = "walltime=2:00" in " ".join(fake.submitted[int(job_id) - 101])
        return ("Waiting", 4_000_000_000) if long else ("Running", None)

    monkeypatch.setattr(g5k, "scheduled_start", start)


def _backed_off(c):
    (cluster,) = ctl_mod.load_clusters(c.store)
    return c.memory.backed_off(cluster, NOW)


def test_long_night_job_that_starts_in_time_stays_long(world):
    c, fake = _night_world(world)
    assert len(c.cycle(NOW).submitted) == 1  # one long job takes every chunk
    assert _minutes(fake) == [120]
    assert fake.cancelled == []


def test_long_night_job_starting_late_falls_back_to_the_short_walltime(world, monkeypatch):
    c, fake = _night_world(world)
    _late_when_long(fake, monkeypatch)
    assert c.cycle(NOW).submitted
    assert _minutes(fake)[:2] == [120, 30]
    assert len(fake.cancelled) >= 1
    assert not _backed_off(c)
    live = c.live()
    assert {a.walltime_s for a in live} == {1800}
    taken = [ch for a in live for ch in a.chunks]
    assert len(taken) == len(set(taken))
    assert "cancelled_late_start" in {a.state for a in c.ledger()}


def test_fallback_capacity_is_recomputed_for_the_short_job(world, monkeypatch):
    c, fake = _night_world(world)
    _late_when_long(fake, monkeypatch)
    c.cycle(NOW)
    cancelled = [a for a in c.ledger() if a.state == "cancelled_late_start"]
    assert cancelled
    assert max(len(a.chunks) for a in c.live()) <= min(len(a.chunks) for a in cancelled)


def test_long_refused_then_fallback_is_submitted(world, monkeypatch):
    c, fake = _night_world(world)
    real = fake.submit
    calls = []

    def submit(site, args):
        calls.append(args)
        if len(calls) == 1:
            raise g5k.RemoteError("refused")
        return real(site, args)

    monkeypatch.setattr(g5k, "submit", submit)
    assert c.cycle(NOW).submitted
    assert _minutes(fake)[0] == 30


def test_both_walltimes_failing_backs_the_cluster_off_once(world):
    c, fake = _night_world(world)
    fake.late = True
    logs = []
    c.log = logs.append
    assert c.cycle(NOW).submitted == []
    assert _minutes(fake) == [120, 30]
    assert sum("backing off" in m for m in logs) == 1
    assert _backed_off(c)


def test_day_window_never_uses_the_long_walltime(world):
    c, fake = _night_world(world)
    day = datetime(2026, 9, 29, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    c.now = day
    c.cycle(day)
    assert len(fake.submitted) == 2
    assert _minutes(fake) == [60, 60]


def test_queue_limit_is_the_night_value_only_at_night():
    c = Controller.__new__(Controller)
    c.settings = Settings(datasets=[], sites=[], max_queued_per_site=2)
    assert (c.queue_limit(night=True), c.queue_limit(night=False)) == (2, 2)
    c.settings.night_max_queued_per_site = 15
    assert (c.queue_limit(night=True), c.queue_limit(night=False)) == (15, 2)
    c.settings.night_max_queued_per_site = 0
    assert c.queue_limit(night=True) == 0


def test_queue_room_counts_waiting_jobs_against_the_window_limit():
    c = Controller.__new__(Controller)
    c.settings = Settings(datasets=[], sites=[], max_queued_per_site=1, night_max_queued_per_site=3)
    jobs = {"nancy": [g5k.Job("nancy", str(i), "n", "Waiting", "abaca") for i in range(2)]}
    assert not c.queue_room("nancy", jobs)
    assert c.queue_room("nancy", jobs, night=True)


def test_run_command_passes_the_fallback_and_night_queue_options(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = {}
    monkeypatch.setattr(cli, "_controller", lambda work, s: seen.setdefault("s", s))
    monkeypatch.setattr(ctl_mod, "run_loop", lambda *a, **k: None)
    args = ["run", "--datasets", "d", "--work", str(tmp_path)]
    args += ["--night-walltime-minutes", "120", "--night-fallback-walltime-minutes", "45"]
    args += ["--night-max-queued-per-site", "15"]
    result = CliRunner().invoke(cli.g5k_app, args)
    assert result.exit_code == 0, result.output
    s = seen["s"]
    assert s.night_fallback_walltime == timedelta(minutes=45)
    assert (s.night_walltime, s.night_max_queued_per_site) == (timedelta(minutes=120), 15)


# --- long day jobs with a short fallback and a self-throttle (ADR-0017) ---------------

DAY = datetime(2026, 9, 29, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))


def _day_world(world, day=60, production=False):
    c, fake = world
    queues = ["abaca"] if production else ["default"]
    c.store.write_json("inventory.json", [{**GRES, "queues": queues}])
    c.settings.walltime = timedelta(minutes=30)
    c.settings.day_walltime = timedelta(minutes=day)
    c.now = DAY
    return c, fake


def _late_when(fake, monkeypatch, long_wall="walltime=1:00"):
    def start(site, job_id):
        long = long_wall in " ".join(fake.submitted[int(job_id) - 101])
        return ("Waiting", 4_000_000_000) if long else ("Running", None)

    monkeypatch.setattr(g5k, "scheduled_start", start)


def _attempt(c, fake, now):
    """One cycle that makes at most one submission (the previous job is gone)."""
    c.settings.max_jobs_total = 1
    fake.jobs["nancy"] = []
    c.now = now
    c.cycle(now)


def test_day_long_job_that_starts_in_time_stays_long(world):
    c, fake = _day_world(world)
    assert c.cycle(DAY).submitted
    assert set(_minutes(fake)) == {60}


def test_day_long_job_starting_late_falls_back_and_logs_it(world, monkeypatch):
    c, fake = _day_world(world)
    _late_when(fake, monkeypatch)
    logs = []
    c.log = logs.append
    assert c.cycle(DAY).submitted
    assert _minutes(fake)[:2] == [60, 30]
    assert not _backed_off(c)
    assert "long_walltime fallback nancy/gres 60->30" in logs


def test_day_long_refused_then_short_is_submitted(world, monkeypatch):
    c, fake = _day_world(world)
    real, calls = fake.submit, []

    def submit(site, args):
        calls.append(args)
        if len(calls) == 1:
            raise g5k.RemoteError("refused")
        return real(site, args)

    monkeypatch.setattr(g5k, "submit", submit)
    assert c.cycle(DAY).submitted
    assert _minutes(fake)[0] == 30


def test_three_failed_long_attempts_pause_long_walltimes_for_an_hour(world, monkeypatch):
    c, fake = _day_world(world)
    _late_when(fake, monkeypatch)
    logs = []
    c.log = logs.append
    for _ in range(3):
        _attempt(c, fake, DAY)
    assert sum("long_walltime throttle nancy/gres" in m for m in logs) == 1
    before = len(fake.submitted)
    _attempt(c, fake, DAY + timedelta(minutes=30))
    assert _minutes(fake)[before:] == [30]
    before = len(fake.submitted)
    _attempt(c, fake, DAY + timedelta(hours=1, minutes=1))
    assert _minutes(fake)[before] == 60


def test_a_successful_long_job_resets_the_failure_count(world, monkeypatch):
    c, fake = _day_world(world)
    _late_when(fake, monkeypatch)
    _attempt(c, fake, DAY)
    _attempt(c, fake, DAY)
    monkeypatch.setattr(g5k, "scheduled_start", lambda site, job_id: ("Running", None))
    _attempt(c, fake, DAY)
    _late_when(fake, monkeypatch)
    _attempt(c, fake, DAY)
    _attempt(c, fake, DAY)
    (cluster,) = ctl_mod.load_clusters(c.store)
    assert not c.memory.long_throttled(cluster, DAY)


def test_production_cluster_uses_the_day_long_walltime_directly(world):
    c, fake = _day_world(world, production=True)
    assert c.cycle(DAY).submitted
    assert set(_minutes(fake)) == {60}


@pytest.mark.parametrize("production", [False, True])
@pytest.mark.parametrize("day", [None, 30])
def test_day_walltime_unset_or_equal_changes_nothing(world, production, day):
    c, fake = _day_world(world, production=production)
    c.settings.day_walltime = None if day is None else timedelta(minutes=day)
    fake.late = True
    logs = []
    c.log = logs.append
    assert c.cycle(DAY).submitted == []
    assert _minutes(fake) == [30]  # one short submission, then back-off, as before
    assert not any("long_walltime" in m for m in logs)


def test_day_options_reach_the_controller_settings(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = {}
    monkeypatch.setattr(cli, "_controller", lambda work, s: seen.setdefault("s", s))
    monkeypatch.setattr(ctl_mod, "run_loop", lambda *a, **k: None)
    base = ["run", "--datasets", "d", "--work", str(tmp_path)]
    args = [*base, "--walltime-minutes", "30", "--day-walltime-minutes", "60"]
    args += ["--day-long-max-failures", "5", "--chunk-overflow", "1.6"]
    assert CliRunner().invoke(cli.g5k_app, args).exit_code == 0
    assert seen["s"].day_walltime == timedelta(minutes=60)
    assert seen["s"].day_long_max_failures == 5
    assert seen["s"].chunk_overflow == 1.6
    seen.clear()
    assert CliRunner().invoke(cli.g5k_app, [*base, "--chunk-overflow", "0.9"]).exit_code != 0
    seen.clear()
    assert CliRunner().invoke(cli.g5k_app, [*base, "--walltime-minutes", "45"]).exit_code == 0
    assert seen["s"].day_walltime == timedelta(minutes=45)  # unset: equals the short value
    assert seen["s"].chunk_overflow == 1.2  # unset: previous behaviour


def _age_besteffort(c, fake, *, queue, minutes):
    c.cycle(NOW)
    a = c.live()[0]
    a.submitted_at = (NOW - timedelta(minutes=minutes)).isoformat()
    c.save(a)
    job = next(j for j in fake.jobs["nancy"] if j.name == a.name)
    fake.jobs["nancy"][fake.jobs["nancy"].index(job)] = g5k.Job(
        job.site, job.job_id, job.name, "Waiting", queue
    )
    c.now = NOW
    return a, job


def test_stale_besteffort_job_is_reaped_and_released(world):
    c, fake = world
    a, job = _age_besteffort(c, fake, queue="besteffort", minutes=25)
    c.reconcile()
    assert job.job_id in fake.cancelled
    assert c.store.exists("backoff/nancy_gres.json")
    assert a.name not in {x.name for x in c.live()}
    assert next(x for x in c.ledger() if x.id == a.id).state == "cancelled_stale"


def test_recent_besteffort_and_other_queues_are_left_alone(world):
    c, fake = world
    _age_besteffort(c, fake, queue="besteffort", minutes=5)
    c.reconcile()
    assert not fake.cancelled
    _age_besteffort(c, fake, queue="night", minutes=600)
    c.reconcile()
    assert not fake.cancelled


# --- immediate-start jobs in the night window (ADR-0027) ------------------------------


def _types(fake):
    return ["night" if "night" in args else "now" for args in fake.submitted]


def test_night_window_submits_immediate_jobs_on_free_gpus_without_night_type(world):
    c, fake = _night_world(world)
    c.settings.immediate_in_night = True
    c.cycle(NOW)
    assert len(fake.submitted) == 2
    assert _types(fake) == ["now", "now"]
    assert _minutes(fake) == [60, 60]
    assert fake.policy_checks >= 2  # usagepolicycheck still runs for these jobs


def test_night_window_flag_off_keeps_queued_night_jobs_only(world):
    c, fake = _night_world(world)
    c.settings.immediate_in_night = False
    c.cycle(NOW)
    assert set(_types(fake)) == {"night"}


def test_night_window_without_free_gpu_has_no_immediate_slot(world):
    c, _ = _night_world(world)
    c.settings.immediate_in_night = True
    (cluster,) = ctl_mod.load_clusters(c.store)
    slots = c._cluster_slots(cluster, {}, NOW, {})
    assert [s for s in slots if s and not s.queued] == []


def test_day_window_has_no_extra_immediate_slot(world):
    c, _ = _night_world(world)
    c.settings.immediate_in_night = True
    (cluster,) = ctl_mod.load_clusters(c.store)
    slots = c._cluster_slots(cluster, {}, DAY, {})
    assert slots[0] is None


def test_run_command_immediate_in_night_defaults_on_and_can_be_disabled(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from landuse_filter.cli import g5k as cli

    seen = []
    monkeypatch.setattr(cli, "_controller", lambda work, s: seen.append(s))
    monkeypatch.setattr(ctl_mod, "run_loop", lambda *a, **k: None)
    base = ["run", "--datasets", "d", "--work", str(tmp_path)]
    for extra in ([], ["--no-immediate-in-night"]):
        assert CliRunner().invoke(cli.g5k_app, base + extra).exit_code == 0
    assert [s.immediate_in_night for s in seen] == [True, False]
