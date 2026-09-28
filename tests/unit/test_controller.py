from datetime import datetime
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from landuse_filter.adapters import g5k
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import controller as ctl_mod
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
    assert len(report["submitted"]) == 2  # two free GPUs -> two one-GPU jobs
    assert fake.policy_checks == 4  # before and after each submission
    assert fake.submitted[0][:2] == ["-q", "abaca"]
    chunks = [set(a["chunks"]) for a in c.live()]
    assert not chunks[0] & chunks[1]  # disjoint assignments


def test_second_cycle_does_not_duplicate(world):
    c, fake = world
    c.cycle(NOW)
    c.cycle(NOW)
    assert all(len(a["chunks"]) for a in c.live())
    taken = [ch for a in c.live() for ch in a["chunks"]]
    assert len(taken) == len(set(taken))


def test_ended_jobs_release_their_chunks(world):
    c, fake = world
    c.cycle(NOW)
    fake.jobs["nancy"] = []  # jobs finished without results
    c.reconcile()
    assert c.live() == []
    assert {a["state"] for a in c.ledger()} == {"ended"}


def test_crash_after_oarsub_is_adopted(world):
    c, fake = world
    c.cycle(NOW)
    a = c.live()[0]
    a.update(state="submitting", job_id=None)  # as if we died before recording the id
    c.store.write_json(f"assignments/{a['id']}.json", a)
    c.reconcile()
    adopted = c.store.read_json(f"assignments/{a['id']}.json")
    assert adopted["state"] == "submitted"
    assert adopted["job_id"]


def test_besteffort_only_cluster_is_remembered(world, monkeypatch):
    c, fake = world
    fake.refuse = "# You can only access the required resources in besteffort."
    report = c.cycle(NOW)
    assert report["submitted"] == []
    assert c.store.exists("access/nancy_gres.json")
    fake.refuse = None
    assert c.cycle(NOW)["submitted"] == []  # not retried


def test_paused_submits_nothing(world):
    c, fake = world
    c.settings.paused = True
    assert c.cycle(NOW)["submitted"] == []


def test_late_start_is_cancelled_and_cluster_backed_off(world):
    c, fake = world
    fake.late = True
    assert c.cycle(NOW)["submitted"] == []
    assert fake.cancelled
    assert c.live() == []
    fake.late = False
    assert c.cycle(NOW)["submitted"] == []  # still backed off


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
    assert job.name not in {a["name"] for a in c.live()}


def test_candidate_namespace_and_window(world):
    c, fake = world
    c.settings.namespace, c.settings.window = "w128", 128
    c.cycle(NOW)
    a = c.live()[0]
    assert a["fp"] == f"{c.fp}-w128"
    assert a["window"] == 128
    assert a["engine_kwargs"]["max_running_requests"] == 128
    assert a["provenance"]["config_fingerprint"] == c.fp  # same generation identity


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
    assert len(c.cycle(NOW)["submitted"]) == 2  # regular jobs preempt besteffort ones
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
    )
    other.reconcile()  # sees none of nancy's jobs
    assert [a["id"] for a in c.live()] == [a["id"] for a in theirs]  # untouched
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
        return {"pending_chunks": 0, "live_jobs": 0, "submitted": []}

    c.cycle = flaky
    logs, sleeps, out = [], [], []
    c.log = logs.append
    run_loop(c, interval=5, sleep=sleeps.append, emit=out.append)
    assert calls["n"] == 2
    assert sleeps == [5]
    assert "cycle failed (RuntimeError" in logs[0]
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
    )
    b = Controller(
        c.store,
        Settings(datasets=["benchmark"], sites=["nancy"], gpu_models=["l40s"], namespace="gpu-b"),
        log=lambda m: None,
        sites=cache,
    )
    cache.reset()
    a.cycle(NOW)
    b.cycle(NOW)
    assert calls["status"] == 1  # one status call per site per cycle
    assert calls["jobs"] <= 3  # initial + one refresh per submission, not one per controller
    assert {x["fp"] for x in a.live()} == {f"{a.fp}-gpu-a"}
    assert {x["fp"] for x in b.live()} == {f"{b.fp}-gpu-b"}


def test_run_many_drops_finished_namespaces(world):
    from landuse_filter.application.controller import run_many
    from landuse_filter.application.site_cache import SiteCache

    c, fake = world
    done = Controller(
        c.store, Settings(datasets=["none"], sites=["nancy"], namespace="gpu-z"), log=lambda m: None
    )
    out, sleeps = [], []
    run_many([done], SiteCache(), interval=1, sleep=sleeps.append, emit=out.append)
    assert len(out) == 1
    assert sleeps == []
