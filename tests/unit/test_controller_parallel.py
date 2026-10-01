"""Parallel submission across sites (ADR-0020): same decisions, concurrent launches."""

import threading
import time
from itertools import pairwise

import pyarrow as pa
import pytest

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import controller as ctl_mod
from landuse_filter.application.controller import Controller, Settings
from tests.unit.test_controller import GRES, NOW, FakeG5K

SITES = ("nancy", "lille", "rennes")


class ThreadedG5K(FakeG5K):
    """Records when and on which thread each oarsub of each site runs."""

    def __init__(self, sites=SITES, pause=0.0, rendezvous=None):
        super().__init__()
        self.jobs = {s: [] for s in sites}
        self.lock = threading.Lock()
        self.pause = pause
        self.rendezvous = rendezvous  # barrier every site's first oarsub must reach together
        self.spans: list[tuple[str, int, float, float]] = []
        self.in_flight: dict[str, int] = {}
        self.max_in_flight: dict[str, int] = {}
        self.first_seen: set[str] = set()

    def submit(self, site, args):
        with self.lock:
            self.in_flight[site] = self.in_flight.get(site, 0) + 1
            self.max_in_flight[site] = max(self.max_in_flight.get(site, 0), self.in_flight[site])
            first = site not in self.first_seen
            self.first_seen.add(site)
        start = time.monotonic()
        try:
            if self.rendezvous is not None and first:
                self.rendezvous.wait(timeout=5)  # BrokenBarrierError unless sites run together
            time.sleep(self.pause)
            with self.lock:
                return super().submit(site, args)
        finally:
            with self.lock:
                self.in_flight[site] -= 1
                self.spans.append((site, threading.get_ident(), start, time.monotonic()))

    def policy_check(self, site):
        with self.lock:
            super().policy_check(site)


def build(tmp_path, monkeypatch, *, workers, fake=None, sites=SITES, chunks=12, **settings):
    fake = fake or ThreadedG5K(sites)
    monkeypatch.setattr(ctl_mod, "POST_SUBMIT_WAIT", 0.0)
    for name in ("our_jobs", "submit", "policy_check", "scheduled_start", "cancel"):
        monkeypatch.setattr(g5k, name, getattr(fake, name))

    def status(site):
        host = f"gres-1.{site}.grid5000.fr"
        free = {"hard": "alive", "free_slots": 48, "freeable_slots": 0, "busy_slots": 0}
        return {"nodes": {host: free}}

    monkeypatch.setattr(g5k, "site_status", status)
    monkeypatch.setattr(g5k, "rsync", lambda *a, **k: None)
    monkeypatch.setattr(g5k, "ssh", lambda *a, **k: "")
    monkeypatch.setattr(g5k, "deploy_code", lambda site, commit, archive: "luf/code/x")
    monkeypatch.setattr(ctl_mod, "git_archive", lambda ref: b"")
    monkeypatch.setattr(ctl_mod, "commit", lambda: "abc")
    store = WorkStore(tmp_path)
    store.write_json("inventory.json", [{**GRES, "site": s} for s in sites])
    c = Controller(
        store,
        Settings(
            datasets=["benchmark"],
            sites=list(sites),
            gpu_models=["l40s"],
            submit_workers=workers,
            **settings,
        ),
        log=lambda m: None,
        remote=DirRemote(tmp_path / "bucket"),
    )
    for i in range(chunks):
        cid = f"c{i:02}"
        table = pa.table(
            {"text_sha256": [f"{cid}s"], "text": ["t"], "input_ids": [[1]]}, schema=CHUNK
        )
        store.write_chunk(cid, table)
        store.append_jsonl(
            f"plans/benchmark/{c.fp}/chunks.jsonl",
            [{"chunk_id": cid, "order": [0, i], "size": 3000}],
        )
    return c, fake


def test_sites_are_submitted_to_concurrently(tmp_path, monkeypatch):
    fake = ThreadedG5K(rendezvous=threading.Barrier(len(SITES)))
    c, _ = build(tmp_path, monkeypatch, workers=8, fake=fake)
    # the barrier only opens when the three sites' first oarsub run at the same moment
    assert len(c.cycle(NOW).submitted) == 6
    assert len({tid for _, tid, _, _ in fake.spans}) == len(SITES)


def test_one_site_is_submitted_to_in_turn(tmp_path, monkeypatch):
    fake = ThreadedG5K(pause=0.02)
    c, _ = build(tmp_path, monkeypatch, workers=8, fake=fake)
    c.cycle(NOW)
    assert set(fake.max_in_flight) == set(SITES)
    assert set(fake.max_in_flight.values()) == {1}
    for site in SITES:
        spans = sorted(s for s in fake.spans if s[0] == site)
        assert len({tid for _, tid, _, _ in spans}) == 1  # one worker owns the site
        assert all(a[3] <= b[2] for a, b in pairwise(spans))


def test_workers_are_capped_by_the_setting(tmp_path, monkeypatch):
    fake = ThreadedG5K(pause=0.02)
    c, _ = build(tmp_path, monkeypatch, workers=2, fake=fake)
    c.cycle(NOW)
    spans = sorted(fake.spans, key=lambda s: s[2])
    assert len({tid for _, tid, _, _ in spans}) <= 2


def summary(c, report):
    ledger = sorted((a.site, a.cluster, tuple(a.chunks), a.walltime_s) for a in c.live())
    return ledger, [s.rsplit(":", 1)[0] for s in report.submitted]


@pytest.mark.parametrize("chunks", [12, 5, 1])
def test_results_equal_the_sequential_run(tmp_path, monkeypatch, chunks):
    seq, _ = build(tmp_path / "seq", monkeypatch, workers=1, chunks=chunks)
    expected = summary(seq, seq.cycle(NOW))
    par, _ = build(tmp_path / "par", monkeypatch, workers=8, chunks=chunks)
    got = summary(par, par.cycle(NOW))
    assert got == expected
    assert len(got[1]) == min(chunks, 6)


def test_no_chunk_is_assigned_twice(tmp_path, monkeypatch):
    c, _ = build(tmp_path, monkeypatch, workers=8, chunks=4)
    c.cycle(NOW)
    c.cycle(NOW)
    taken = [ch for a in c.live() for ch in a.chunks]
    assert len(taken) == len(set(taken))
    assert set(taken) <= {f"c{i:02}" for i in range(4)}


def test_caps_hold_in_parallel(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8, max_jobs_total=4, max_jobs_per_site=1)
    report = c.cycle(NOW)
    assert len(report.submitted) == 3  # one per site, per-site cap before the total cap
    c.settings.max_jobs_per_site = 2
    assert len(c.cycle(NOW).submitted) == 1  # total cap 4 minus the 3 live jobs


def test_a_refusing_site_does_not_stop_the_others_and_frees_its_chunks(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8, chunks=12)
    real = fake.submit

    def submit(site, args):
        if site == "lille":
            raise g5k.RemoteError("refused")
        return real(site, args)

    monkeypatch.setattr(g5k, "submit", submit)
    report = c.cycle(NOW)
    assert sorted({s.split("/")[0] for s in report.submitted}) == ["nancy", "rennes"]
    assert {a.site for a in c.live()} == {"nancy", "rennes"}
    assert "failed_submit" in {a.state for a in c.ledger()}
    monkeypatch.setattr(g5k, "submit", real)
    assert any(s.startswith("lille/") for s in c.cycle(NOW).submitted)  # chunks were released


def test_a_crashing_worker_is_logged_and_the_others_still_submit(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8)
    logs = []
    c.log = logs.append
    real = fake.submit

    def submit(site, args):
        if site == "lille":
            raise ValueError("bug")
        return real(site, args)

    monkeypatch.setattr(g5k, "submit", submit)
    report = c.cycle(NOW)
    assert sorted({s.split("/")[0] for s in report.submitted}) == ["nancy", "rennes"]
    assert any("lille: submission worker failed (ValueError: bug)" in m for m in logs)


def test_per_batch_policy_check_runs_twice_per_site_in_parallel(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8, policy_check="per-batch")
    assert len(c.cycle(NOW).submitted) == 6
    checks = [e for e in fake.events if e.startswith("check:")]
    assert sorted(checks) == sorted(f"check:{s}" for s in SITES for _ in range(2))
    for site in SITES:  # before the first oarsub and after the last of that site
        mine = [e for e in fake.events if e.endswith(f":{site}")]
        assert mine == [f"check:{site}", f"submit:{site}", f"submit:{site}", f"check:{site}"]


def test_per_job_policy_check_is_unchanged_in_parallel(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8)
    c.cycle(NOW)
    assert fake.policy_checks == 12  # before and after each of 6 jobs


def test_cycle_logs_one_summary_line(tmp_path, monkeypatch):
    for workers in (1, 8):
        c, _ = build(tmp_path / str(workers), monkeypatch, workers=workers)
        logs = []
        c.log = logs.append
        c.cycle(NOW)
        (line,) = [m for m in logs if m.startswith("cycle submitted")]
        assert line.startswith("cycle submitted 6 jobs in ")
        assert line.endswith(" s (sites 3)")


def night(c):
    c.store.write_json(
        "inventory.json", [{**GRES, "site": s, "queues": ["default"]} for s in SITES]
    )
    from datetime import timedelta

    c.settings.night_walltime = timedelta(minutes=120)
    c.settings.night_fallback_walltime = timedelta(minutes=30)


def test_long_walltime_fallback_works_in_parallel(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8, chunks=40)
    night(c)

    walls = {}
    real = fake.submit

    def submit(site, args):
        job_id = real(site, args)
        walls[job_id] = "walltime=2:00" in " ".join(args)
        return job_id

    monkeypatch.setattr(g5k, "submit", submit)
    monkeypatch.setattr(
        g5k,
        "scheduled_start",
        lambda site, job_id: ("Waiting", 4_000_000_000) if walls[job_id] else ("Running", None),
    )
    report = c.cycle(NOW)
    assert report.submitted
    live = c.live()
    assert {a.walltime_s for a in live} == {1800}  # every site fell back to the short job
    taken = [ch for a in live for ch in a.chunks]
    assert len(taken) == len(set(taken))  # no chunk twice, cancelled attempts released theirs
    assert "cancelled_late_start" in {a.state for a in c.ledger()}
    for state in ("submitted", "cancelled_late_start"):
        assert state in {a.state for a in c.ledger()}


def test_memory_counters_are_not_lost_when_clusters_fail_together(tmp_path, monkeypatch):
    c, fake = build(tmp_path, monkeypatch, workers=8, chunks=40)
    night(c)
    fake.late = True  # every attempt on every site is cancelled as a late start
    c.cycle(NOW)
    for site in SITES:
        assert c.store.exists(f"backoff/{site}_gres.json")  # one per site, none overwritten
