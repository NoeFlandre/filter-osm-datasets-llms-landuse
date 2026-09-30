"""Controller cycles are idempotent when nothing changes between two of them."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from landuse_filter.adapters import g5k
from landuse_filter.adapters.remote import DirRemote
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
    "nodes": 4,
    "queues": ["abaca", "production"],
    "exotic": False,
}


class FakeG5K:
    """Jobs submitted stay in the queue; the site's free view shrinks as nodes fill up."""

    def __init__(self, free_nodes: int) -> None:
        self.free_nodes = free_nodes
        self.jobs: list[g5k.Job] = []
        self.submitted: list[list[str]] = []

    def our_jobs(self, site):
        return list(self.jobs)

    def submit(self, site, args):
        self.submitted.append(args)
        job_id = str(100 + len(self.submitted))
        self.jobs.append(g5k.Job(site, job_id, args[args.index("-n") + 1], "Waiting", "abaca"))
        return job_id

    def site_status(self, site):
        free = max(self.free_nodes - len(self.jobs), 0)
        alive = {"hard": "alive", "free_slots": 48, "freeable_slots": 0, "busy_slots": 0}
        busy = {"hard": "alive", "free_slots": 0, "freeable_slots": 0, "busy_slots": 48}
        return {
            "nodes": {f"gres-{i}.nancy.grid5000.fr": alive if i < free else busy for i in range(4)}
        }


def build(tmp_path, monkeypatch, *, free_nodes, n_chunks, max_jobs, per_site):
    fake = FakeG5K(free_nodes)
    monkeypatch.setattr(ctl_mod, "POST_SUBMIT_WAIT", 0.0)
    monkeypatch.setattr(g5k, "our_jobs", fake.our_jobs)
    monkeypatch.setattr(g5k, "submit", fake.submit)
    monkeypatch.setattr(g5k, "policy_check", lambda site: None)
    monkeypatch.setattr(g5k, "scheduled_start", lambda site, job_id: ("Running", None))
    monkeypatch.setattr(g5k, "cancel", lambda site, job_id: None)
    monkeypatch.setattr(g5k, "site_status", fake.site_status)
    monkeypatch.setattr(g5k, "rsync", lambda *a, **k: None)
    monkeypatch.setattr(g5k, "ssh", lambda *a, **k: "")
    monkeypatch.setattr(g5k, "deploy_code", lambda site, commit, archive: "luf/code/x")
    monkeypatch.setattr(ctl_mod, "git_archive", lambda ref: b"")
    monkeypatch.setattr(ctl_mod, "commit", lambda: "abc")
    store = WorkStore(tmp_path)
    store.write_json("inventory.json", [GRES])
    ctl = Controller(
        store,
        Settings(
            datasets=["benchmark"],
            sites=["nancy"],
            gpu_models=["l40s"],
            max_jobs_total=max_jobs,
            max_jobs_per_site=per_site,
        ),
        log=lambda m: None,
        remote=DirRemote(tmp_path / "bucket"),
    )
    for i in range(n_chunks):
        cid = f"c{i}"
        table = pa.table(
            {"text_sha256": [f"{cid}s"], "text": ["t"], "input_ids": [[1]]}, schema=CHUNK
        )
        store.write_chunk(cid, table)
        store.append_jsonl(
            f"plans/benchmark/{ctl.fp}/chunks.jsonl",
            [{"chunk_id": cid, "order": [0, 0], "size": 3000}],
        )
    return ctl, fake


def snapshot(ctl):
    return sorted((a.id, a.state, a.job_id, tuple(a.chunks), a.name) for a in ctl.ledger())


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    free_nodes=st.integers(0, 4),
    n_chunks=st.integers(0, 6),
    max_jobs=st.integers(1, 6),
    per_site=st.integers(1, 4),
)
def test_a_second_cycle_without_changes_submits_nothing_and_keeps_assignments(
    tmp_path_factory, free_nodes, n_chunks, max_jobs, per_site
):
    with pytest.MonkeyPatch.context() as mp:
        tmp = tmp_path_factory.mktemp("ctl")
        ctl, fake = build(
            tmp, mp, free_nodes=free_nodes, n_chunks=n_chunks, max_jobs=max_jobs, per_site=per_site
        )
        first = ctl.cycle(NOW)
        before, jobs = snapshot(ctl), len(fake.submitted)
        assert len(first.submitted) == jobs <= min(max_jobs, per_site)
        second = ctl.cycle(NOW)
        assert second.submitted == []
        assert len(fake.submitted) == jobs
        assert snapshot(ctl) == before
        taken = [c for a in ctl.live() for c in a.chunks]
        assert len(taken) == len(set(taken))  # no chunk is assigned to two live jobs


def test_the_property_is_not_vacuous(tmp_path, monkeypatch):
    ctl, fake = build(tmp_path, monkeypatch, free_nodes=3, n_chunks=5, max_jobs=6, per_site=4)
    assert len(ctl.cycle(NOW).submitted) > 0
    assert len(ctl.cycle(NOW).submitted) == 0
