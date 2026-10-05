import json
from pathlib import Path

import pytest

from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.staging import Transport
from landuse_filter.application.work_progress import WorkProgress


class CountingRemote(DirRemote):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.listed: list[str] = []
        self.failing = False

    def ls(self, prefix: str) -> list[str]:
        if self.failing:
            raise OSError("429")
        self.listed.append(prefix)
        return super().ls(prefix)


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def upload(remote: DirRemote, chunk: str, part: str, shas: list[str]) -> None:
    path = remote.root / "parts" / "fp" / chunk / f"{part}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text_sha256s": shas}))


def setup(tmp_path: Path, clock: Clock | None = None):
    store = WorkStore(tmp_path / "work")
    remote = CountingRemote(tmp_path / "bucket")
    progress = WorkProgress(
        store,
        plan_fp="fp",
        work_fp="fp",
        datasets=["a"],
        complete_log="complete.jsonl",
        log=lambda m: None,
    )
    clock = clock or Clock()
    transport = Transport(
        store,
        remote,
        "b",
        lambda m: None,
        reconcile_interval=3600,
        live_interval=900,
        clock=clock,
    )
    return transport, progress, remote, clock


def test_first_pull_lists_everything_then_only_ended_chunks(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    upload(remote, "c1", "p1", ["a"])
    upload(remote, "c2", "p1", ["b"])
    transport.pull(progress, {"c1"})
    assert remote.listed == ["parts/fp/", "jobs/"]
    assert progress.index("fp").count("c2") == 1
    remote.listed.clear()
    upload(remote, "c1", "p2", ["c"])
    clock.now = 180
    transport.pull(progress, {"c1"})  # c1 live but its live interval is not due
    assert remote.listed == []
    transport.pull(progress, set())  # c1 ended: one final listing of c1 and the job summaries
    assert remote.listed == ["parts/fp/c1/", "jobs/"]
    assert progress.index("fp").count("c1") == 2
    remote.listed.clear()
    transport.pull(progress, set())
    assert remote.listed == []


def test_live_chunks_are_refreshed_each_live_interval(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    transport.pull(progress, {"c1"})
    remote.listed.clear()
    upload(remote, "c1", "p1", ["a"])
    clock.now = 900
    transport.pull(progress, {"c1"})
    assert remote.listed == ["parts/fp/c1/"]
    assert progress.index("fp").count("c1") == 1


def test_a_full_listing_reconciles_every_interval_and_catches_stray_parts(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    transport.pull(progress, set())
    upload(remote, "orphan", "p1", ["z"])  # uploaded by a job nobody tracks
    clock.now = 3599
    transport.pull(progress, set())
    assert progress.index("fp").count("orphan") == 0
    clock.now = 3600
    transport.pull(progress, set())
    assert progress.index("fp").count("orphan") == 1


def test_unknown_live_chunks_or_a_restart_list_everything(tmp_path):
    transport, progress, remote, _ = setup(tmp_path)
    transport.pull(progress, {"c1"})
    remote.listed.clear()
    transport.pull(progress)
    assert remote.listed == ["parts/fp/", "jobs/"]
    upload(remote, "c9", "p1", ["q"])
    restarted, progress2, remote2, _ = setup(tmp_path)
    restarted.pull(progress2, set())
    assert remote2.listed == ["parts/fp/", "jobs/"]
    assert progress2.index("fp").count("c9") == 1


def test_a_failed_final_listing_is_retried_on_the_next_pull(tmp_path):
    transport, progress, remote, clock = setup(tmp_path)
    transport.pull(progress, {"c1"})
    upload(remote, "c1", "p1", ["a"])
    remote.failing = True
    with pytest.raises(OSError, match="429"):
        transport.pull(progress, set())
    remote.failing = False
    clock.now = 180
    transport.pull(progress, set())
    assert progress.index("fp").count("c1") == 1


def test_an_unseen_local_manifest_is_ingested_not_skipped(tmp_path):
    transport, progress, remote, _ = setup(tmp_path)
    upload(remote, "c1", "p1", ["a"])
    # a download whose ingest was interrupted left the manifest on disk, unseen
    local = progress.store.path("parts/fp/c1/p1.json")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps({"text_sha256s": ["a"]}))
    transport.pull(progress, set())
    assert progress.index("fp").count("c1") == 1
