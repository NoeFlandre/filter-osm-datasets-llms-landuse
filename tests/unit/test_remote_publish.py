from pathlib import Path

import pytest

from landuse_filter import config
from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import remote_publish
from landuse_filter.application.plan import Planner
from tests.unit.test_publish import fake_hub, generate_all

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def test_publish_job_restores_index_and_parts_from_the_bucket(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    planning = WorkStore(tmp_path / "planning-node")
    planner = Planner(planning, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    planner.db.commit()
    generate_all(planning)
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(planning.path(f"index/{WEBSITE}.sqlite"), f"index/{WEBSITE}.sqlite")])
    for p in planning.part_paths(config.GENERATION_FP):
        remote.put([(p, str(p.relative_to(planning.root)))])
    report = remote_publish.run_publish(
        remote, WorkStore(tmp_path / "publish-node"), WEBSITE, "rev", config.GENERATION_FP, hub=hub
    )
    assert report.labelled_files == 1
    assert "labels/polygons/a.parquet" in [p for batch in hub.uploads for p in batch]
    assert remote.ls(f"published/{WEBSITE}.jsonl")


def test_publish_job_requires_planning(tmp_path):
    with pytest.raises(FileNotFoundError, match="planning"):
        remote_publish.run_publish(
            DirRemote(tmp_path / "b"), WorkStore(tmp_path / "s"), WEBSITE, "rev", "fp"
        )


def test_reset_card_cache_drops_generation_counts_and_the_card_hash(tmp_path):
    from landuse_filter.adapters.remote import DirRemote
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.remote_publish import reset_card_cache

    remote = DirRemote(tmp_path / "bucket")
    source = WorkStore(tmp_path / "src")
    source.append_jsonl(
        "published/d.stats.jsonl",
        [
            {"path": "labels/a.parquet", "decisions": {"yes": 1}},
            {"path": "generations/x", "rows": 3},
        ],
    )
    remote.put([(source.path("published/d.stats.jsonl"), "published/d.stats.jsonl")])
    scratch = WorkStore(tmp_path / "scratch")
    reset_card_cache(remote, scratch, "d")
    kept = scratch.read_jsonl("published/d.stats.jsonl")
    assert [r["path"] for r in kept] == ["labels/a.parquet"]
    assert "published/d.card.sha256" in remote.ls("published/")


def test_publish_job_saves_the_partial_ledger_to_the_bucket_while_it_works(tmp_path, monkeypatch):
    """Regression: a job stopped at its walltime lost the node-local ledger, so the next job
    rebuilt and re-uploaded every partial file again."""
    from landuse_filter.application import publish as pub
    from tests.unit.test_publish import generate_some

    hub = fake_hub(tmp_path)
    monkeypatch.setattr(pub, "PARTIAL_FLUSH", 1)
    planning = WorkStore(tmp_path / "planning-node")
    planner = Planner(planning, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    planner.db.commit()
    generate_some(planning, "p1", 0, 20)
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(planning.path(f"index/{WEBSITE}.sqlite"), f"index/{WEBSITE}.sqlite")])
    for p in planning.part_paths(config.GENERATION_FP):
        remote.put([(p, str(p.relative_to(planning.root)))])

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pub, "_refresh_card", stopped)
    with pytest.raises(KeyboardInterrupt):
        remote_publish.run_publish(
            remote,
            WorkStore(tmp_path / "publish-node"),
            WEBSITE,
            "rev",
            config.GENERATION_FP,
            hub=hub,
        )
    assert remote.ls(f"published/{WEBSITE}.partial.jsonl")


def test_a_stopped_publish_job_still_saves_the_ledgers_to_the_bucket(tmp_path, monkeypatch):
    from landuse_filter.application import publish as pub

    monkeypatch.setattr(pub, "BATCH", 1)
    hub = fake_hub(tmp_path)
    hub.files = ["polygons/a.parquet", "stats.json"]
    planning = WorkStore(tmp_path / "planning-node")
    planner = Planner(planning, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    planner.db.commit()
    generate_all(planning)
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(planning.path(f"index/{WEBSITE}.sqlite"), f"index/{WEBSITE}.sqlite")])
    for p in planning.part_paths(config.GENERATION_FP):
        remote.put([(p, str(p.relative_to(planning.root)))])
    calls = iter([None, "signal SIGTERM"])  # one mirror batch, then the stop
    report = remote_publish.run_publish(
        remote,
        WorkStore(tmp_path / "publish-node"),
        WEBSITE,
        "rev",
        config.GENERATION_FP,
        hub=hub,
        should_stop=lambda: next(calls, "signal SIGTERM"),
    )
    assert report.stopped == "signal SIGTERM"
    assert remote.ls(f"published/{WEBSITE}.mirror.jsonl")  # progress survives the node
