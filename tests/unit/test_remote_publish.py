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


def _published_bucket(tmp_path):
    """A bucket holding the ledgers of a finished full publish, and the README it uploaded."""
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
    remote_publish.run_publish(
        remote, WorkStore(tmp_path / "publish-node"), WEBSITE, "rev", config.GENERATION_FP, hub=hub
    )
    readme = hub.sources["README.md"].read_bytes()
    uploaded = dict(hub.sources)

    def opened(_repo, path, _revision):
        return Path(uploaded[path]) if path in uploaded else hub.local

    hub.opened = opened
    hub.uploads.clear()
    hub.sources.clear()
    return hub, remote, readme


def test_card_only_renders_the_same_card_without_planner_index_or_parts(tmp_path):
    hub, remote, readme = _published_bucket(tmp_path)
    for name in ("stats.jsonl", "card.sha256"):
        (tmp_path / "bucket" / "published" / f"{WEBSITE}.{name}").unlink()  # recount, re-upload
    for p in remote.ls("index/") + remote.ls(f"parts/{config.GENERATION_FP}/"):
        (tmp_path / "bucket" / p).unlink()  # card-only needs neither
    report = remote_publish.run_card_only(
        remote, WorkStore(tmp_path / "card-node"), WEBSITE, "rev", hub=hub
    )
    assert report.labelled_files == report.total_files == 1
    assert hub.sources["README.md"].read_bytes() == readme
    assert remote.ls(f"published/{WEBSITE}.stats.jsonl")  # the recount was saved to the bucket
    assert [p for batch in hub.uploads for p in batch if p.startswith("labels/")] == []


def test_card_only_with_only_a_mirror_leaves_the_card_alone(tmp_path):
    hub = fake_hub(tmp_path)
    remote = DirRemote(tmp_path / "bucket")
    seed = WorkStore(tmp_path / "seed")
    seed.append_jsonl(f"published/{WEBSITE}.jsonl", [{"path": "mirror:rev"}])
    remote.put([(seed.path(f"published/{WEBSITE}.jsonl"), f"published/{WEBSITE}.jsonl")])
    report = remote_publish.run_card_only(
        remote, WorkStore(tmp_path / "card-node"), WEBSITE, "rev", hub=hub
    )
    assert (report.labelled_files, report.total_files) == (0, 1)
    assert hub.uploads == []


def test_counting_skips_failing_files_and_never_aborts(tmp_path):
    from landuse_filter.application import published_stats

    store = WorkStore(tmp_path / "s")

    def opener(path):
        raise OSError("hub hiccup")

    records = published_stats.complete(
        store, "d", {"labels/bad.parquet", "generations/x.parquet"}, opener
    )
    assert records == []
    assert store.read_jsonl(published_stats.ledger("d")) == []
