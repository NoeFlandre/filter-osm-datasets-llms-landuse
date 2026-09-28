import shutil
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters import hub
from landuse_filter.adapters.readers import WEBSITE, read_website
from landuse_filter.adapters.schema import PROVENANCE, generation_table
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import publish as pub
from landuse_filter.application.plan import Planner
from landuse_filter.domain.records import Generation

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def fake_hub(monkeypatch, tmp_path):
    uploads: list[list[str]] = []
    local = tmp_path / "hubcache" / "polygons" / "a.parquet"
    local.parent.mkdir(parents=True)
    shutil.copy(INPUTS / "website.parquet", local)
    monkeypatch.setattr(hub, "ensure_dataset", lambda repo: None)
    monkeypatch.setattr(hub, "remote_files", lambda repo: set())
    monkeypatch.setattr(hub, "list_files", lambda repo, rev: ["polygons/a.parquet", "stats.json"])
    monkeypatch.setattr(hub, "download_all", lambda repo, rev, paths: [(local, p) for p in paths])
    monkeypatch.setattr(
        hub, "upload", lambda repo, files, msg: uploads.append([d for _, d in files])
    )
    return uploads


def generate_all(store):
    refs = [
        r for r in read_website(INPUTS / "website.parquet", "polygons/a.parquet") if not r.unsplit
    ]
    rows = [
        Generation(r.text_sha256, "x</think>yes", 5, 3, "stop", False, None, None, None, None)
        for r in refs
    ]
    prov = {name: "p" for name, _ in PROVENANCE}
    store.write_part(
        config.GENERATION_FP,
        "c",
        generation_table(list({g.text_sha256: g for g in rows}.values()), prov),
    )


def test_publishes_mirror_labels_generations_once(tmp_path, monkeypatch):
    uploads = fake_hub(monkeypatch, tmp_path)
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    first = pub.publish(store, WEBSITE, "rev")
    assert first.new_files == 0  # nothing generated yet: only the mirror goes up
    assert uploads == [["polygons/a.parquet", "stats.json"]]
    generate_all(store)
    second = pub.publish(store, WEBSITE, "rev")
    assert second.labelled_files == second.total_files == 1
    flat = [p for batch in uploads for p in batch]
    assert "labels/polygons/a.parquet" in flat
    assert any(p.startswith("generations/") for p in flat)
    assert "README.md" in flat
    labels = pq.read_table(store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet")).to_pylist()
    assert {r["decision"] for r in labels} <= {"yes", "skipped_unsplit"}
    before = len(uploads)
    assert pub.publish(store, WEBSITE, "rev").new_files == 0  # idempotent
    assert len(uploads) == before
