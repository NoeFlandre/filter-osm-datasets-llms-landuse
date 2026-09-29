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


def test_card_counts_cover_files_published_by_earlier_runs(tmp_path, monkeypatch):
    """Regression: the card showed only the last run's new files (638,229 of 1,057,002 rows)."""
    uploads = fake_hub(monkeypatch, tmp_path)
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    generate_all(store)
    pub.publish(store, WEBSITE, "rev")
    published = store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet")
    total_rows = pq.read_table(published).num_rows
    # Simulate a dataset published by older code: no stats ledger, no card marker.
    store.path(f"published/{WEBSITE}.stats.jsonl").unlink()
    store.path(f"published/{WEBSITE}.card.sha256").unlink()
    gen_file = next(store.path("publish").glob(f"{WEBSITE}-gen-*/generations/*/*.parquet"))
    cards: list[str] = []
    monkeypatch.setattr(
        hub, "open_file", lambda repo, p: published if p.startswith("labels/") else gen_file
    )
    monkeypatch.setattr(
        hub,
        "upload",
        lambda repo, files, msg: cards.extend(
            src.read_text() for src, dst in files if dst == "README.md"
        ),
    )
    pub.publish(store, WEBSITE, "rev")
    assert f"**total** | **{total_rows:,}**" in cards[-1]
    assert uploads  # the earlier run did upload the labels


def test_a_text_shared_by_two_files_is_uploaded_to_generations_once(tmp_path, monkeypatch):
    """Regression: texts repeated across input files were re-uploaded in every later batch
    (465,986 generation rows for 461,463 unique texts), which multiplies rows on the join."""
    from landuse_filter.application import assemble

    uploads = fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hub, "list_files", lambda repo, rev: ["polygons/a.parquet", "polygons/b.parquet"]
    )
    shutil.copy(INPUTS / "website.parquet", tmp_path / "hubcache" / "polygons" / "b.parquet")
    monkeypatch.setattr(
        hub,
        "download_all",
        lambda repo, rev, paths: [(tmp_path / "hubcache" / p, p) for p in paths],
    )
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet", "polygons/b.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")  # both files carry the same texts
    generate_all(store)
    real = assemble.build_labels

    def only_a(dataset, path, *args, **kwargs):
        if path == "polygons/b.parquet":
            raise assemble.MissingGenerationError(path)
        return real(dataset, path, *args, **kwargs)

    monkeypatch.setattr(pub, "build_labels", only_a)
    pub.publish(store, WEBSITE, "rev")  # run 1: only file a
    monkeypatch.setattr(pub, "build_labels", real)
    pub.publish(store, WEBSITE, "rev")  # run 2: file b, whose texts a already published
    generation_files = [p for batch in uploads for p in batch if p.startswith("generations/")]
    rows = sum(
        pq.read_table(f).num_rows
        for f in store.path("publish").glob(f"{WEBSITE}-gen-*/generations/*/*.parquet")
    )
    unique = len(
        {
            r["text_sha256"]
            for f in store.path("publish").glob(f"{WEBSITE}-gen-*/generations/*/*.parquet")
            for r in pq.read_table(f).to_pylist()
        }
    )
    assert generation_files
    assert rows == unique


def test_description_publish_uploads_the_yes_share_map_with_computed_figures(tmp_path, monkeypatch):
    import pyarrow as pa

    from landuse_filter.adapters.readers import DESCRIPTION, read_description

    uploads: dict[str, Path] = {}
    monkeypatch.setattr(hub, "ensure_dataset", lambda repo: None)
    monkeypatch.setattr(hub, "remote_files", lambda repo: set())
    monkeypatch.setattr(hub, "list_files", lambda repo, rev: ["language-v1/data/a.parquet"])
    language = INPUTS / "description.parquet"
    monkeypatch.setattr(
        hub, "download_all", lambda repo, rev, paths: [(language, p) for p in paths]
    )
    monkeypatch.setattr(
        hub, "upload", lambda repo, files, msg: uploads.update({d: s for s, d in files})
    )
    ids = pq.read_table(language, columns=["osm_type", "osm_id"]).to_pylist()
    polygons = tmp_path / "polygons.parquet"
    pq.write_table(
        pa.table(
            {
                "osm_type": [r["osm_type"] for r in ids],
                "osm_id": [r["osm_id"] for r in ids],
                "bbox_min_x": [2.0 + i for i in range(len(ids))],
                "bbox_min_y": [48.0] * len(ids),
                "bbox_max_x": [2.1 + i for i in range(len(ids))],
                "bbox_max_y": [48.1] * len(ids),
            }
        ),
        polygons,
    )
    monkeypatch.setattr(
        hub,
        "open_file",
        lambda repo, path, revision=None: polygons if path.startswith("data/") else language,
    )
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, DESCRIPTION, config.GENERATION_FP)
    planner.register(["language-v1/data/a.parquet"])
    planner.scan(lambda _: language)
    rows = [
        Generation(r.text_sha256, "x</think>yes", 5, 3, "stop", False, None, None, None, None)
        for r in read_description(language, "language-v1/data/a.parquet")
        if not r.unsplit
    ]
    prov = {name: "p" for name, _ in PROVENANCE}
    store.write_part(
        config.GENERATION_FP,
        "c",
        generation_table(list({g.text_sha256: g for g in rows}.values()), prov),
    )
    report = pub.publish(store, DESCRIPTION, "rev")
    assert report.labelled_files == 1
    assert uploads["assets/yes_share_map.png"].read_bytes()[:4] == b"\x89PNG"
    card = uploads["README.md"].read_text()
    yes = report.decisions.get("yes", 0) + report.decisions.get("no", 0)
    assert f"{yes} of {yes} `yes`/`no` sentences (100.0%) are placed" in card
