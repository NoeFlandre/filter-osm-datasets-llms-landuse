import contextlib
import shutil
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter import config
from landuse_filter.adapters.readers import WEBSITE, read_website
from landuse_filter.adapters.schema import PROVENANCE, generation_table
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import publish as pub
from landuse_filter.application.plan import Planner
from landuse_filter.domain.records import Generation

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


class FakeHub:
    """In-memory Hub port: records uploads, serves local files."""

    def __init__(self, local: Path, files: list[str] | None = None) -> None:
        self.local = local
        self.files = files or ["polygons/a.parquet", "stats.json"]
        self.uploads: list[list[str]] = []
        self.sources: dict[str, Path] = {}  # repo path -> local file, last upload wins
        self.opened: Callable[[str, str, str | None], Path] | None = None
        self.downloaded: Callable[[str], Path] | None = None

    def ensure_dataset(self, repo_id):
        pass

    def remote_files(self, repo_id):
        return set()

    def list_files(self, repo_id, revision):
        return self.files

    def download_all(self, repo_id, revision, paths):
        return [((self.downloaded(p) if self.downloaded else self.local), p) for p in paths]

    def open_file(self, repo_id, path, revision=None):
        return self.opened(repo_id, path, revision) if self.opened else self.local

    def upload(self, repo_id, files, message):
        self.uploads.append([d for _, d in files])
        self.sources.update({d: s for s, d in files})


def fake_hub(tmp_path):
    local = tmp_path / "hubcache" / "polygons" / "a.parquet"
    local.parent.mkdir(parents=True)
    shutil.copy(INPUTS / "website.parquet", local)
    return FakeHub(local)


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
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    first = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert first.new_files == 0  # nothing generated yet: only the mirror goes up
    assert uploads == [["polygons/a.parquet", "stats.json"]]
    generate_all(store)
    second = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert second.labelled_files == second.total_files == 1
    flat = [p for batch in uploads for p in batch]
    assert "labels/polygons/a.parquet" in flat
    assert any(p.startswith("generations/") for p in flat)
    assert "README.md" in flat
    assert "assets/yes_share_map.png" in flat  # website rows carry lat/lon: the map is drawn
    assert "viewer/polygons/a.parquet" in flat  # the default dataset-viewer table
    labels = pq.read_table(store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet")).to_pylist()
    assert {r["decision"] for r in labels} <= {"yes", "skipped_unsplit"}
    before = len(uploads)
    assert pub.publish(store, WEBSITE, "rev", hub=hub).new_files == 0  # idempotent
    assert len(uploads) == before


def test_card_counts_cover_files_published_by_earlier_runs(tmp_path, monkeypatch):
    """Regression: the card showed only the last run's new files (638,229 of 1,057,002 rows)."""
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    generate_all(store)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    published = store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet")
    total_rows = pq.read_table(published).num_rows
    # Simulate a dataset published by older code: no stats ledger, no card marker.
    store.path(f"published/{WEBSITE}.stats.jsonl").unlink()
    store.path(f"published/{WEBSITE}.card.sha256").unlink()
    gen_file = next(store.path("publish").glob(f"{WEBSITE}-gen-*/generations/*/*.parquet"))

    def open_file(repo, path, revision=None):
        if path.startswith("labels/"):
            return published
        return gen_file if path.startswith("generations/") else INPUTS / "website.parquet"

    hub.opened = open_file
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert f"**total** | **{total_rows:,}**" in hub.sources["README.md"].read_text()
    assert uploads  # the earlier run did upload the labels


def test_a_text_shared_by_two_files_is_uploaded_to_generations_once(tmp_path, monkeypatch):
    """Regression: texts repeated across input files were re-uploaded in every later batch
    (465,986 generation rows for 461,463 unique texts), which multiplies rows on the join."""
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    hub.files = ["polygons/a.parquet", "polygons/b.parquet"]
    shutil.copy(INPUTS / "website.parquet", tmp_path / "hubcache" / "polygons" / "b.parquet")
    hub.downloaded = lambda p: tmp_path / "hubcache" / p
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet", "polygons/b.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")  # both files carry the same texts
    generate_all(store)
    real = pub._resolved_counts

    def only_a(refs, resolved):
        if len(only_a.calls) % 2:  # file b: pretend it is not fully generated yet
            only_a.calls.append(1)
            return 0, 46, False
        only_a.calls.append(0)
        return real(refs, resolved)

    only_a.calls = []
    monkeypatch.setattr(pub, "_resolved_counts", only_a)
    pub.publish(store, WEBSITE, "rev", hub=hub)  # run 1: only file a
    monkeypatch.setattr(pub, "_resolved_counts", real)
    pub.publish(store, WEBSITE, "rev", hub=hub)  # run 2: file b, whose texts a already published
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

    language = INPUTS / "description.parquet"
    hub = FakeHub(language, ["language-v1/data/a.parquet"])
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
    hub.opened = lambda repo, path, revision: polygons if path.startswith("data/") else language
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
    report = pub.publish(store, DESCRIPTION, "rev", hub=hub)
    assert report.labelled_files == 1
    assert hub.sources["assets/yes_share_map.png"].read_bytes()[:4] == b"\x89PNG"
    card = hub.sources["README.md"].read_text()
    yes = report.decisions.get("yes", 0) + report.decisions.get("no", 0)
    assert f"{yes} of {yes} `yes`/`no` sentences (100.0%) are placed" in card


def test_files_labelled_before_the_viewer_existed_get_their_viewer_table(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    generate_all(store)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    # Simulate a repo published by older code: labels are recorded, the viewer is not.
    ledger = store.path(f"published/{WEBSITE}.jsonl")
    kept = [r for r in store.read_jsonl(f"published/{WEBSITE}.jsonl") if "viewer/" not in r["path"]]
    ledger.unlink()
    store.append_jsonl(f"published/{WEBSITE}.jsonl", kept)
    before = len(uploads)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert ["viewer/polygons/a.parquet"] in uploads[before:]


def generate_some(store, part, start, stop):
    refs = [
        r for r in read_website(INPUTS / "website.parquet", "polygons/a.parquet") if not r.unsplit
    ]
    shas = sorted({r.text_sha256 for r in refs})[start:stop]
    rows = [
        Generation(s, "x</think>yes", 5, 3, "stop", False, None, None, None, None) for s in shas
    ]
    prov = {name: "p" for name, _ in PROVENANCE}
    store.write_part(config.GENERATION_FP, part, generation_table(rows, prov))


def planned(tmp_path):
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WEBSITE, config.GENERATION_FP)
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: INPUTS / "website.parquet")
    return store


def test_a_partly_generated_file_is_published_with_pending_rows(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    flat = [p for batch in uploads for p in batch]
    assert (report.labelled_files, report.partial_files) == (0, 1)
    assert "labels/polygons/a.parquet" in flat
    assert "viewer/polygons/a.parquet" in flat
    assert not any(p.startswith("generations/") for p in flat)  # shipped with the complete file
    labels = pq.read_table(store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet")).to_pylist()
    assert sum(r["decision"] == "pending" for r in labels) > 0
    assert report.decisions["pending"] > 0
    done = {r["path"] for r in store.read_jsonl(f"published/{WEBSITE}.jsonl")}
    assert "labels/polygons/a.parquet" not in done  # a partial file stays open for later runs


def test_partial_files_are_refreshed_only_after_enough_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PARTIAL_STEP", 0.10)  # a 46-sentence file needs a large step
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    generate_some(store, "p2", 20, 22)  # +2 of 46: under the step
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads) == 1
    generate_some(store, "p3", 22, 30)  # +10 in total
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads) == 2


def test_a_file_becoming_complete_replaces_its_partial_version(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    generate_some(store, "p2", 20, 46)
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    flat = [p for batch in uploads for p in batch]
    assert (report.labelled_files, report.partial_files) == (1, 0)
    assert any(p.startswith("generations/") for p in flat)
    assert "pending" not in report.decisions  # the card counts the final file, not the partial
    done = {r["path"] for r in store.read_jsonl(f"published/{WEBSITE}.jsonl")}
    assert "labels/polygons/a.parquet" in done


def test_partial_uploads_are_recorded_as_they_happen(tmp_path, monkeypatch):
    """Regression: a job checkpointed after its last commit lost the whole partial upload
    (the ledger and card were written only at the end) and redid 50 minutes of work."""
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    monkeypatch.setattr(pub, "PARTIAL_FLUSH", 1)
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    real = pub._refresh_card

    def die(*args, **kwargs):
        raise KeyboardInterrupt  # the job is stopped after the commits, before the card

    monkeypatch.setattr(pub, "_refresh_card", die)
    with contextlib.suppress(KeyboardInterrupt):
        pub.publish(store, WEBSITE, "rev", hub=hub)
    monkeypatch.setattr(pub, "_refresh_card", real)
    served = tmp_path / "served.parquet"  # what the stopped job uploaded
    shutil.copy(store.path(f"publish/{WEBSITE}/labels/polygons/a.parquet"), served)
    hub.opened = lambda repo, path, revision: (
        served if path.startswith("labels/") else INPUTS / "website.parquet"
    )
    assert any("labels/polygons/a.parquet" in batch for batch in uploads)
    before = len(uploads)
    pub.publish(store, WEBSITE, "rev", hub=hub)  # the next run skips the partial file
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads[:before]) == 1
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads[before:]) == 0


def test_admitted_gates_keep_only_admitted_gpu_types(tmp_path):
    store = WorkStore(tmp_path)
    store.write_json("gates/admission/a100.json", {"status": "admitted", "gate": {"x": 1}})
    store.write_json("gates/admission/t4.json", {"status": "rejected", "gate": {"x": 2}})
    assert store.admitted_gates() == {"a100": {"x": 1}}
    assert WorkStore(tmp_path / "empty").admitted_gates() == {}


def test_wiki_map_locator_reads_sentences_links_and_polygons_through_the_hub(tmp_path):
    import pyarrow as pa

    from landuse_filter.adapters.readers import WIKI
    from landuse_filter.application.datasets import SPECS

    served = {}
    for name, table in {
        "wikipedia/sentences/fr.parquet": {"sentence_id": ["s1", "s2"], "document_id": ["d", "e"]},
        "polygon_document_links/fr.parquet": {"polygon_id": ["p"], "document_id": ["d"]},
        "polygons/fr.parquet": {"polygon_id": ["p"], "lat": [48.0], "lon": [2.0]},
    }.items():
        served[name] = tmp_path / name.replace("/", "_")
        pq.write_table(pa.table(table), served[name])
    hub = FakeHub(tmp_path)
    hub.opened = lambda repo, path, revision: served[path]
    labels = tmp_path / "labels.parquet"
    pq.write_table(pa.table({"sentence_id": ["s1", "s2"], "decision": ["yes", "yes"]}), labels)
    spec = SPECS[WIKI]
    assert spec.map_locator is not None
    locate = spec.map_locator(spec.source.repo_id, hub, "rev")
    where = locate("labels/wikipedia/sentences/fr.parquet", labels)
    assert (where.labelled, where.located) == (2, 1)  # s2's document has no linked polygon


def test_every_dataset_declares_its_location_capabilities():
    from landuse_filter.adapters.readers import DESCRIPTION, SOURCES, WEBSITE, WIKI
    from landuse_filter.application.datasets import SPECS

    assert set(SPECS) == set(SOURCES)
    assert {d for d, s in SPECS.items() if s.planning_locator} == {WIKI, WEBSITE}
    assert {d for d, s in SPECS.items() if s.map_locator} == {DESCRIPTION, WIKI, WEBSITE}


def test_a_never_published_file_appears_as_soon_as_one_sentence_is_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PARTIAL_STEP", 0.50)  # far above one sentence of 46
    hub = fake_hub(tmp_path)
    uploads = hub.uploads
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 1)
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert report.partial_files == 1
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads) == 1
    viewer = pq.read_table(store.path(f"publish/{WEBSITE}/viewer/polygons/a.parquet"))
    assert "pending" not in viewer.column("label").to_pylist()
    assert 0 < viewer.num_rows < 46
    generate_some(store, "p2", 1, 3)  # published already: the step rule applies again
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert sum("labels/polygons/a.parquet" in batch for batch in uploads) == 1


def test_a_file_with_nothing_resolved_is_not_published(tmp_path):
    hub = fake_hub(tmp_path)
    store = planned(tmp_path)
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert report.partial_files == 0
    assert not any("labels/polygons/a.parquet" in batch for batch in hub.uploads)


def counting_builds(monkeypatch):
    calls = {"labels": 0, "viewer": 0}
    real_labels, real_viewer = pub.build_labels, pub.build_viewer

    def labels(*args, **kwargs):
        calls["labels"] += 1
        return real_labels(*args, **kwargs)

    def viewer(*args, **kwargs):
        calls["viewer"] += 1
        return real_viewer(*args, **kwargs)

    monkeypatch.setattr(pub, "build_labels", labels)
    monkeypatch.setattr(pub, "build_viewer", viewer)
    return calls


def test_a_partial_file_that_cannot_have_changed_builds_no_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "PARTIAL_STEP", 0.50)
    hub = fake_hub(tmp_path)
    store = planned(tmp_path)
    calls = counting_builds(monkeypatch)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert calls["labels"] == 0  # nothing resolved: counted, not built
    generate_some(store, "p1", 0, 5)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert calls["labels"] == 1
    generate_some(store, "p2", 5, 7)  # under the step
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert calls["labels"] == 1
    generate_some(store, "p3", 7, 46)  # complete: still built and published
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert calls["labels"] == 2
    assert report.labelled_files == 1


def test_the_card_is_refreshed_after_each_partial_flush(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    monkeypatch.setattr(pub, "PARTIAL_FLUSH", 1)
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    seen = []
    real = pub._refresh_card

    def spy(*args, **kwargs):
        seen.append(len(hub.uploads))
        return real(*args, **kwargs)

    monkeypatch.setattr(pub, "_refresh_card", spy)
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert seen[0] == 2  # mirror, then the partial commit, before the final refresh
    cards = [b for b in hub.uploads if "README.md" in b]
    assert len(cards) == 1  # unchanged card: the final refresh does not re-upload it
    assert "pending" in hub.sources["README.md"].read_text()


def test_a_failing_card_refresh_does_not_fail_the_job(tmp_path, monkeypatch):
    hub = fake_hub(tmp_path)
    monkeypatch.setattr(pub, "PARTIAL_FLUSH", 1)
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    real = pub._refresh_card
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("hub hiccup")
        return real(*args, **kwargs)

    monkeypatch.setattr(pub, "_refresh_card", flaky)
    report = pub.publish(store, WEBSITE, "rev", hub=hub)
    assert report.partial_files == 1
    assert "pending" in hub.sources["README.md"].read_text()  # the final refresh still ran


def test_mirror_downloads_in_parallel_records_progress_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "BATCH", 2)
    names = [f"f{i}.bin" for i in range(5)]
    hub = FakeHub(tmp_path, names)
    store = WorkStore(tmp_path / "work")
    saved = []
    pub._mirror(
        hub,
        store,
        "d",
        input_repo="in",
        revision="r" * 40,
        repo="out",
        done=set(),
        on_progress=lambda: saved.append(1),
    )
    assert hub.uploads == [names[:2], names[2:4], names[4:]]
    assert len(saved) == 3
    ledger = {r["path"] for r in store.read_jsonl(pub.mirror_ledger("d"))}
    assert ledger == set(names)
    done = {r["path"] for r in store.read_jsonl("published/d.jsonl")}
    assert done == {"mirror:" + "r" * 40}
    # A job stopped after two batches: the restart redoes only the rest, without listing out.
    store2 = WorkStore(tmp_path / "work2")
    store2.append_jsonl(pub.mirror_ledger("d"), [{"path": p, "revision": "r"} for p in names[:4]])
    hub2 = FakeHub(tmp_path, names)
    hub2.remote_files = lambda repo: (_ for _ in ()).throw(AssertionError("listed"))
    pub._mirror(hub2, store2, "d", input_repo="in", revision="r", repo="out", done=set())
    assert hub2.uploads == [names[4:]]


def test_mirror_frees_each_batch_after_its_upload(tmp_path):
    cache = tmp_path / "blobs"
    cache.mkdir()
    (cache / "blob").write_text("x")
    link = tmp_path / "snap.bin"
    link.symlink_to(cache / "blob")
    hub = FakeHub(link, ["a.bin"])
    pub._mirror(
        hub, WorkStore(tmp_path / "w"), "d", input_repo="in", revision="r", repo="out", done=set()
    )
    assert not link.exists()
    assert not (cache / "blob").exists()


def test_the_card_is_refreshed_at_the_start_of_a_run_from_the_ledgers(tmp_path, monkeypatch):
    """Regression: a run that spent its walltime downloading input files (386 for website) before
    any upload left the card stale for hours; the card must match the ledgers before any work."""
    hub = fake_hub(tmp_path)
    store = planned(tmp_path)
    generate_some(store, "p1", 0, 20)
    pub.publish(store, WEBSITE, "rev", hub=hub)  # run 1 publishes the partial file and the card
    store.path(f"published/{WEBSITE}.card.sha256").unlink()  # the Hub card is "stale" now
    order = []
    real_card, real_build = pub._refresh_card, pub._build_files
    monkeypatch.setattr(
        pub, "_refresh_card", lambda *a, **k: (order.append("card"), real_card(*a, **k))[1]
    )
    monkeypatch.setattr(
        pub, "_build_files", lambda run: (order.append("build"), real_build(run))[1]
    )
    pub.publish(store, WEBSITE, "rev", hub=hub)
    assert order[:2] == ["card", "build"]  # the card first, before the slow loop over files
