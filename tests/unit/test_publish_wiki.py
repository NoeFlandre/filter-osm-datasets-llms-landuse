"""`publish` with the wiki dataset: sentences keyed by sentence_id, a map placed via documents."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from landuse_filter import config
from landuse_filter.adapters.readers import WIKI, read_wiki
from landuse_filter.adapters.schema import PROVENANCE, generation_table
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import publish as pub
from landuse_filter.application.plan import Planner
from landuse_filter.domain.records import Generation
from tests.unit.test_publish import FakeHub

SENTENCES = "wikipedia/sentences/fr.parquet"
N = 12  # sentences s0..s10 are split, s11 is not segmented upstream


@pytest.fixture
def world(tmp_path):
    pytest.importorskip("h3")
    pytest.importorskip("matplotlib")
    served = {
        SENTENCES: tmp_path / "sentences.parquet",
        "polygon_document_links/fr.parquet": tmp_path / "links.parquet",
        "polygons/fr.parquet": tmp_path / "polygons.parquet",
    }
    pq.write_table(
        pa.table(
            {
                "sentence_id": [f"s{i}" for i in range(N)],
                "document_id": ["d1" if i < N - 2 else "d-orphan" for i in range(N)],
                "language": ["en"] * N,
                "text": [f"text {i}" for i in range(N)],
                "segmentation_status": ["split"] * (N - 1) + ["unsupported_language"],
            }
        ),
        served[SENTENCES],
    )
    pq.write_table(
        pa.table({"polygon_id": ["p9", "p1"], "document_id": ["d1", "d1"]}),
        served["polygon_document_links/fr.parquet"],
    )
    pq.write_table(
        pa.table({"polygon_id": ["p9", "p1"], "lat": [-30.0, 48.0], "lon": [150.0, 2.0]}),
        served["polygons/fr.parquet"],
    )
    hub = FakeHub(served[SENTENCES], [SENTENCES])
    hub.opened = lambda repo, path, revision: served[path]
    store = WorkStore(tmp_path / "work")
    planner = Planner(store, WIKI, config.GENERATION_FP)
    planner.register([SENTENCES])
    planner.scan(lambda _: served[SENTENCES])
    return hub, store, served[SENTENCES]


def generate(store, sentences: Path, part: str, start: int, stop: int) -> None:
    refs = [r for r in read_wiki(sentences, SENTENCES) if not r.unsplit]
    shas = sorted({r.text_sha256 for r in refs})[start:stop]
    rows = [
        Generation(s, "x</think>yes", 5, 3, "stop", False, None, None, None, None) for s in shas
    ]
    prov = {name: "p" for name, _ in PROVENANCE}
    store.write_part(config.GENERATION_FP, part, generation_table(rows, prov))


def labels_of(store) -> list[dict]:
    return pq.read_table(store.path(f"publish/{WIKI}/labels/{SENTENCES}")).to_pylist()


def test_complete_wiki_file_publishes_labels_keyed_by_sentence_id_and_a_world_map(world):
    hub, store, sentences = world
    generate(store, sentences, "p1", 0, N)
    report = pub.publish(store, WIKI, "rev", hub=hub)
    assert (report.labelled_files, report.partial_files) == (1, 0)
    flat = [p for batch in hub.uploads for p in batch]
    assert f"labels/{SENTENCES}" in flat
    assert "assets/yes_share_map.png" in flat
    rows = labels_of(store)
    assert [r["sentence_id"] for r in rows] == [f"s{i}" for i in range(N)]
    assert rows[-1]["decision"] == "skipped_unsplit"
    assert all(r["decision"] == "yes" for r in rows[:-1])
    card = hub.sources["README.md"].read_text()
    assert "smallest `polygon_id` with lat/lon" in card  # the wiki wording of the map
    assert "Wikipedia and Wikivoyage text is CC BY-SA 4.0" in card
    # s10 belongs to a document without polygons: 10 of the 11 yes sentences are placed
    assert "10 of 11 `yes`/`no` sentences (90.9%) are placed, in 1 cells" in card
    assert "USING (sentence_id)" in card


def test_partial_wiki_file_is_published_with_pending_rows_and_no_generations(world):
    hub, store, sentences = world
    generate(store, sentences, "p1", 0, 6)
    report = pub.publish(store, WIKI, "rev", hub=hub)
    assert (report.labelled_files, report.partial_files) == (0, 1)
    flat = [p for batch in hub.uploads for p in batch]
    assert f"labels/{SENTENCES}" in flat
    assert not any(p.startswith("generations/") for p in flat)
    decisions = [r["decision"] for r in labels_of(store)]
    assert decisions.count("pending") == N - 1 - 6
    assert decisions.count("skipped_unsplit") == 1
    assert report.decisions["pending"] == N - 1 - 6
    card = hub.sources["README.md"].read_text()
    assert "0 of 1 input files fully labelled, 1 more partially" in card
    assert "| `pending` |" in card
    assert "dataset_status: in_progress" in card
    done = {r["path"] for r in store.read_jsonl(f"published/{WIKI}.jsonl")}
    assert f"labels/{SENTENCES}" not in done  # stays open until the file is complete
    # completing the file replaces the partial version and drops the pending row from the card
    generate(store, sentences, "p2", 6, N)
    final = pub.publish(store, WIKI, "rev", hub=hub)
    assert (final.labelled_files, final.partial_files) == (1, 0)
    assert "pending" not in final.decisions
    assert "| `pending` |" not in hub.sources["README.md"].read_text()
