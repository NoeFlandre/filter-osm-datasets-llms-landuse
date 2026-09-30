"""Location of texts and labelled rows: website/wiki planning locators, registry, cell binning."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import DESCRIPTION, WEBSITE, WIKI, read_website
from landuse_filter.application import geo, locate
from landuse_filter.application.datasets import SPECS
from landuse_filter.application.locators import wiki_planning
from landuse_filter.domain.hashing import sha256_text
from landuse_filter.domain.sentences import SentenceRef

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def cell_of(lon: float, lat: float) -> str:
    return f"{lat:g}/{lon:g}"


def write(path: Path, columns: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def wiki_world(tmp_path: Path, rows, links, polygons):
    """Region ``fr``: rows (sentence, document, text, status), links, polygons (id, lat, lon)."""
    served = {
        "polygon_document_links/fr.parquet": write(
            tmp_path / "links.parquet",
            {"polygon_id": [p for p, _ in links], "document_id": [d for _, d in links]},
        ),
        "polygons/fr.parquet": write(
            tmp_path / "polygons.parquet",
            {
                "polygon_id": [p[0] for p in polygons],
                "lat": [p[1] for p in polygons],
                "lon": [p[2] for p in polygons],
            },
        ),
    }
    local = write(
        tmp_path / "sentences.parquet",
        {
            "sentence_id": [r[0] for r in rows],
            "document_id": [r[1] for r in rows],
            "language": ["en"] * len(rows),
            "text": [r[2] for r in rows],
            "segmentation_status": [r[3] for r in rows],
        },
    )
    locator = wiki_planning(lambda path: served[path], cell_of)
    return list(locator(WIKI, "wikipedia/sentences/fr.parquet", local))


def test_website_locator_places_split_sentences_at_their_polygon():
    locator = locate.locator(WEBSITE, lambda _: INPUTS, cell_of)
    pairs = list(locator(WEBSITE, "polygons/a.parquet", INPUTS / "website.parquet"))
    refs = list(read_website(INPUTS / "website.parquet", "polygons/a.parquet"))
    split = [r for r in refs if not r.unsplit]
    assert len(pairs) == len(split) > 0
    assert {s for s, _ in pairs} == {r.text_sha256 for r in split}
    assert all("/" in cell for _, cell in pairs)


def test_website_text_cells_skips_unlocated_polygons_and_unsplit_texts():
    def ref(pid, text, unsplit=False):
        locator = (("polygon_id", pid), ("field", "website"))
        return SentenceRef(WEBSITE, "f", locator, text, "en", unsplit)

    polygons = pa.table({"polygon_id": ["a", "b"], "lat": [1.0, None], "lon": [2.0, 3.0]})
    refs = [ref("a", "t1"), ref("a", "t2", unsplit=True), ref("b", "t3"), ref("zzz", "t4")]
    out = list(geo.website_text_cells(refs, polygons, cell_of))
    assert out == [(refs[0].text_sha256, "1/2")]


def test_locate_refuses_a_dataset_without_coordinates():
    with pytest.raises(ValueError, match="no coordinates"):
        locate.locator(DESCRIPTION, lambda _: INPUTS, cell_of)


def test_wiki_first_polygon_is_the_smallest_id_with_coordinates_whatever_the_row_order(tmp_path):
    rows = [("s1", "d1", "hello", "split")]
    links = [("p9", "d1"), ("p2", "d1"), ("p5", "d1")]  # p2 has no coordinates: p5 is first
    polygons = [("p9", 9.0, 9.0), ("p2", None, None), ("p5", 5.0, 5.0)]
    out = wiki_world(tmp_path, rows, links, polygons)
    assert [c for _, c in out] == ["5/5"]
    assert wiki_world(tmp_path, rows, links[::-1], polygons[::-1]) == out


def test_wiki_skips_unsplit_sentences_and_documents_without_polygons(tmp_path):
    rows = [
        ("s1", "d1", "kept", "split"),
        ("s2", "d1", "unsplit", "unsupported_language"),
        ("s3", "d2", "no links", "split"),
        ("s4", "d3", "linked to a polygon without coordinates", "split"),
    ]
    links = [("p1", "d1"), ("p2", "d3")]
    polygons = [("p1", 1.0, 2.0), ("p2", None, None)]
    assert wiki_world(tmp_path, rows, links, polygons) == [(sha256_text("kept"), "1/2")]


def test_wiki_repeated_texts_are_reported_at_each_sentence_location(tmp_path):
    rows = [("s1", "d1", "same", "split"), ("s2", "d2", "same", "split")]
    links = [("p1", "d1"), ("p2", "d2")]
    polygons = [("p1", 1.0, 1.0), ("p2", 2.0, 2.0)]
    out = wiki_world(tmp_path, rows, links, polygons)
    assert out == [(sha256_text("same"), "1/1"), (sha256_text("same"), "2/2")]  # first one wins


def test_the_planner_keeps_the_first_location_of_a_repeated_text(tmp_path):
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.plan import Planner

    rows = [("s1", "d1", "same", "split"), ("s2", "d2", "same", "split")]
    pairs = wiki_world(
        tmp_path, rows, [("p1", "d1"), ("p2", "d2")], [("p1", 1.0, 1.0), ("p2", 2.0, 2.0)]
    )
    planner = Planner(WorkStore(tmp_path / "work"), WIKI, "fp")
    planner.register(["wikipedia/sentences/fr.parquet"])
    planner.scan(lambda _: tmp_path / "sentences.parquet")
    planner.set_cells(pairs)
    cells = planner.db.execute("SELECT sha, cell FROM texts").fetchall()
    assert cells == [(sha256_text("same"), "1/1")]


def test_wiki_cells_bins_labelled_rows_and_counts_unplaced():
    documents = pa.table(
        {"sentence_id": ["a", "b", "c", "d"], "document_id": ["d1", "d1", "d2", "d3"]}
    )
    links = pa.table({"polygon_id": ["p1", "p1"], "document_id": ["d1", "d2"]})
    polygons = pa.table({"polygon_id": ["p1"], "lat": [1.0], "lon": [2.0]})
    labels = pa.table(
        {"sentence_id": ["a", "b", "c", "d", "a"], "decision": ["yes", "no", "no", "yes", "failed"]}
    )
    where = geo.wiki_cells(labels, documents, links, polygons, cell_of)
    assert where.cells == {"1/2": [1, 2]}
    assert (where.labelled, where.located) == (4, 3)


def test_bin_ignores_non_binary_decisions_and_sorts_cells():
    where = geo._bin([("yes", "b"), ("no", "a"), ("failed", "a"), ("pending", None), ("yes", None)])
    assert where.cells == {"a": [0, 1], "b": [1, 0]}
    assert (where.labelled, where.located) == (3, 2)
    assert list(where.cells) == ["a", "b"]


def test_website_cells_places_labels_at_polygon_coordinates():
    labels = pa.table({"polygon_id": ["a", "b", "c"], "decision": ["yes", "no", "yes"]})
    polygons = pa.table({"polygon_id": ["a", "b"], "lat": [1.0, None], "lon": [2.0, 3.0]})
    where = geo.website_cells(labels, polygons, cell_of)
    assert where.cells == {"1/2": [1, 0]}
    assert (where.labelled, where.located) == (3, 1)


def test_description_cells_uses_the_bounding_box_centre():
    labels = pa.table({"description_identity": ["i1", "i2"], "decision": ["yes", "no"]})
    language = pa.table(
        {"description_identity": ["i1", "i2"], "osm_type": ["way", "way"], "osm_id": [1, 2]}
    )
    polygons = pa.table(
        {
            "osm_type": ["way", "way"],
            "osm_id": [1, 2],
            "bbox_min_x": [0.0, None],
            "bbox_min_y": [0.0, 0.0],
            "bbox_max_x": [2.0, 2.0],
            "bbox_max_y": [4.0, 4.0],
        }
    )
    where = geo.description_cells(labels, language, polygons, cell_of)
    assert sum(sum(v) for v in where.cells.values()) == 1
    assert (where.labelled, where.located) == (2, 1)


def test_registry_declares_join_keys_and_a_source_per_dataset():
    assert set(SPECS) == {DESCRIPTION, WIKI, WEBSITE}
    for spec in SPECS.values():
        assert spec.join_keys
        for key, _ in spec.join_keys:
            assert key in spec.card_keys
    assert SPECS[WIKI].text_license
    assert not SPECS[WEBSITE].text_license
