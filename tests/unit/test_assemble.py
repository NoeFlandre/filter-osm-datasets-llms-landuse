from pathlib import Path

import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import WEBSITE, read_website
from landuse_filter.application.assemble import (
    MissingGenerationError,
    Resolved,
    Stamp,
    build_labels,
    label_rows,
)

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def refs():
    return list(read_website(INPUTS / "website.parquet", "polygons/x.parquet"))


def test_every_position_gets_exactly_one_decision(tmp_path):
    rs = refs()
    resolved = {r.text_sha256: Resolved("yes", "exact", None) for r in rs if not r.unsplit}
    n = build_labels(
        WEBSITE,
        "polygons/x.parquet",
        INPUTS / "website.parquet",
        resolved=resolved,
        stamp=Stamp("fp", "rev"),
        out=tmp_path,
    )
    table = pq.read_table(tmp_path / "labels" / "polygons" / "x.parquet").to_pylist()
    assert n == len(rs) == len(table)
    assert len({r["label_id"] for r in table}) == n
    skipped = [r for r in table if r["decision"] == "skipped_unsplit"]
    assert skipped
    assert all(r["generation_id"] is None for r in skipped)
    assert all(r["generation_id"] for r in table if r["decision"] != "skipped_unsplit")
    assert {"polygon_id", "field", "sentence_index"} <= table[0].keys()


def test_unresolved_text_blocks_the_file():
    with pytest.raises(MissingGenerationError):
        label_rows(refs(), {}, "fp", "rev")


def test_failed_is_kept_with_reason():
    rs = [r for r in refs() if not r.unsplit][:1]
    rows = label_rows(rs, {rs[0].text_sha256: Resolved("failed", None, "truncated")}, "fp", "rev")
    assert (rows[0]["decision"], rows[0]["failure_reason"]) == ("failed", "truncated")


def test_viewer_table_holds_only_what_a_reader_cares_about(tmp_path):
    from landuse_filter.application.assemble import VIEWER_COLUMNS, build_viewer

    rs = refs()
    resolved = {r.text_sha256: Resolved("no", "exact", None) for r in rs if not r.unsplit}
    n = build_viewer(
        WEBSITE, "polygons/x.parquet", INPUTS / "website.parquet", resolved=resolved, out=tmp_path
    )
    table = pq.read_table(tmp_path / "viewer" / "polygons" / "x.parquet")
    assert table.column_names == VIEWER_COLUMNS == ["sentence", "label", "language", "region"]
    assert table.num_rows == n == len(rs)
    rows = table.to_pylist()
    by_text = {r["sentence"]: r for r in rows}
    for ref in rs:
        assert by_text[ref.text]["label"] == ("skipped_unsplit" if ref.unsplit else "no")
        assert by_text[ref.text]["language"] == ref.language
    assert {r["region"] for r in rows} == {"x"}  # the input file's stem


def test_build_labels_writes_the_viewer_table_too(tmp_path):
    rs = refs()
    resolved = {r.text_sha256: Resolved("yes", "exact", None) for r in rs if not r.unsplit}
    build_labels(
        WEBSITE,
        "polygons/x.parquet",
        INPUTS / "website.parquet",
        resolved=resolved,
        stamp=Stamp("fp", "rev"),
        out=tmp_path,
    )
    viewer = pq.read_table(tmp_path / "viewer" / "polygons" / "x.parquet")
    labels = pq.read_table(tmp_path / "labels" / "polygons" / "x.parquet")
    assert viewer.num_rows == labels.num_rows
    assert viewer.column("label").to_pylist() == labels.column("decision").to_pylist()


def test_unresolved_texts_are_pending_when_the_file_is_published_partially():
    rs = refs()
    some = {r.text_sha256 for r in rs if not r.unsplit}
    resolved = {sha: Resolved("yes", "exact", None) for sha in list(some)[:10]}
    rows = label_rows(rs, resolved, "fp", "rev", allow_pending=True)
    pending = [r for r in rows if r["decision"] == "pending"]
    assert len(rows) == len(rs)
    assert pending
    assert all(r["generation_id"] is None and r["failure_reason"] is None for r in pending)
    assert {r["decision"] for r in rows} == {"yes", "pending", "skipped_unsplit"}


def test_partial_build_shows_pending_in_the_labels_and_the_viewer(tmp_path):
    rs = refs()
    shas = [r.text_sha256 for r in rs if not r.unsplit]
    resolved = {sha: Resolved("no", "exact", None) for sha in shas[:5]}
    build_labels(
        WEBSITE,
        "polygons/x.parquet",
        INPUTS / "website.parquet",
        resolved=resolved,
        stamp=Stamp("fp", "rev"),
        out=tmp_path,
        allow_pending=True,
    )
    viewer = pq.read_table(tmp_path / "viewer" / "polygons" / "x.parquet").to_pylist()
    assert "pending" in {r["label"] for r in viewer}
    assert "no" in {r["label"] for r in viewer}
