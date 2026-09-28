from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import (
    UnknownStatusError,
    read_description,
    read_website,
    read_wiki,
)

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def test_description_covers_every_sentence_and_unsplit_text():
    rows = pq.read_table(INPUTS / "description.parquet").to_pylist()
    refs = list(read_description(INPUTS / "description.parquet", "language-v1/data/x.parquet"))
    split = [r for r in rows if r["split_status"] == "split"]
    assert sum(not r.unsplit for r in refs) == sum(len(r["sentences"]) for r in split)
    assert sum(r.unsplit for r in refs) == len(rows) - len(split)
    assert len({r.label_id for r in refs}) == len(refs)


def test_website_reads_both_fields_and_marks_unsplit_pages():
    rows = pq.read_table(INPUTS / "website.parquet").to_pylist()
    refs = list(read_website(INPUTS / "website.parquet", "polygons/x.parquet"))
    expected = sum(
        len(r[f"{f}_sentences"] or [])
        if r[f"{f}_sentence_status"] == "success"
        else int(r[f"{f}_sentence_status"] == "unsupported_language")
        for r in rows
        for f in ("website", "contact_website")
    )
    assert len(refs) == expected
    assert {dict(r.locator)["field"] for r in refs} == {"website", "contact_website"}
    assert any(r.unsplit for r in refs)
    assert len({r.label_id for r in refs}) == len(refs)


def test_wiki_rows_are_sentences():
    refs = list(read_wiki(INPUTS / "wikipedia.parquet", "wikipedia/sentences/x.parquet"))
    rows = pq.read_table(INPUTS / "wikipedia.parquet").to_pylist()
    assert len(refs) == len(rows)
    assert (
        sum(r.unsplit for r in refs)
        == sum(r["segmentation_status"] == "unsupported_language" for r in rows)
        > 0
    )


def test_unknown_status_fails_loudly(tmp_path):
    path = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["a"],
                "language": ["en"],
                "text": ["t"],
                "segmentation_status": ["brand_new_status"],
            }
        ),
        path,
    )
    with pytest.raises(UnknownStatusError, match="brand_new_status"):
        list(read_wiki(path, "x"))
