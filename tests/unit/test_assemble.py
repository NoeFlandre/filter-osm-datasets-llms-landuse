from pathlib import Path

import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import WEBSITE, read_website
from landuse_filter.application.assemble import Missing, Resolved, build_labels, label_rows

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def refs():
    return list(read_website(INPUTS / "website.parquet", "polygons/x.parquet"))


def test_every_position_gets_exactly_one_decision(tmp_path):
    rs = refs()
    resolved = {r.text_sha256: Resolved("yes", "exact", None) for r in rs if not r.unsplit}
    n = build_labels(WEBSITE, "polygons/x.parquet", INPUTS / "website.parquet", resolved, "fp", "rev", tmp_path)
    table = pq.read_table(tmp_path / "labels" / "polygons" / "x.parquet").to_pylist()
    assert n == len(rs) == len(table)
    assert len({r["label_id"] for r in table}) == n
    skipped = [r for r in table if r["decision"] == "skipped_unsplit"]
    assert skipped and all(r["generation_id"] is None for r in skipped)
    assert all(r["generation_id"] for r in table if r["decision"] != "skipped_unsplit")
    assert {"polygon_id", "field", "sentence_index"} <= table[0].keys()


def test_unresolved_text_blocks_the_file():
    with pytest.raises(Missing):
        label_rows(refs(), {}, "fp", "rev")


def test_failed_is_kept_with_reason():
    rs = [r for r in refs() if not r.unsplit][:1]
    rows = label_rows(rs, {rs[0].text_sha256: Resolved("failed", None, "truncated")}, "fp", "rev")
    assert (rows[0]["decision"], rows[0]["failure_reason"]) == ("failed", "truncated")
