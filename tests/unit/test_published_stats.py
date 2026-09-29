import pyarrow as pa
import pyarrow.parquet as pq

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import published_stats as ps


def labels_file(tmp_path, name, rows):
    path = tmp_path / name
    pq.write_table(
        pa.table(
            {
                "decision": [d for d, _ in rows],
                "failure_reason": [r for _, r in rows],
            }
        ),
        path,
    )
    return path


def test_label_stats_count_decisions_and_failure_reasons_of_failed_rows_only(tmp_path):
    path = labels_file(
        tmp_path,
        "a.parquet",
        [("yes", None), ("no", None), ("failed", "truncated"), ("failed", "empty"),
         ("skipped_unsplit", "unsplit_upstream")],
    )  # fmt: skip
    stats = ps.file_stats("labels/a.parquet", path)
    assert stats["decisions"] == {"yes": 1, "no": 1, "failed": 2, "skipped_unsplit": 1}
    assert stats["failures"] == {"truncated": 1, "empty": 1}


def test_generation_stats_count_rows_per_gpu_key(tmp_path):
    path = tmp_path / "g.parquet"
    pq.write_table(pa.table({"gpu": ["NVIDIA L40S", "NVIDIA L40S", "A100-SXM4-40GB"]}), path)
    stats = ps.file_stats("generations/fp/b.parquet", path)
    assert stats["rows"] == 3
    assert stats["gpus"] == {"l40s": 2, "a100_sxm4_40gb": 1}


def test_missing_records_are_counted_once_and_cached(tmp_path):
    store = WorkStore(tmp_path / "w")
    path = labels_file(tmp_path, "a.parquet", [("yes", None), ("no", None)])
    opened = []

    def opener(p):
        opened.append(p)
        return path

    published = {"labels/a.parquet", "mirror:rev", "polygons/a.parquet"}
    first = ps.complete(store, "d", published, opener)
    second = ps.complete(store, "d", published, opener)
    assert first == second
    assert opened == ["labels/a.parquet"]  # only labels/generations files, counted once


def test_totals_sum_every_record():
    records = [
        {"path": "labels/a", "decisions": {"yes": 2, "no": 1}, "failures": {}},
        {"path": "labels/b", "decisions": {"yes": 1, "failed": 1}, "failures": {"empty": 1}},
        {"path": "generations/x", "rows": 5, "gpus": {"l40s": 5}},
        {"path": "generations/y", "rows": 2, "gpus": {"l40s": 1, "a40": 1}},
    ]
    assert ps.totals(records) == {
        "decisions": {"yes": 3, "no": 1, "failed": 1},
        "failures": {"empty": 1},
        "gpus": {"l40s": 6, "a40": 1},
        "unique_texts": 7,
        "cells": {},
        "labelled": 0,
        "located": 0,
    }


def where():
    from landuse_filter.application.geo import Located

    return Located({"c1": [2, 1]}, labelled=4, located=3)


def test_label_stats_carry_the_map_cells_when_a_locator_is_given(tmp_path):
    path = labels_file(tmp_path, "a.parquet", [("yes", None), ("no", None)])
    stats = ps.file_stats("labels/a.parquet", path, lambda p, s: where())
    assert stats["cells"] == {"c1": [2, 1]}
    assert (stats["labelled"], stats["located"]) == (4, 3)


def test_records_without_cells_are_counted_again_when_a_locator_is_given(tmp_path):
    store = WorkStore(tmp_path / "w")
    path = labels_file(tmp_path, "a.parquet", [("yes", None)])
    ps.complete(store, "d", {"labels/a.parquet"}, lambda p: path)  # old record, no cells
    calls = []

    def locate(p, source):
        calls.append(p)
        return where()

    records = ps.complete(store, "d", {"labels/a.parquet"}, lambda p: path, locate)
    ps.complete(store, "d", {"labels/a.parquet"}, lambda p: path, locate)  # now cached
    assert calls == ["labels/a.parquet"]
    assert records[0]["cells"] == {"c1": [2, 1]}


def test_totals_merge_cells_and_sum_located_rows():
    records = [
        {"path": "labels/a", "cells": {"c1": [1, 1]}, "labelled": 3, "located": 2},
        {
            "path": "labels/b",
            "cells": {"c1": [2, 0], "c2": [0, 1]},
            "labelled": 4,
            "located": 3,
        },
    ]
    totals = ps.totals(records)
    assert totals["cells"] == {"c1": (3, 1), "c2": (0, 1)}
    assert (totals["labelled"], totals["located"]) == (7, 5)
