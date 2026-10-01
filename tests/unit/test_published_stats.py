import contextlib

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
    assert stats.decisions == {"yes": 1, "no": 1, "failed": 2, "skipped_unsplit": 1}
    assert stats.failures == {"truncated": 1, "empty": 1}


def test_generation_stats_count_rows_per_gpu_key(tmp_path):
    path = tmp_path / "g.parquet"
    pq.write_table(pa.table({"gpu": ["NVIDIA L40S", "NVIDIA L40S", "A100-SXM4-40GB"]}), path)
    stats = ps.file_stats("generations/fp/b.parquet", path)
    assert stats.rows == 3
    assert stats.gpus == {"l40s": 2, "a100_sxm4_40gb": 1}


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
        ps.FileStats("labels/a", decisions={"yes": 2, "no": 1}),
        ps.FileStats("labels/b", decisions={"yes": 1, "failed": 1}, failures={"empty": 1}),
        ps.FileStats("generations/x", rows=5, gpus={"l40s": 5}),
        ps.FileStats("generations/y", rows=2, gpus={"l40s": 1, "a40": 1}),
    ]
    assert ps.totals(records) == ps.PublishedStats(
        decisions={"yes": 3, "no": 1, "failed": 1},
        failures={"empty": 1},
        gpus={"l40s": 6, "a40": 1},
        unique_texts=7,
        cells={},
        labelled=0,
        located=0,
    )


def test_ledger_lines_written_by_older_code_still_load_and_round_trip():
    labels = {"path": "labels/a", "decisions": {"yes": 1}, "failures": {}}
    located = {**labels, "cells": {"c": [1, 0]}, "labelled": 1, "located": 1}
    generations = {"path": "generations/fp/b", "rows": 3, "gpus": {"l40s": 3}}
    for line in (labels, located, generations):
        assert ps.FileStats.from_json(line).to_json() == line
    assert ps.FileStats.from_json(labels).cells is None  # counted again once a locator is given
    assert ps.FileStats.from_json({"path": "labels/x"}) == ps.FileStats("labels/x")


def where():
    from landuse_filter.application.geo import Located

    return Located({"c1": [2, 1]}, labelled=4, located=3)


def test_label_stats_carry_the_map_cells_when_a_locator_is_given(tmp_path):
    path = labels_file(tmp_path, "a.parquet", [("yes", None), ("no", None)])
    stats = ps.file_stats("labels/a.parquet", path, lambda p, s: where())
    assert stats.cells == {"c1": [2, 1]}
    assert (stats.labelled, stats.located) == (4, 3)


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
    assert records[0].cells == {"c1": [2, 1]}


def test_totals_merge_cells_and_sum_located_rows():
    records = [
        ps.FileStats("labels/a", cells={"c1": [1, 1]}, labelled=3, located=2),
        ps.FileStats("labels/b", cells={"c1": [2, 0], "c2": [0, 1]}, labelled=4, located=3),
    ]
    totals = ps.totals(records)
    assert totals.cells == {"c1": (3, 1), "c2": (0, 1)}
    assert (totals.labelled, totals.located) == (7, 5)


def test_compact_jsonl_keeps_the_last_line_per_key_only_when_it_pays(tmp_path):
    store = WorkStore(tmp_path)
    store.append_jsonl("l.jsonl", [{"path": "a", "n": 1}, {"path": "a", "n": 2}])
    assert store.compact_jsonl("l.jsonl") == [{"path": "a", "n": 2}]
    assert len(store.read_jsonl("l.jsonl")) == 2  # not worth a rewrite yet
    store.append_jsonl("l.jsonl", [{"path": "a", "n": i} for i in range(300)])
    assert store.compact_jsonl("l.jsonl") == [{"path": "a", "n": 299}]
    assert store.read_jsonl("l.jsonl") == [{"path": "a", "n": 299}]


def test_records_are_appended_in_chunks_so_a_stopped_job_keeps_them(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RECORD_FLUSH", 2)
    store = WorkStore(tmp_path / "w")
    paths = []
    for i in range(5):
        labels_file(tmp_path, f"f{i}.parquet", [("yes", None)])
        paths.append(f"labels/f{i}.parquet")
    seen = []

    def opener(p):
        if p.endswith("f3.parquet"):
            raise KeyboardInterrupt  # stopped while counting the fourth file
        return tmp_path / p.split("/")[-1]

    with contextlib.suppress(KeyboardInterrupt):
        ps.complete(store, "d", set(paths), opener, on_progress=lambda: seen.append(1))
    assert len(store.read_jsonl(ps.ledger("d"))) == 2  # the first chunk survived
    assert seen == [1]
