"""Exact-value tests that pin the small pure application modules for mutation testing."""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import CorruptPartError, WorkStore
from landuse_filter.application import results
from landuse_filter.application.assignment import Assignment
from landuse_filter.application.repair import dedupe_generations
from landuse_filter.application.results import (
    canonical_generations,
    decisions_by_sha,
    gathered_decisions,
    verdict_of,
)
from landuse_filter.application.status import summarize
from landuse_filter.config import GENERATION_FP


def test_assignment_keeps_a_key_named_extra_in_extra():
    assignment = Assignment.from_json({"id": "x", "extra": 1})
    assert assignment.id == "x"
    assert assignment.extra == {"extra": 1}


def test_dedupe_orders_files_by_name_not_by_directory(tmp_path):
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    early = tmp_path / "z" / "a.parquet"
    late = tmp_path / "a" / "b.parquet"
    for path in (early, late):
        pq.write_table(pa.table({"generation_id": ["x"]}), path)
    assert dedupe_generations([late, early]) == {late: None}


# --- results ---------------------------------------------------------------------


def _row(sha, raw="</think>yes", truncated=False):
    return {"text_sha256": sha, "raw_output": raw, "truncated": truncated}


class FakeStore:
    """Serves parts by path: a dict path -> rows, or an exception to raise."""

    def __init__(self, parts):
        self.parts = parts

    def part_paths(self, fp):
        return list(self.parts)

    def read_part(self, path):
        rows = self.parts[path]
        if isinstance(rows, Exception):
            raise rows
        return pa.Table.from_pylist(rows)


def test_canonical_generation_order_is_chunk_then_part_id():
    parts = {
        Path("/p/fp/c2/a.parquet"): [_row("s", "</think>no")],
        Path("/q/fp/c1/b.parquet"): [_row("s", "</think>yes")],
    }
    (row,) = canonical_generations(FakeStore(parts), "fp")
    assert row["raw_output"] == "</think>yes"


def test_corrupt_part_is_skipped_and_later_parts_still_read():
    parts = {
        Path("/p/fp/c1/a.parquet"): CorruptPartError("a"),
        Path("/p/fp/c1/b.parquet"): [_row("s")],
    }
    assert [r["text_sha256"] for r in canonical_generations(FakeStore(parts), "fp")] == ["s"]


def test_a_text_seen_in_an_earlier_part_is_not_yielded_again():
    parts = {
        Path("/p/fp/c1/a.parquet"): [_row("s", "</think>yes")],
        Path("/p/fp/c1/b.parquet"): [_row("s", "</think>no"), _row("t")],
    }
    rows = list(canonical_generations(FakeStore(parts), "fp"))
    assert [(r["text_sha256"], r["raw_output"]) for r in rows] == [
        ("s", "</think>yes"),
        ("t", "</think>yes"),
    ]


def test_truncated_generation_is_a_failure_even_with_an_answer():
    assert verdict_of(_row("s", "</think>yes", truncated=True)).decision.value not in ("yes", "no")
    assert verdict_of(_row("s", "</think>yes", truncated=False)).decision.value == "yes"


def test_decisions_map_yes_no_and_failures_to_none():
    parts = {
        Path("/p/fp/c1/a.parquet"): [
            _row("y", "</think>yes"),
            _row("n", "</think>no"),
            _row("f", "no end of thinking"),
        ]
    }
    assert decisions_by_sha(FakeStore(parts), "fp") == {"y": "yes", "n": "no", "f": None}


class SpyRemote(DirRemote):
    def __init__(self, root, *, wipe_tree=False):
        super().__init__(root)
        self.fetched: list[str] = []
        self.targets: list[Path] = []
        self.wipe_tree = wipe_tree

    def get(self, files):
        self.fetched += [src for src, _ in files]
        self.targets += [dst for _, dst in files]
        if self.wipe_tree:
            dst = files[0][1]
            shutil.rmtree(next(p for p in dst.parents if p.name.startswith("luf-gate-")))
            return
        super().get(files)


def _write_part(store, fp, chunk, shas):
    table = pa.table(
        {
            "text_sha256": shas,
            "raw_output": ["</think>yes"] * len(shas),
            "truncated": [False] * len(shas),
        }
    )
    return store.write_part(fp, chunk, table)


def test_gathered_decisions_fetches_only_new_parquet_parts_into_a_luf_gate_tree(tmp_path):
    local = WorkStore(tmp_path / "local")
    own = _write_part(local, "fp", "c1", ["a"])
    _write_part(local, "fp", "c1", ["b"])  # two parts share one chunk directory
    node = WorkStore(tmp_path / "node")
    new = _write_part(node, "fp", "c2", ["z"])
    bucket = SpyRemote(tmp_path / "bucket")
    for store in (local, node):
        for path in store.part_paths("fp"):
            bucket.put([(path, str(path.relative_to(store.root)))])
    (tmp_path / "bucket" / "parts" / "fp" / "manifest.json").write_text("{}")
    decisions = gathered_decisions(local, bucket, "fp")
    assert decisions == {"a": "yes", "b": "yes", "z": "yes"}
    assert bucket.fetched == [f"parts/fp/c2/{new}.parquet"]
    assert own not in bucket.fetched
    assert all(any(p.name.startswith("luf-gate-") for p in t.parents) for t in bucket.targets)


def test_gathered_decisions_tolerates_a_tree_that_vanished(tmp_path):
    local = WorkStore(tmp_path / "local")
    node = WorkStore(tmp_path / "node")
    _write_part(node, "fp", "c2", ["z"])
    bucket = SpyRemote(tmp_path / "bucket", wipe_tree=True)
    for path in node.part_paths("fp"):
        bucket.put([(path, str(path.relative_to(node.root)))])
    assert gathered_decisions(local, bucket, "fp") == {}


def test_results_module_exposes_gathered_decisions():
    assert results.gathered_decisions is gathered_decisions


# --- status ----------------------------------------------------------------------


class SpyStore(WorkStore):
    def __init__(self, root):
        super().__init__(root)
        self.jsonl_reads: list[str] = []

    def read_jsonl(self, relative):
        self.jsonl_reads.append(relative)
        return super().read_jsonl(relative)


def test_status_exact_report(tmp_path):
    store = SpyStore(tmp_path)
    store.append_jsonl(
        f"plans/d/{GENERATION_FP}/chunks.jsonl",
        [{"chunk_id": "a", "size": 3600}, {"chunk_id": "b", "size": 10}],
    )
    store.append_jsonl(
        f"plans/idle/{GENERATION_FP}/chunks.jsonl", [{"chunk_id": "i", "size": 5000}]
    )
    store.append_jsonl(f"plans/dead/{GENERATION_FP}/chunks.jsonl", [{"chunk_id": "x", "size": 7}])
    store.append_jsonl(f"plans/full/{GENERATION_FP}/chunks.jsonl", [{"chunk_id": "f", "size": 4}])
    store.append_jsonl("complete.jsonl", [{"chunk_id": "b"}, {"chunk_id": "f"}])
    live = {"state": "submitted", "fp": GENERATION_FP, "gpu": "NVIDIA L40S"}
    store.write_json("assignments/1.json", {**live, "chunks": ["a"]})
    store.write_json("assignments/2.json", {**live, "chunks": ["i"], "gpu": "no-profile"})
    store.write_json("assignments/3.json", {**live, "state": "submitting", "chunks": ["b"]})
    store.write_json("assignments/4.json", {**live, "state": "done", "chunks": ["a"]})
    store.write_json("assignments/5.json", {**live, "fp": "old", "chunks": ["a"]})
    store.write_json("profiles/l40s.json", {"gpu": "l40s", "sentences_per_second": 1.2345})
    store.write_json(
        "jobs/s/1.json",
        {"gpu": "L40S", "sentences_per_second": 1.0, "completed": 100, "failed": 10},
    )
    store.write_json("jobs/s/2.json", {"gpu": "L40S", "sentences_per_second": 1.0015})
    store.write_json("jobs/s/3.json", {"gpu": "H100", "sentences_per_second": 7.0})
    report = summarize(store, ["d", "idle", "dead", "full"])
    assert store.jsonl_reads[0] == "complete.jsonl"
    assert report == {
        "config_fingerprint": GENERATION_FP,
        "d": {
            "chunks": 2,
            "chunks_complete": 1,
            "texts": 3610,
            "texts_complete": 10,
            "live_jobs": 2,
            "sentences_per_second": 2.47,
            "eta_hours": 0.4,
        },
        "idle": {
            "chunks": 1,
            "chunks_complete": 0,
            "texts": 5000,
            "texts_complete": 0,
            "live_jobs": 1,
            "sentences_per_second": 1.4,
            "eta_hours": 1.0,
        },
        "dead": {
            "chunks": 1,
            "chunks_complete": 0,
            "texts": 7,
            "texts_complete": 0,
            "live_jobs": 0,
            "sentences_per_second": 0,
            "eta_hours": None,
        },
        "full": {
            "chunks": 1,
            "chunks_complete": 1,
            "texts": 4,
            "texts_complete": 4,
            "live_jobs": 0,
            "sentences_per_second": 0,
            "eta_hours": 0.0,
        },
        "assignments": {"submitted": 3, "submitting": 1, "done": 1},
        "throughput_by_gpu": {"L40S": 1.001, "H100": 7.0},
        "alerts": ["failed rate 10.0% > 5% (benchmark 2.4%)"],
    }
