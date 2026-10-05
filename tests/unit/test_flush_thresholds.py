"""Part-upload economy (ADR-0030): large flush thresholds, one commit per part."""

import asyncio

import pyarrow as pa

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import node
from landuse_filter.application.node import Runner
from landuse_filter.application.sync import upload_part
from tests.unit.test_node import PROV, FakeEngine, chunk, runner


def test_default_flush_thresholds_are_large():
    assert node.FLUSH_EVERY == 2048
    assert node.FLUSH_SECONDS == 300.0
    r = Runner(WorkStore("/nonexistent"), FakeEngine(), "fp", PROV)
    assert (r.flush_every, r.flush_seconds) == (2048, 300.0)


def test_large_threshold_gives_one_part(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store)
    stats = asyncio.run(runner(store, FakeEngine(), flush_every=10**6).run(["c1"]))
    assert stats.parts == 1
    assert stats.completed == 10
    assert stats.chunks_done == ["c1"]


def test_stop_flushes_the_remainder_in_one_part(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store)
    engine = FakeEngine()
    stats = asyncio.run(
        runner(
            store, engine, window=1, flush_every=10**6, should_stop=lambda: engine.calls >= 4
        ).run(["c1"])
    )
    assert stats.parts == 1
    assert 0 < stats.completed < 10


def test_upload_part_is_one_remote_commit(tmp_path):
    store = WorkStore(tmp_path / "w")
    part = store.write_part("fp", "c", pa.table({"text_sha256": ["a"]}))
    calls = []

    class Spy:
        def put(self, files):
            calls.append([dst for _, dst in files])

    upload_part(Spy(), store, "fp", "c", part_id=part, shas=["a"])
    assert calls == [[f"parts/fp/c/{part}.parquet", f"parts/fp/c/{part}.json"]]
