"""Part-upload economy (ADR-0030): large flush thresholds, one commit per part."""

import asyncio
from types import SimpleNamespace

import pyarrow as pa

from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import node
from landuse_filter.application.node import Runner
from landuse_filter.application.sync import upload_part
from tests.unit.test_node import PROV, FakeEngine, chunk, runner


def _default_runner(store, engine=None):
    return Runner(store, engine or FakeEngine(), "fp", PROV, window=1)


def test_a_run_of_exactly_the_default_count_threshold_gives_one_part(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store, n=2048)
    stats = asyncio.run(_default_runner(store).run(["c1"]))
    assert (stats.parts, stats.completed) == (1, 2048)


def test_the_default_count_threshold_flushes_exactly_at_2048(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store, n=2049)
    stats = asyncio.run(_default_runner(store).run(["c1"]))
    assert (stats.parts, stats.completed) == (2, 2049)


class ClockedEngine(FakeEngine):
    """Each generation advances a fake clock by ``step`` seconds."""

    def __init__(self, clock, step):
        super().__init__()
        self.clock, self.step = clock, step

    async def generate(self, input_ids):
        self.clock[0] += self.step
        return await super().generate(input_ids)


def _part_sizes_with_clock(tmp_path, monkeypatch, step, n):
    clock = [0.0]
    # Only the runner's clock is faked; asyncio keeps the real one.
    monkeypatch.setattr(node, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    store = WorkStore(tmp_path)
    chunk(store, n=n)
    sizes = []
    r = Runner(
        store,
        ClockedEngine(clock, step),
        "fp",
        PROV,
        window=1,
        on_part=lambda _chunk, _part, shas: sizes.append(len(shas)),
    )
    asyncio.run(r.run(["c1"]))
    return sizes


def test_default_time_threshold_does_not_flush_just_under_300_seconds(tmp_path, monkeypatch):
    assert _part_sizes_with_clock(tmp_path, monkeypatch, step=29.0, n=10) == [10]  # 290 s


def test_default_time_threshold_flushes_at_300_seconds(tmp_path, monkeypatch):
    # 20 results at 30 s: the flush happens at 300 s (10 rows), not later; 10 rows remain
    assert _part_sizes_with_clock(tmp_path, monkeypatch, step=30.0, n=20) == [10, 10]


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
