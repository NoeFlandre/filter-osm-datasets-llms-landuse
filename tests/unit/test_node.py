import asyncio

import pyarrow as pa

from landuse_filter.adapters.schema import CHUNK, PROVENANCE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.node import Runner

PROV = {name: "x" for name, _ in PROVENANCE if name != "created_at"}


class FakeEngine:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    async def generate(self, input_ids):
        self.calls += 1
        await asyncio.sleep(0.001 * (input_ids[0] % 3))
        return {
            "text": "think</think>yes",
            "meta_info": {"completion_tokens": 5, "finish_reason": {"type": "stop"}},
        }


def chunk(store, n=10):
    shas = [f"s{i:02d}" for i in range(n)]
    store.write_chunk(
        "c1",
        pa.table(
            {"text_sha256": shas, "text": shas, "input_ids": [[i, 1] for i in range(n)]},
            schema=CHUNK,
        ),
    )
    return shas


def runner(store, engine, **kw):
    return Runner(store, engine, "fp", PROV, window=4, flush_every=3, flush_seconds=999, **kw)


def test_runs_chunk_to_completion_in_parts(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store)
    stats = asyncio.run(runner(store, FakeEngine()).run(["c1"]))
    assert stats.completed == 10
    assert stats.chunks_done == ["c1"]
    assert stats.parts >= 2  # flushed along the way, not only at the end


def test_stop_then_resume_redoes_only_missing(tmp_path):
    store = WorkStore(tmp_path)
    chunk(store)
    engine = FakeEngine()
    state = {"n": 0}

    def stop():
        state["n"] += 1
        return state["n"] > 6

    first = asyncio.run(runner(store, engine, should_stop=stop).run(["c1"]))
    assert first.chunks_done == []
    done = first.completed
    second_engine = FakeEngine()
    second = asyncio.run(runner(store, second_engine).run(["c1"]))
    assert second.chunks_done == ["c1"]
    assert second_engine.calls == 10 - done


def test_decisions_from_parts(tmp_path):
    from landuse_filter.application.results import decisions_by_sha

    store = WorkStore(tmp_path)
    chunk(store, n=3)
    asyncio.run(runner(store, FakeEngine()).run(["c1"]))
    assert set(decisions_by_sha(store, "fp").values()) == {"yes"}


def test_identical_part_repairs_a_corrupt_copy(tmp_path):
    """Regression (flaky resume scenario): identical content reuses the name; repair it."""
    import pyarrow as pa

    from landuse_filter.adapters.store import WorkStore

    store = WorkStore(tmp_path)
    table = pa.table({"text_sha256": ["a"]})
    part = store.write_part("fp", "c", table)
    path = next(store.part_paths("fp", "c"))
    path.write_bytes(b"torn")
    assert store.write_part("fp", "c", table) == part
    assert store.read_part(path).column("text_sha256").to_pylist() == ["a"]


def test_gathered_decisions_merge_local_and_bucket_parts_without_keeping_them(tmp_path):
    import tempfile
    from pathlib import Path

    from landuse_filter.adapters.remote import DirRemote
    from landuse_filter.application.results import gathered_decisions

    local = WorkStore(tmp_path / "local")
    chunk(local, n=2)
    asyncio.run(runner(local, FakeEngine()).run(["c1"]))
    bucket = DirRemote(tmp_path / "bucket")
    elsewhere = WorkStore(tmp_path / "node")
    elsewhere.write_chunk(
        "c2", pa.table({"text_sha256": ["z"], "text": ["z"], "input_ids": [[1]]}, schema=CHUNK)
    )
    asyncio.run(Runner(elsewhere, FakeEngine(), "fp", PROV, window=2).run(["c2"]))
    for p in elsewhere.part_paths("fp"):
        bucket.put([(p, str(p.relative_to(elsewhere.root)))])
    before = set(Path(tempfile.gettempdir()).glob("luf-gate-*"))
    decisions = gathered_decisions(local, bucket, "fp")
    assert set(decisions) == {"s00", "s01", "z"}
    assert set(Path(tempfile.gettempdir()).glob("luf-gate-*")) == before  # temp tree removed
