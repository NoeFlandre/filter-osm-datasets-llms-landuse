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
