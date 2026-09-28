import asyncio

import pyarrow as pa
from pytest_bdd import given, parsers, scenarios, then, when

from landuse_filter.adapters.schema import CHUNK, PROVENANCE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.node import Runner
from landuse_filter.application.results import canonical_generations, decisions_by_sha

scenarios("features/resume.feature")
PROV = {name: "x" for name, _ in PROVENANCE if name != "created_at"}


class Engine:
    """Deterministic: the verdict depends only on the prompt (like greedy decoding)."""

    def __init__(self):
        self.calls = 0

    async def generate(self, input_ids):
        self.calls += 1
        await asyncio.sleep(0.0005 * (input_ids[0] % 5))
        verdict = "yes" if input_ids[0] % 3 else "no"
        return {
            "text": f"reasoning</think>{verdict}",
            "meta_info": {"completion_tokens": 3, "finish_reason": {"type": "stop"}},
        }


def run(store, engine, stop_after=None):
    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return stop_after is not None and seen["n"] > stop_after

    runner = Runner(
        store, engine, "fp", PROV, window=4, flush_every=4, flush_seconds=999, should_stop=stop
    )
    return asyncio.run(runner.run(["c"]))


@given(parsers.parse("a chunk of {n:d} sentences"), target_fixture="ctx")
def chunk(tmp_path, n):
    store = WorkStore(tmp_path / "w")
    shas = [f"s{i:03d}" for i in range(n)]
    store.write_chunk(
        "c",
        pa.table(
            {"text_sha256": shas, "text": shas, "input_ids": [[i] for i in range(n)]}, schema=CHUNK
        ),
    )
    return {"store": store, "n": n, "tmp": tmp_path, "shas": shas}


@given("a completed run whose first part was corrupted on disk")
def corrupted(ctx):
    run(ctx["store"], Engine())
    first = next(ctx["store"].part_paths("fp", "c"))
    first.write_bytes(first.read_bytes()[:-10])


@when("a job is stopped after about half of them")
def first_job(ctx):
    ctx["first"] = run(ctx["store"], Engine(), stop_after=12)


@when("a new job runs the same chunk")
def second_job(ctx):
    engine = Engine()
    ctx["second"] = run(ctx["store"], engine)
    ctx["second_calls"] = engine.calls


@then("every sentence has exactly one canonical generation")
def one_each(ctx):
    rows = list(canonical_generations(ctx["store"], "fp"))
    assert sorted(r["text_sha256"] for r in rows) == ctx["shas"]


@then("the second job only generated the sentences the first one had not finished")
def only_missing(ctx):
    assert ctx["second_calls"] == ctx["n"] - ctx["first"].completed


@then("the decisions equal those of an uninterrupted run")
def same_decisions(ctx):
    clean = WorkStore(ctx["tmp"] / "clean")
    clean.write_chunk("c", ctx["store"].read_chunk("c"))
    run(clean, Engine())
    assert decisions_by_sha(ctx["store"], "fp") == decisions_by_sha(clean, "fp")
