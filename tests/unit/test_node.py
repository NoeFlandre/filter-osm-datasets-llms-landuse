import asyncio

import pyarrow as pa

from landuse_filter.adapters.engine import SGLangEngine
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import CHUNK, PROVENANCE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.node import Runner
from landuse_filter.application.results import decisions_by_sha
from landuse_filter.application.sync import fetch, fetch_manifests, prune_local_part, upload_part
from landuse_filter.domain.parsing import Failure, parse_generation

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


class MappedEngine:
    """Deterministic responses whose completion order differs from input order."""

    responses = {
        101: ("analysis alpha</think>yes", 7, 0.12),
        202: ("analysis beta</think>no", 4, 0.06),
        303: ("analysis gamma</think>yes", 9, 0.01),
    }

    def __init__(self):
        self.calls = 0
        self.completion_order = []

    async def generate(self, input_ids):
        prompt_id = input_ids[0]
        raw_output, token_count, delay = self.responses[prompt_id]
        await asyncio.sleep(delay)
        self.calls += 1
        self.completion_order.append(prompt_id)
        return {
            "text": raw_output,
            "meta_info": {"completion_tokens": token_count, "finish_reason": {"type": "stop"}},
        }


class ContextAwareEngine(SGLangEngine):
    """Drive the real SGLangEngine error translation through the Runner."""

    def __init__(self):
        self._engine = self
        self._sampling = {"max_new_tokens": 4_096}
        self.received_lengths = []

    async def async_generate(self, input_ids, sampling_params):
        self.received_lengths.append(len(input_ids))
        if len(input_ids) >= 131_072:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the model's "
                "context length (131072 tokens)"
            )
        max_new_tokens = sampling_params["max_new_tokens"]
        if len(input_ids) + max_new_tokens > 131_072:
            total_tokens = len(input_ids) + max_new_tokens
            raise ValueError(
                "Requested token count exceeds the model's maximum context length "
                f"of 131072 tokens. You requested a total of {total_tokens} tokens: "
                f"{len(input_ids)} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                "of tokens in the input messages or the completion to fit within the limit."
            )
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


def mapped_chunk(store):
    store.write_chunk(
        "mapped",
        pa.table(
            {
                "text_sha256": ["sha-alpha", "sha-beta", "sha-gamma"],
                "text": ["prompt alpha", "prompt beta", "prompt gamma"],
                "input_ids": [[101, 11], [202, 22], [303, 33]],
            },
            schema=CHUNK,
        ),
    )


def stored_result_map(store):
    rows = {}
    for path in store.part_paths("fp", "mapped"):
        for row in store.read_part(path).to_pylist():
            rows[row["text_sha256"]] = row
    decisions = decisions_by_sha(store, "fp")
    return {
        sha: (decisions[sha], row["raw_output"], row["generated_tokens"])
        for sha, row in rows.items()
    }


def runner(store, engine, **kw):
    return Runner(
        store,
        engine,
        "fp",
        PROV,
        window=kw.pop("window", 4),
        flush_every=kw.pop("flush_every", 3),
        flush_seconds=kw.pop("flush_seconds", 999),
        **kw,
    )


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
    store = WorkStore(tmp_path)
    chunk(store, n=3)
    asyncio.run(runner(store, FakeEngine()).run(["c1"]))
    assert set(decisions_by_sha(store, "fp").values()) == {"yes"}


def test_context_rejected_requests_are_manifested_while_other_rows_continue(tmp_path):
    store = WorkStore(tmp_path / "first")
    store.write_chunk(
        "oversized",
        pa.table(
            {
                "text_sha256": [
                    "too-long",
                    "at-limit",
                    "budget-too-long",
                    "ordinary-1",
                    "ordinary-2",
                ],
                "text": [
                    "long request",
                    "limit request",
                    "budget request",
                    "ordinary one",
                    "ordinary two",
                ],
                "input_ids": [
                    [1] * 330_533,
                    [5] * 131_072,
                    [6] * 127_000,
                    [2, 3],
                    [4, 5],
                ],
            },
            schema=CHUNK,
        ),
    )
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(store.path("chunks/oversized.parquet"), "chunks/oversized.parquet")])
    engine = ContextAwareEngine()

    def persist_part(chunk_id, part_id, shas):
        upload_part(remote, store, "fp", chunk_id, part_id=part_id, shas=shas)
        prune_local_part(store, "fp", chunk_id, part_id)

    stats = asyncio.run(
        runner(store, engine, on_part=persist_part, flush_every=10).run(["oversized"])
    )

    assert sorted(engine.received_lengths) == [2, 2, 127_000, 131_072, 330_533]
    assert stats.completed == 5
    assert stats.failed == 3
    assert stats.chunks_done == ["oversized"]

    remote_parts = remote.ls("parts/fp/oversized/")
    assert len([p for p in remote_parts if p.endswith(".parquet")]) == 1
    assert len([p for p in remote_parts if p.endswith(".json")]) == 1
    verification = WorkStore(tmp_path / "verification")
    parquet_path = next(p for p in remote_parts if p.endswith(".parquet"))
    remote.get([(parquet_path, verification.path(parquet_path))])
    rows = {
        row["text_sha256"]: row
        for row in verification.read_part(verification.path(parquet_path)).to_pylist()
    }
    failure = rows["too-long"]
    assert failure["prompt_tokens"] == 330_533
    assert failure["generated_tokens"] == 0
    assert failure["finish_reason"] == "input_too_long"
    assert failure["truncated"] is False
    assert (
        parse_generation(failure["raw_output"], truncated=False).failure is Failure.INPUT_TOO_LONG
    )
    limit_failure = rows["at-limit"]
    assert limit_failure["prompt_tokens"] == 131_072
    assert limit_failure["raw_output"] == "luf_failure:input_too_long:131072>=131072"
    assert (
        parse_generation(limit_failure["raw_output"], truncated=False).failure
        is Failure.INPUT_TOO_LONG
    )
    budget_failure = rows["budget-too-long"]
    assert budget_failure["prompt_tokens"] == 127_000
    assert budget_failure["raw_output"] == "luf_failure:input_too_long:127000+4096>131072"
    assert (
        parse_generation(budget_failure["raw_output"], truncated=False).failure
        is Failure.INPUT_TOO_LONG
    )
    assert decisions_by_sha(verification, "fp") == {
        "too-long": None,
        "at-limit": None,
        "budget-too-long": None,
        "ordinary-1": "yes",
        "ordinary-2": "yes",
    }

    restarted = WorkStore(tmp_path / "restart")
    fetch(remote, restarted, ["chunks/oversized.parquet"])
    manifests = fetch_manifests(remote, restarted, "parts/fp/oversized/")
    assert len(manifests) == 1
    retry_engine = FakeEngine()
    resumed = asyncio.run(runner(restarted, retry_engine).run(["oversized"]))
    assert retry_engine.calls == 0
    assert resumed.chunks_done == ["oversized"]


def test_prompt_results_match_literal_map_after_out_of_order_and_resume(tmp_path):
    expected = {
        "sha-alpha": ("yes", "analysis alpha</think>yes", 7),
        "sha-beta": ("no", "analysis beta</think>no", 4),
        "sha-gamma": ("yes", "analysis gamma</think>yes", 9),
    }

    uninterrupted_store = WorkStore(tmp_path / "uninterrupted")
    mapped_chunk(uninterrupted_store)
    uninterrupted_engine = MappedEngine()
    asyncio.run(runner(uninterrupted_store, uninterrupted_engine).run(["mapped"]))

    assert uninterrupted_engine.completion_order == [303, 202, 101]
    assert stored_result_map(uninterrupted_store) == expected

    resumed_store = WorkStore(tmp_path / "resumed")
    mapped_chunk(resumed_store)
    first_engine = MappedEngine()
    first = asyncio.run(
        runner(
            resumed_store,
            first_engine,
            window=1,
            flush_every=10,
            should_stop=lambda: len(first_engine.completion_order) >= 2,
        ).run(["mapped"])
    )
    assert first.completed == 2
    assert first.chunks_done == []

    resume_engine = MappedEngine()
    resumed = asyncio.run(runner(resumed_store, resume_engine).run(["mapped"]))
    assert resume_engine.calls == 1
    assert resumed.chunks_done == ["mapped"]
    assert stored_result_map(resumed_store) == expected


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


def test_gathered_decisions_merge_local_and_bucket_parts_without_keeping_them(
    tmp_path, monkeypatch
):
    """The temp tree is checked in a private tmp dir: other processes (parallel test runs,
    `luf bench admit`) create luf-gate-* dirs in the shared one, which made this flaky."""
    import tempfile
    from pathlib import Path

    private = tmp_path / "tmp"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))

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


def test_checkpoint_signal_with_more_chunks_than_can_finish_flushes_and_exits(tmp_path):
    """Over-assigned job (ADR-0018): SIGUSR2/SIGTERM flush finished texts, unfinished stay open."""
    import os
    import signal

    from landuse_filter.application.node_main import Stop

    for sig in (signal.SIGUSR2, signal.SIGTERM):
        store = WorkStore(tmp_path / str(int(sig)))
        for cid in ("c1", "c2", "c3"):
            shas = [f"{cid}-{i:02d}" for i in range(10)]
            store.write_chunk(
                cid,
                pa.table(
                    {"text_sha256": shas, "text": shas, "input_ids": [[i, 1] for i in range(10)]},
                    schema=CHUNK,
                ),
            )

        class SignallingEngine(FakeEngine):
            async def generate(self, input_ids, _sig=sig):
                if self.calls == 14:
                    os.kill(os.getpid(), _sig)
                return await super().generate(input_ids)

        old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR2, signal.SIGINT)}
        uploaded: list[tuple[str, list[str]]] = []
        try:
            stop = Stop()
            stats = asyncio.run(
                runner(
                    store,
                    SignallingEngine(),
                    should_stop=stop,
                    on_part=lambda chunk, _part, shas, sink=uploaded: sink.append((chunk, shas)),
                ).run(["c1", "c2", "c3"])
            )
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
        assert stop.requested
        assert 10 <= stats.completed < 30  # stopped early, nothing thrown away
        assert sum(len(shas) for _, shas in uploaded) == stats.completed  # all flushed
        per_chunk: dict[str, int] = {}
        for chunk_id, shas in uploaded:
            per_chunk[chunk_id] = per_chunk.get(chunk_id, 0) + len(shas)
        # a chunk is reported done only when every one of its texts was uploaded
        assert all(per_chunk[c] == 10 for c in stats.chunks_done)
        assert "c3" not in stats.chunks_done
