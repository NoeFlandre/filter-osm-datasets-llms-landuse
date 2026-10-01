"""Node-side runner: stream an assignment's chunks through the engine, checkpointing.

Requests are kept in flight continuously (no batch barrier) up to ``window``; every
completion is buffered and flushed as a content-addressed part every ``flush_every``
results or ``flush_seconds``. On a stop request (SIGTERM / OAR checkpoint) the runner
stops admitting, cancels in-flight requests and flushes what completed, so a killed
job loses only requests that had not finished.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import pyarrow as pa

from landuse_filter.adapters.schema import generation_table
from landuse_filter.adapters.store import CorruptPartError, WorkStore
from landuse_filter.domain.completion import Part
from landuse_filter.domain.parsing import parse_generation
from landuse_filter.domain.records import RULE_NO_LETTERS, Generation, from_sglang, rule_decided_no
from landuse_filter.domain.sentences import Decision, has_no_letters


class AsyncEngine(Protocol):
    def generate(self, input_ids: list[int]) -> Awaitable[dict]: ...


@dataclass
class RunStats:
    completed: int = 0
    rule_decided: int = 0  # letterless texts labelled no without the model (ADR-0024)
    generated_tokens: int = 0
    parts: int = 0
    failed: int = 0  # generations whose verdict failed to parse (drift signal)
    chunks_done: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    first_result: float | None = None  # monotonic instants of the first/last finished text
    last_result: float | None = None

    def note_results(self) -> None:
        now = time.monotonic()
        self.last_result = now
        if self.first_result is None:
            self.first_result = now

    @property
    def sentences_per_second(self) -> float:
        elapsed = time.monotonic() - self.started
        return (self.completed - self.rule_decided) / elapsed if elapsed > 0 else 0.0


@dataclass
class Runner:
    store: WorkStore
    engine: AsyncEngine
    fp: str
    provenance: dict[str, str]
    window: int = 64
    flush_every: int = 256
    flush_seconds: float = 120.0
    should_stop: Callable[[], bool] = lambda: False
    # Called after each part is written locally: (chunk id, part id, text hashes).
    on_part: Callable[[str, str, list[str]], None] | None = None
    stats: RunStats = field(default_factory=RunStats)

    def valid_parts(self, chunk_id: str) -> list[Part]:
        """Parts whose bytes still match their name; corrupt ones are ignored (redone)."""
        parts = []
        for path in self.store.part_paths(self.fp, chunk_id):
            try:
                shas = self.store.read_part(path).column("text_sha256").to_pylist()
            except CorruptPartError:
                continue
            parts.append(Part(path.stem, tuple(shas)))
        return parts

    def done_shas(self, chunk_id: str) -> set[str]:
        """Hashes already generated: local parts plus remote manifests fetched for them."""
        from landuse_filter.application.sync import manifest_shas

        local = {sha for part in self.valid_parts(chunk_id) for sha in part.text_sha256s}
        return local | manifest_shas(self.store, self.fp, chunk_id)

    async def run(self, chunk_ids: list[str]) -> RunStats:
        for chunk_id in chunk_ids:
            if self.should_stop():
                break
            await self.run_chunk(chunk_id)
        return self.stats

    async def run_chunk(self, chunk_id: str) -> None:
        table = self.store.read_chunk(chunk_id)
        expected = table.column("text_sha256").to_pylist()
        done = self.done_shas(chunk_id)
        todo = iter([i for i, sha in enumerate(expected) if sha not in done])
        await self._stream(chunk_id, table, todo)
        if not set(expected) - self.done_shas(chunk_id):
            self.stats.chunks_done.append(chunk_id)

    async def _stream(self, chunk_id: str, table: pa.Table, todo: Iterator[int]) -> None:
        """Keep up to ``window`` requests in flight; flush completions periodically."""
        buffer: list[Generation] = []
        last_flush = time.monotonic()
        pending: set[asyncio.Task] = set()
        self._admit(pending, table, todo)
        while pending:
            finished, _ = await asyncio.wait(
                pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            pending -= finished
            buffer.extend(task.result() for task in finished)
            if finished:
                self.stats.note_results()
            if self.should_stop():
                for task in pending:
                    task.cancel()
                break
            if (
                len(buffer) >= self.flush_every
                or time.monotonic() - last_flush >= self.flush_seconds
            ):
                self._flush(chunk_id, buffer)
                buffer, last_flush = [], time.monotonic()
            self._admit(pending, table, todo)
        self._flush(chunk_id, buffer)

    def _admit(self, pending: set[asyncio.Task], table: pa.Table, todo: Iterator[int]) -> None:
        while len(pending) < self.window and not self.should_stop():
            i = next(todo, None)
            if i is None:
                return
            pending.add(asyncio.ensure_future(self._one(table, i)))

    async def _one(self, table: pa.Table, i: int) -> Generation:
        ids = table.column("input_ids")[i].as_py()
        sha = table.column("text_sha256")[i].as_py()
        if has_no_letters(table.column("text")[i].as_py()):
            return rule_decided_no(sha, len(ids))
        output = await self.engine.generate(ids)
        return from_sglang(sha, output, len(ids))

    def _flush(self, chunk_id: str, rows: list[Generation]) -> None:
        if not rows:
            return
        stamped = {**self.provenance, "created_at": _utc_now()}
        part = self.store.write_part(self.fp, chunk_id, generation_table(rows, stamped))
        if self.on_part:
            self.on_part(chunk_id, part, [r.text_sha256 for r in rows])
        self.stats.completed += len(rows)
        self.stats.rule_decided += sum(r.finish_reason == RULE_NO_LETTERS for r in rows)
        self.stats.generated_tokens += sum(r.generated_tokens for r in rows)
        self.stats.parts += 1
        self.stats.failed += sum(
            parse_generation(r.raw_output, truncated=r.truncated).decision is Decision.FAILED
            for r in rows
        )


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
