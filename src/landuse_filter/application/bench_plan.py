"""The 25,500-item benchmark as a work source, so parity runs use the production path."""

from collections.abc import Callable

import pyarrow as pa

from landuse_filter.adapters.benchmark import BenchmarkItem
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.domain.hashing import sha256_text
from landuse_filter.domain.planning import UniqueText, plan_chunks
from landuse_filter.domain.prompting import render_prompt

BENCH = "benchmark"


def plan_benchmark(
    store: WorkStore,
    items: list[BenchmarkItem],
    encode: Callable[[str], list[int]],
    *,
    template: str,
    fp: str,
    chunk_size: int,
    name: str = BENCH,
) -> int:
    text_of = {sha256_text(i.sentence): i.sentence for i in items}
    ids = {sha: encode(render_prompt(template, text)) for sha, text in text_of.items()}
    texts = [UniqueText(sha, len(ids[sha]), (-1, 0)) for sha in text_of]
    chunks = plan_chunks(texts, fp, chunk_size)
    existing = {row["chunk_id"] for row in store.read_jsonl(f"plans/{name}/{fp}/chunks.jsonl")}
    for chunk in chunks:
        table = pa.table(
            {
                "text_sha256": list(chunk.text_sha256s),
                "text": [text_of[s] for s in chunk.text_sha256s],
                "input_ids": [ids[s] for s in chunk.text_sha256s],
            },
            schema=CHUNK,
        )
        store.write_chunk(chunk.chunk_id, table)
        if chunk.chunk_id not in existing:
            store.append_jsonl(
                f"plans/{name}/{fp}/chunks.jsonl",
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "order": list(chunk.order),
                        "size": len(chunk.text_sha256s),
                        "prompt_tokens": sum(len(ids[s]) for s in chunk.text_sha256s),
                    }
                ],
            )
    store.write_json(
        f"plans/{name}/{fp}/items.json", {i.item_id: sha256_text(i.sentence) for i in items}
    )
    return len(chunks)


SMOKE = "smoke"
SMOKE_PER_LANGUAGE = 20
SMOKE_SEED = 0


def smoke_items(items: list[BenchmarkItem]) -> list[BenchmarkItem]:
    """Fixed, label-stratified subset: 20 items per language (10 yes / 10 no if possible)."""
    import random

    rng = random.Random(SMOKE_SEED)  # noqa: S311 - reproducible sampling, not security
    chosen: list[BenchmarkItem] = []
    by_language: dict[str, list[BenchmarkItem]] = {}
    for item in sorted(items, key=lambda i: i.item_id):
        by_language.setdefault(item.language, []).append(item)
    for language in sorted(by_language):
        group = by_language[language]
        half = SMOKE_PER_LANGUAGE // 2
        for label in ("yes", "no"):
            pool = [i for i in group if i.label == label]
            chosen += rng.sample(pool, min(half, len(pool)))
    return chosen


def smoke_fp(fp: str, gpu: str) -> str:
    """Smoke results live apart from production so dedup never mixes them."""
    return f"{fp}-smoke-{gpu}"
