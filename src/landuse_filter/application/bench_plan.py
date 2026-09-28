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
) -> int:
    text_of = {sha256_text(i.sentence): i.sentence for i in items}
    ids = {sha: encode(render_prompt(template, text)) for sha, text in text_of.items()}
    texts = [UniqueText(sha, len(ids[sha]), (-1, 0)) for sha in text_of]
    chunks = plan_chunks(texts, fp, chunk_size)
    existing = {row["chunk_id"] for row in store.read_jsonl(f"plans/{BENCH}/{fp}/chunks.jsonl")}
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
                f"plans/{BENCH}/{fp}/chunks.jsonl",
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
        f"plans/{BENCH}/{fp}/items.json", {i.item_id: sha256_text(i.sentence) for i in items}
    )
    return len(chunks)
