"""Deterministic partition of unique texts into small, idempotent work chunks."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from landuse_filter.domain.hashing import sha256_parts


@dataclass(frozen=True, slots=True)
class UniqueText:
    """A distinct sentence text to classify; ``order`` sets its planning priority."""

    text_sha256: str
    prompt_tokens: int
    order: tuple[int, int]  # (dataset rank, first input-file index)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    order: tuple[int, int]
    text_sha256s: tuple[str, ...]


def chunk_id(config_fp: str, shas: Iterable[str]) -> str:
    return sha256_parts(config_fp, *sorted(shas))[:24]


def plan_chunks(texts: Sequence[UniqueText], config_fp: str, chunk_size: int) -> list[Chunk]:
    """Group texts in priority order; each chunk is sorted by prompt length.

    The chunk boundaries depend only on the set of texts and their order keys, so the
    same inputs always produce the same chunk ids (resumable replanning).
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ordered = sorted(texts, key=lambda t: (t.order, t.text_sha256))
    chunks = []
    for start in range(0, len(ordered), chunk_size):
        group = sorted(
            ordered[start : start + chunk_size], key=lambda t: (t.prompt_tokens, t.text_sha256)
        )
        shas = tuple(t.text_sha256 for t in group)
        chunks.append(Chunk(chunk_id(config_fp, shas), group[0].order, shas))
    return chunks
