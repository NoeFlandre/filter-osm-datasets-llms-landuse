"""Concurrency sweep on one GPU (issue #14): throughput only, outputs discarded.

Quality is never judged here: a profile found by calibration still has to pass the
benchmark gate in its own namespace before production uses it (ADR-0006).
"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

from landuse_filter.application.node import AsyncEngine


@dataclass(frozen=True, slots=True)
class Point:
    window: int
    sentences: int
    seconds: float
    generated_tokens: int

    @property
    def sentences_per_second(self) -> float:
        return self.sentences / self.seconds if self.seconds else 0.0


async def _measure(engine: AsyncEngine, prompts: Sequence[list[int]], window: int) -> Point:
    queue = iter(prompts)
    tokens = 0
    done = 0

    async def worker() -> None:
        nonlocal tokens, done
        for ids in queue:
            out = await engine.generate(ids)
            tokens += int(out.get("meta_info", {}).get("completion_tokens") or 0)
            done += 1

    start = time.monotonic()
    await asyncio.gather(*(worker() for _ in range(window)))
    return Point(window, done, time.monotonic() - start, tokens)


def sweep(engine: AsyncEngine, prompts: Sequence[list[int]], windows: Sequence[int]) -> list[Point]:
    """Same prompts at each concurrency level; prompts should be >= 4x the largest window."""
    return [asyncio.run(_measure(engine, prompts, w)) for w in windows]


def best(points: Sequence[Point], tolerance: float = 0.03) -> Point:
    """Smallest window within ``tolerance`` of the best throughput (less memory risk)."""
    top = max(p.sentences_per_second for p in points)
    return min(
        (p for p in points if p.sentences_per_second >= top * (1 - tolerance)),
        key=lambda p: p.window,
    )
