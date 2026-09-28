"""Offline simulation of a lower ``max_new_tokens`` budget.

Greedy decoding is prefix-deterministic: a generation that finished within ``cap``
tokens is identical under that cap, and one that did not is truncated (-> failed).
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Generated:
    predicted: str | None
    generated_tokens: int


def predicted_under_cap(item: Generated, cap: int) -> str | None:
    return None if item.generated_tokens > cap else item.predicted


def truncation_rate(items: Sequence[Generated], cap: int) -> float:
    if not items:
        raise ValueError("no generations")
    return sum(i.generated_tokens > cap for i in items) / len(items)
