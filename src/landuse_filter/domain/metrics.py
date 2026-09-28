"""Binary metrics identical to the benchmark's (positive = yes; failed is an error).

A failed prediction is its own bucket: it lowers accuracy but, as in the benchmark,
is outside the precision/recall/MCC confusion cells.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

Outcome = tuple[str, str | None]  # (expected yes/no, predicted yes/no/None)


@dataclass(frozen=True, slots=True)
class Confusion:
    tp: int
    fn: int
    tn: int
    fp: int
    failed: int

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.tn + self.fp + self.failed


@dataclass(frozen=True, slots=True)
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    mcc: float
    failed_rate: float


def confusion(outcomes: Sequence[Outcome]) -> Confusion:
    counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "failed": 0}
    for expected, predicted in outcomes:
        counts[_bucket(expected, predicted)] += 1
    return Confusion(**counts)


def _bucket(expected: str, predicted: str | None) -> str:
    if predicted is None:
        return "failed"
    if predicted == "yes":
        return "tp" if expected == "yes" else "fp"
    return "fn" if expected == "yes" else "tn"


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def scores(c: Confusion) -> Scores:
    if not c.total:
        raise ValueError("cannot score an empty set of outcomes")
    precision = _ratio(c.tp, c.tp + c.fp)
    recall = _ratio(c.tp, c.tp + c.fn)
    den = sqrt(float((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)))
    return Scores(
        accuracy=_ratio(c.tp + c.tn, c.total),
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        balanced_accuracy=(_ratio(c.tp, c.tp + c.fn) + _ratio(c.tn, c.tn + c.fp)) / 2,
        mcc=_ratio(float(c.tp * c.tn - c.fp * c.fn), den),
        failed_rate=_ratio(c.failed, c.total),
    )
