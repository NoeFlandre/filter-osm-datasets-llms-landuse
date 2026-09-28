"""Paired, language-stratified non-inferiority test for serving-config changes.

The benchmark's F1 and MCC leave failed items out of their cells, so a config that
fails *more* can raise them; macro-accuracy (failed = wrong) closes that loophole.

Pre-registered margins (ADR-0006, approved 2026-09-28): the one-sided 95% lower bound
of candidate - reference macro-F1, macro-MCC and macro-accuracy must exceed -0.01; the failed-rate
increase upper bound must stay under +0.5 pp; no language may lose more than 0.05 F1.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

F1_MARGIN = 0.01
MCC_MARGIN = 0.01
ACCURACY_MARGIN = 0.01
FAILED_RATE_MARGIN = 0.005
LANGUAGE_F1_DROP = 0.05
RESAMPLES = 10_000
SEED = 0
ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class PairedItem:
    language: str
    expected: str
    reference: str | None
    candidate: str | None


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    delta_f1: float
    delta_mcc: float
    delta_failed_rate: float
    delta_accuracy: float
    f1_lower: float
    mcc_lower: float
    accuracy_lower: float
    failed_rate_upper: float
    agreement: float
    worst_language: str
    worst_language_delta_f1: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _codes(expected: np.ndarray, predicted: Sequence[str | None]) -> tuple[np.ndarray, ...]:
    pred_yes = np.array([p == "yes" for p in predicted])
    pred_no = np.array([p == "no" for p in predicted])
    failed = ~(pred_yes | pred_no)
    return (
        expected & pred_yes,
        expected & pred_no,
        ~expected & pred_no,
        ~expected & pred_yes,
        failed,
    )


def _metrics(cells: tuple[np.ndarray, ...], idx: np.ndarray) -> tuple[np.ndarray, ...]:
    """F1, MCC, failed rate and accuracy for each resample row of ``idx`` (shape B x n)."""
    tp, fn, tn, fp, failed = (c[idx].sum(axis=-1).astype(float) for c in cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    n = idx.shape[-1]
    return f1, mcc, failed / n, (tp + tn) / n


def _by_language(items: Sequence[PairedItem]) -> Mapping[str, list[PairedItem]]:
    groups: dict[str, list[PairedItem]] = {}
    for item in items:
        groups.setdefault(item.language, []).append(item)
    return dict(sorted(groups.items()))


def _deltas(
    groups: Mapping[str, list[PairedItem]], resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, float]]:
    """Macro point deltas, bootstrap deltas (4 x B) and the worst language by F1."""
    rng = np.random.default_rng(seed)
    boot = np.zeros((4, resamples))
    point = np.zeros(4)
    worst = ("", np.inf)
    for language, group in groups.items():
        expected = np.array([g.expected == "yes" for g in group])
        ref = _codes(expected, [g.reference for g in group])
        cand = _codes(expected, [g.candidate for g in group])
        full = np.arange(len(group))[None, :]
        delta = np.stack(_metrics(cand, full)) - np.stack(_metrics(ref, full))
        point += delta[:, 0]
        if delta[0, 0] < worst[1]:
            worst = (language, float(delta[0, 0]))
        idx = rng.integers(0, len(group), size=(resamples, len(group)))
        boot += np.stack(_metrics(cand, idx)) - np.stack(_metrics(ref, idx))
    return point / len(groups), boot / len(groups), worst


def _reasons(lower: np.ndarray, upper: np.ndarray, worst: tuple[str, float]) -> list[str]:
    checks = [
        (lower[0] <= -F1_MARGIN, f"macro-F1 lower bound {lower[0]:.4f} <= -{F1_MARGIN}"),
        (lower[1] <= -MCC_MARGIN, f"macro-MCC lower bound {lower[1]:.4f} <= -{MCC_MARGIN}"),
        (
            lower[3] <= -ACCURACY_MARGIN,
            f"macro-accuracy lower bound {lower[3]:.4f} <= -{ACCURACY_MARGIN}",
        ),
        (
            upper[2] >= FAILED_RATE_MARGIN,
            f"failed-rate upper bound {upper[2]:.4f} >= +{FAILED_RATE_MARGIN}",
        ),
        (
            worst[1] < -LANGUAGE_F1_DROP,
            f"language {worst[0]} F1 drop {worst[1]:.4f} > {LANGUAGE_F1_DROP}",
        ),
    ]
    return [message for failed, message in checks if failed]


def evaluate_gate(
    items: Sequence[PairedItem], *, resamples: int = RESAMPLES, seed: int = SEED
) -> GateResult:
    """Compare candidate to reference predictions on the same items."""
    if not items:
        raise ValueError("no paired items")
    point, boot, worst = _deltas(_by_language(items), resamples, seed)
    lower = np.quantile(boot, ALPHA, axis=1)
    upper = np.quantile(boot, 1 - ALPHA, axis=1)
    reasons = _reasons(lower, upper, worst)
    return GateResult(
        passed=not reasons,
        delta_f1=float(point[0]),
        delta_mcc=float(point[1]),
        delta_failed_rate=float(point[2]),
        delta_accuracy=float(point[3]),
        f1_lower=float(lower[0]),
        mcc_lower=float(lower[1]),
        accuracy_lower=float(lower[3]),
        failed_rate_upper=float(upper[2]),
        agreement=sum(i.reference == i.candidate for i in items) / len(items),
        worst_language=worst[0],
        worst_language_delta_f1=worst[1],
        reasons=tuple(reasons),
    )
