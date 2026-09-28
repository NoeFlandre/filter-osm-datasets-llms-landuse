"""Benchmark use cases: offline budget simulation and candidate-vs-reference gating."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from landuse_filter.adapters.benchmark import ReferencePrediction
from landuse_filter.domain.budget import Generated, predicted_under_cap, truncation_rate
from landuse_filter.domain.metrics import confusion, scores
from landuse_filter.domain.noninferiority import GateResult, PairedItem, evaluate_gate


@dataclass(frozen=True, slots=True)
class BudgetRow:
    cap: int
    truncation_rate: float
    gate: GateResult


def simulate_budgets(
    reference: Sequence[ReferencePrediction], caps: Sequence[int], resamples: int
) -> list[BudgetRow]:
    rows = []
    for cap in caps:
        paired = [
            PairedItem(r.language, r.expected, r.predicted,
                       predicted_under_cap(Generated(r.predicted, r.generated_tokens), cap))
            for r in reference
        ]
        generated = [Generated(r.predicted, r.generated_tokens) for r in reference]
        rows.append(BudgetRow(cap, truncation_rate(generated, cap), evaluate_gate(paired, resamples=resamples)))
    return rows


def macro_scores(reference: Sequence[ReferencePrediction]) -> Mapping[str, float]:
    """Language-macro scores, the benchmark's headline aggregation."""
    by_language: dict[str, list[tuple[str, str | None]]] = {}
    for r in reference:
        by_language.setdefault(r.language, []).append((r.expected, r.predicted))
    per = [asdict(scores(confusion(v))) for v in by_language.values()]
    return {k: sum(p[k] for p in per) / len(per) for k in per[0]}


def compare(
    reference: Sequence[ReferencePrediction], candidate: Mapping[str, str | None], resamples: int
) -> GateResult:
    """Gate candidate predictions (item id -> yes/no/None) against the reference run."""
    missing = [r.item_id for r in reference if r.item_id not in candidate]
    if missing:
        raise ValueError(f"candidate lacks {len(missing)} benchmark items, e.g. {missing[:3]}")
    return evaluate_gate(
        [PairedItem(r.language, r.expected, r.predicted, candidate[r.item_id]) for r in reference],
        resamples=resamples,
    )
