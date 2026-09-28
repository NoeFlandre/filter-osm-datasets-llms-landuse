from pathlib import Path

import pytest

from landuse_filter.adapters.benchmark import read_items, read_reference
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.bench import compare, macro_scores, simulate_budgets
from landuse_filter.application.bench_plan import plan_benchmark
from landuse_filter.application.status import summarize

ROOT = Path(__file__).parents[1] / "fixtures" / "benchmark"


def test_reads_items_and_reference():
    items = read_items(ROOT)
    reference = list(read_reference(ROOT))
    assert len(items) == len(reference) == 16
    assert {r.language for r in reference} == {"en", "fr"}


def test_budget_simulation_at_full_budget_is_identity():
    reference = list(read_reference(ROOT))
    (row,) = simulate_budgets(reference, [10_000], resamples=50)
    assert row.truncation_rate == 0
    assert row.gate.agreement == 1.0


def test_compare_requires_every_item():
    reference = list(read_reference(ROOT))
    with pytest.raises(ValueError, match="lacks"):
        compare(reference, {}, resamples=10)
    same = compare(reference, {r.item_id: r.predicted for r in reference}, resamples=50)
    assert same.passed


def test_macro_scores_average_languages():
    scores = macro_scores(list(read_reference(ROOT)))
    assert 0 <= scores["f1"] <= 1


def test_plan_benchmark_is_idempotent_and_tracked_by_status(tmp_path):
    store = WorkStore(tmp_path)
    items = read_items(ROOT)
    kw = {"template": "S: {}", "fp": "fp", "chunk_size": 5}
    n = plan_benchmark(store, items, lambda ps: [[len(p)] for p in ps], **kw)
    assert plan_benchmark(store, items, lambda ps: [[len(p)] for p in ps], **kw) == n
    lines = store.read_jsonl("plans/benchmark/fp/chunks.jsonl")
    assert len(lines) == n
    assert sum(r["size"] for r in lines) == len({i.sentence for i in items})
    report = summarize(store, ["benchmark"])
    assert "benchmark" in report
