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


def test_status_reports_eta_and_alerts(tmp_path):
    from landuse_filter.config import GENERATION_FP

    store = WorkStore(tmp_path)
    store.append_jsonl(
        f"plans/d/{GENERATION_FP}/chunks.jsonl",
        [{"chunk_id": "a", "size": 3600}, {"chunk_id": "b", "size": 10}],
    )
    store.append_jsonl("complete.jsonl", [{"chunk_id": "b"}])
    store.write_json(
        "assignments/x.json",
        {"state": "submitted", "fp": GENERATION_FP, "gpu": "NVIDIA L40S", "chunks": ["a"]},
    )
    store.write_json("profiles/l40s.json", {"gpu": "l40s", "sentences_per_second": 2.0})
    store.write_json(
        "jobs/s/1.json",
        {"gpu": "L40S", "sentences_per_second": 2.0, "completed": 100, "failed": 20},
    )
    report = summarize(store, ["d"])
    assert report["d"]["texts_complete"] == 10
    assert report["d"]["eta_hours"] == 0.5
    assert report["alerts"]
