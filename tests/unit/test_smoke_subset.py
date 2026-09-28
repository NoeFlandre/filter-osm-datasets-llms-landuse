from landuse_filter.adapters.benchmark import BenchmarkItem
from landuse_filter.application.bench_plan import smoke_fp, smoke_items


def items():
    return [
        BenchmarkItem(f"{lang}{i:03d}", lang, f"s{lang}{i}", "yes" if i % 3 else "no")
        for lang in ("en", "fr")
        for i in range(60)
    ]


def test_subset_is_fixed_stratified_and_sized():
    a, b = smoke_items(items()), smoke_items(list(reversed(items())))
    assert a == b
    assert len(a) == 40
    for lang in ("en", "fr"):
        labels = [i.label for i in a if i.language == lang]
        assert labels.count("yes") == labels.count("no") == 10


def test_smoke_namespace_is_separate():
    assert smoke_fp("fp", "l40s") == "fp-smoke-l40s"
