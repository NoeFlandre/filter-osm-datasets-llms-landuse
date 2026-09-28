from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.cleanup import Entry, Keep, cleanup_plan

KEEP = Keep(frozenset({"live"}), "abc", frozenset({"fp/c1/p1.parquet"}))


def test_plan_removes_only_stale_project_files():
    entries = [
        Entry("luf/code/old", 1),
        Entry("luf/code/live", 1),
        Entry("luf/cache/venv-zzz", 1),
        Entry("luf/cache/venv-abc", 1),
        Entry("luf/cache/venv-abc.lock", 1),
        Entry("luf/cache/hf", 1),
        Entry("luf/hf_token", 100),
        Entry("luf/logs/1.out", 8),
        Entry("luf/logs/2.out", 1),
        Entry("luf/work/parts/fp/c1/p1.parquet", 1),
        Entry("luf/work/parts/fp/c1/p2.parquet", 1),
        Entry("luf/work/assignments/a.json", 30),
        Entry("geoparser-venv", 99),
    ]
    assert cleanup_plan(entries, KEEP) == [
        "luf/cache/venv-zzz",
        "luf/code/old",
        "luf/logs/1.out",
        "luf/work/parts/fp/c1/p1.parquet",
    ]


@given(st.lists(st.builds(Entry, st.text(max_size=30), st.floats(0, 100))))
def test_never_leaves_the_project_tree_or_touches_the_token(entries):
    plan = cleanup_plan(entries, KEEP)
    assert all(p.startswith("luf/") for p in plan)
    assert "luf/hf_token" not in plan


def test_logs_are_kept_up_to_exactly_seven_days():
    assert cleanup_plan([Entry("luf/logs/a.out", 7.0)], KEEP) == []
    assert cleanup_plan([Entry("luf/logs/a.out", 7.01)], KEEP) == ["luf/logs/a.out"]
