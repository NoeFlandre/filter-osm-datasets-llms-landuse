from datetime import timedelta

from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.launch_plan import by_site, plan_launches
from landuse_filter.domain.scheduling import Slot

SITES = ("a", "b", "c")
PENDING = [(f"c{i}", 10) for i in range(10)]


def slot(site="a", cluster="x", free=2):
    return Slot(site, cluster, "g", 1, free, timedelta(0), timedelta(hours=1), None, 1.0)


def plan(items, pending, taken=(), **options):
    settings = {
        "max_total": 99,
        "max_per_site": 99,
        "total": 0,
        "per_site": {},
        "capacity": lambda s: 10.0,
        "overflow": 1.0,
        **options,
    }
    return plan_launches(items, pending, set(taken), **settings)


def test_one_launch_per_free_node_with_disjoint_chunks_in_ranking_order():
    p = plan([(slot("a", free=2), "A"), (slot("b", free=1), "B")], PENDING)
    assert [(x.index, x.slot.site, x.payload, x.chunks) for x in p] == [
        (0, "a", "A", ("c0",)),
        (1, "a", "A", ("c1",)),
        (2, "b", "B", ("c2",)),
    ]


def test_taken_chunks_are_skipped_and_not_modified():
    taken = {"c0"}
    p = plan([(slot(free=1), None)], PENDING, taken)
    assert p[0].chunks == ("c1",)
    assert taken == {"c0"}


def test_total_cap_counts_existing_jobs_and_is_exact():
    items = [(slot("a", free=5), None)]
    assert len(plan(items, PENDING, max_total=3, total=1)) == 2
    assert len(plan(items, PENDING, max_total=3, total=3)) == 0
    assert len(plan(items, PENDING, max_total=3, total=2)) == 1


def test_per_site_cap_counts_existing_jobs_and_is_exact():
    items = [(slot("a", free=5), None), (slot("b", free=5), None)]
    p = plan(items, PENDING, max_per_site=2, per_site={"a": 1})
    assert [x.slot.site for x in p] == ["a", "b", "b"]


def test_capacity_is_per_slot_and_scaled_by_the_overflow():
    items = [(slot("a", "x", 1), None), (slot("b", "y", 1), None)]
    p = plan(
        items,
        PENDING,
        overflow=2.0,
        capacity=lambda s: 10.0 if s.cluster == "x" else 30.0,
    )
    assert [len(x.chunks) for x in p] == [2, 6]  # budgets 20 and 60 for chunks of 10


def test_a_site_without_jobs_gets_exactly_the_per_site_cap():
    items = [(slot("a", free=5), None), (slot("b", free=5), None)]
    for cap in (1, 2, 3):
        p = plan(items, PENDING, max_per_site=cap)
        assert [x.slot.site for x in p] == ["a"] * cap + ["b"] * cap


def test_the_callers_counters_are_not_modified():
    per_site = {"a": 1}
    plan([(slot("a", free=2), None)], PENDING, per_site=per_site)
    assert per_site == {"a": 1}


def test_stops_when_no_chunk_is_left():
    p = plan([(slot("a", free=5), None), (slot("b", free=5), None)], PENDING[:2])
    assert [x.chunks for x in p] == [("c0",), ("c1",)]


def test_by_site_keeps_plan_order_per_site():
    items = [(slot("a", free=2), None), (slot("b", free=1), None), (slot("a", "y", 1), None)]
    grouped = by_site(plan(items, PENDING))
    assert list(grouped) == ["a", "b"]
    assert [x.index for x in grouped["a"]] == [0, 1, 3]


@given(
    st.lists(st.tuples(st.sampled_from(SITES), st.integers(0, 4)), max_size=6),
    st.integers(0, 12),
    st.integers(0, 5),
    st.integers(0, 8),
    st.sets(st.sampled_from([f"c{i}" for i in range(10)])),
)
def test_plan_never_assigns_a_chunk_twice_respects_caps_and_is_deterministic(
    sites, max_total, max_per_site, n_pending, taken
):
    items = [(slot(s, f"k{i}", free), i) for i, (s, free) in enumerate(sites)]
    pending = PENDING[:n_pending]
    kwargs = {"taken": taken, "max_total": max_total, "max_per_site": max_per_site}
    first = plan(items, pending, **kwargs)
    assert first == plan(items, pending, **kwargs)
    chunks = [c for x in first for c in x.chunks]
    assert len(chunks) == len(set(chunks))
    assert not set(chunks) & taken
    assert len(first) <= max_total
    assert all(len(v) <= max_per_site for v in by_site(first).values())
    assert [x.index for x in first] == list(range(len(first)))
