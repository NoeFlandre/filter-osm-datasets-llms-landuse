from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain import geomap

record = st.dictionaries(
    st.sampled_from(["a", "b", "c"]),
    st.tuples(st.integers(0, 50), st.integers(0, 50)).map(list),
)


def test_bbox_centre():
    assert geomap.bbox_centre(0.0, 10.0, 4.0, 20.0) == (2.0, 15.0)


def test_merge_sums_counts_per_cell():
    merged = geomap.merge([{"a": [1, 2], "b": [3, 0]}, {"a": [4, 4]}])
    assert merged == {"a": (5, 6), "b": (3, 0)}


@given(st.lists(record, max_size=6))
def test_merge_preserves_totals_and_ignores_order(records):
    merged = geomap.merge(records)
    assert geomap.sentences(merged) == sum(y + n for r in records for y, n in r.values())
    assert merged == geomap.merge(list(reversed(records)))


def test_global_share_and_empty_map():
    assert geomap.global_share({"a": (3, 1), "b": (1, 3)}) == 0.5
    assert geomap.global_share({}) == 0.0


def test_cell_share_needs_enough_sentences():
    assert geomap.cell_share((geomap.MIN_SENTENCES - 1, 0)) is None
    assert geomap.cell_share((6, geomap.MIN_SENTENCES - 6)) == 0.6
