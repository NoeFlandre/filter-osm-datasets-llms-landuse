from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.completion import Part, State, progress
from landuse_filter.domain.planning import UniqueText, plan_chunks

unique_texts = st.lists(
    st.builds(
        UniqueText,
        st.text("0123456789abcdef", min_size=8, max_size=8),
        st.integers(1, 500),
        st.tuples(st.integers(0, 2), st.integers(0, 5)),
    ),
    unique_by=lambda t: t.text_sha256,
    max_size=60,
)


@given(unique_texts, st.integers(1, 10), st.randoms())
def test_plan_partitions_and_is_permutation_invariant(items, size, rnd):
    chunks = plan_chunks(items, "fp", size)
    flat = [s for c in chunks for s in c.text_sha256s]
    assert sorted(flat) == sorted(t.text_sha256 for t in items)
    shuffled = list(items)
    rnd.shuffle(shuffled)
    assert plan_chunks(shuffled, "fp", size) == chunks


@given(st.lists(st.text(min_size=1, max_size=3), unique=True, min_size=1, max_size=20), st.data())
def test_merge_is_order_independent(expected, data):
    parts = data.draw(
        st.lists(
            st.builds(
                Part,
                st.text(min_size=1, max_size=4),
                st.lists(st.sampled_from(expected)).map(tuple),
            ),
            max_size=6,
        )
    )
    a = progress(expected, parts)
    b = progress(expected, list(reversed(parts)) + parts)  # reordered and duplicated
    assert a == b
    assert (a.state is State.COMPLETE) == (not a.missing)
