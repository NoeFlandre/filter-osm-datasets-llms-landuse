import pytest

from landuse_filter.domain.completion import Part, State, progress
from landuse_filter.domain.planning import UniqueText, plan_chunks


def texts(n):
    return [UniqueText(f"{i:04x}", prompt_tokens=n - i, order=(i % 2, 0)) for i in range(n)]


def test_chunks_follow_priority_and_are_length_sorted():
    chunks = plan_chunks(texts(6), "fp", chunk_size=2)
    assert [c.order[0] for c in chunks] == [0, 0, 0, 1, 1, 1][::2]
    assert all(len(c.text_sha256s) == 2 for c in chunks)


def test_bad_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        plan_chunks(texts(2), "fp", 0)


def test_chunk_ids_depend_on_config():
    assert plan_chunks(texts(4), "a", 2)[0].chunk_id != plan_chunks(texts(4), "b", 2)[0].chunk_id


def test_progress_states():
    assert progress(["a", "b"], []).state is State.PENDING
    p = progress(["a", "b"], [Part("p2", ("a",)), Part("p1", ("a", "z"))])
    assert p.state is State.PARTIAL
    assert p.missing == ("b",)
    assert p.owner == {"a": "p1"}
    assert p.foreign == ("z",)
    assert progress(["a"], [Part("p", ("a",))]).state is State.COMPLETE


def test_chunk_id_is_truncated_hash_of_sorted_shas():
    from landuse_filter.domain.hashing import sha256_parts
    from landuse_filter.domain.planning import chunk_id

    assert chunk_id("fp", ["b", "a"]) == sha256_parts("fp", "a", "b")[:24]


def test_bad_chunk_size_message():
    with pytest.raises(ValueError, match=r"^chunk_size must be positive$"):
        plan_chunks(texts(2), "fp", 0)
