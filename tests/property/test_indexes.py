import json

from hypothesis import given, settings
from hypothesis import strategies as st

from landuse_filter.adapters.indexes import ProgressIndex, ResolutionIndex
from landuse_filter.domain.parsing import parse_generation

shas = st.sampled_from([f"s{i}" for i in range(12)])
outputs = st.sampled_from(["x</think>yes", "x</think>no", "unclosed", "x</think>maybe"])


@settings(max_examples=40)
@given(st.lists(st.tuples(st.sampled_from(["c1", "c2"]), st.lists(shas, max_size=6)), max_size=8))
def test_progress_index_counts_distinct_hashes_like_a_set_union(tmp_path_factory, parts):
    root = tmp_path_factory.mktemp("w")
    expected: dict[str, set[str]] = {}
    paths = []
    for i, (chunk, hashes) in enumerate(parts):
        path = root / "parts" / "fp" / chunk / f"p{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text_sha256s": hashes}))
        paths.append(path)
        expected.setdefault(chunk, set()).update(hashes)
    index = ProgressIndex(root / "idx.sqlite")
    assert index.ingest(paths) == len(paths)
    assert index.ingest(paths) == 0  # idempotent
    for chunk in ("c1", "c2"):
        assert index.count(chunk) == len(expected.get(chunk, set()))


@settings(max_examples=40)
@given(st.lists(st.tuples(shas, outputs, st.booleans()), max_size=20))
def test_resolution_index_keeps_the_first_generation_per_hash(tmp_path_factory, rows):
    index = ResolutionIndex(tmp_path_factory.mktemp("r") / "idx.sqlite")
    index.build(iter([{"text_sha256": s, "raw_output": o, "truncated": t} for s, o, t in rows]))
    first: dict[str, tuple] = {}
    for s, o, t in rows:
        if s not in first:
            v = parse_generation(o, truncated=t)
            first[s] = (v.decision.value, v.mode and v.mode.value, v.failure and v.failure.value)
    for s in {r[0] for r in rows}:
        assert index.get(s) == first[s]
    assert index.get("missing") is None
