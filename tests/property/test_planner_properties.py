"""Planner.emit / emit_uniform / release_unfinished: plan-once and keep-finished properties."""

import hashlib
from collections import Counter

import pyarrow.parquet as pq
from hypothesis import given, settings
from hypothesis import strategies as st

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Planner

TEMPLATE = "S: {}"
PLAN = f"plans/{WEBSITE}/fp/chunks.jsonl"


def encode(prompts):
    return [[1, 2, 3] for _ in prompts]


def sha(i: int) -> str:
    return hashlib.sha256(str(i).encode()).hexdigest()


def seed(tmp_path, cells: list[str | None], *, scanned: bool = False) -> tuple[Planner, list[str]]:
    """A planner whose index holds one text per entry of ``cells`` (``None``: unlocated)."""
    p = Planner(WorkStore(tmp_path), WEBSITE, "fp")
    p.register(["polygons/a.parquet"])
    rows = [(sha(i), f"text-{i}", 0) for i in range(len(cells))]
    with p.db:
        p.db.executemany("INSERT INTO texts (sha, text, file_idx) VALUES (?, ?, ?)", rows)
        if scanned:
            p.db.execute("UPDATE files SET done = 1")
    p.set_cells((sha(i), c) for i, c in enumerate(cells) if c)
    return p, [r[0] for r in rows]


def planned(tmp_path) -> list[list[str]]:
    store = WorkStore(tmp_path)
    return [
        pq.read_table(store.path(f"chunks/{r['chunk_id']}.parquet"), columns=["text_sha256"])
        .column("text_sha256")
        .to_pylist()
        for r in store.read_jsonl(PLAN)
    ]


cell_lists = st.lists(st.sampled_from(["A", "B", "C", None]), min_size=1, max_size=40)
chunk_sizes = st.integers(1, 7)


@settings(max_examples=40, deadline=None)
@given(cell_lists, chunk_sizes)
def test_emit_uniform_plans_every_text_exactly_once(tmp_path_factory, cells, size):
    tmp = tmp_path_factory.mktemp("uniform")
    p, shas = seed(tmp, cells)
    p.emit_uniform(encode, TEMPLATE, size)
    flat = [s for c in planned(tmp) for s in c]
    assert Counter(flat) == Counter(shas)
    assert all(len(c) <= size for c in planned(tmp))
    assert p.emit_uniform(encode, TEMPLATE, size) == 0  # nothing left to plan


@settings(max_examples=40, deadline=None)
@given(cell_lists, chunk_sizes, st.booleans())
def test_emit_plans_each_text_once_and_holds_back_a_partial_tail(
    tmp_path_factory, cells, size, final
):
    tmp = tmp_path_factory.mktemp("emit")
    p, shas = seed(tmp, cells, scanned=True)
    p.emit(encode, TEMPLATE, size, final=final)
    flat = [s for c in planned(tmp) for s in c]
    assert len(flat) == len(set(flat))
    assert set(flat) <= set(shas)
    assert len(flat) == (len(shas) if final else len(shas) - len(shas) % size)
    p.emit(encode, TEMPLATE, size, final=True)  # a later final pass plans the rest, once
    assert Counter(s for c in planned(tmp) for s in c) == Counter(shas)


@settings(max_examples=40, deadline=None)
@given(cell_lists, chunk_sizes, st.data())
def test_replan_keeps_finished_chunks_and_plans_the_rest_once(tmp_path_factory, cells, size, data):
    tmp = tmp_path_factory.mktemp("replan")
    p, shas = seed(tmp, cells)
    p.emit_uniform(encode, TEMPLATE, size)
    first = planned(tmp)
    finished_idx = data.draw(st.sets(st.integers(0, len(first) - 1)))
    done = {s for i in finished_idx for s in first[i]}
    # texts of unfinished chunks may also be generated individually
    extra = data.draw(st.sets(st.sampled_from(shas)))
    p.mark_done(done | extra)
    fully_done = [c for c in first if set(c) <= done | extra]

    released = p.release_unfinished()
    assert len(released) == len(first) - len(fully_done)
    assert planned(tmp) == fully_done  # finished chunks stay, in their original order

    p.emit_uniform(encode, TEMPLATE, size)
    flat = [s for c in planned(tmp) for s in c]
    pending = set(shas) - done - extra
    kept = {s for c in fully_done for s in c}
    assert len(flat) == len(set(flat))  # nothing planned twice
    assert pending <= set(flat)  # every pending text is planned
    assert set(flat) == kept | pending
    assert planned(tmp)[: len(fully_done)] == fully_done
