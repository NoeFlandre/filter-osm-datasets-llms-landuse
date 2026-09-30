import hashlib
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from hypothesis import given, settings
from hypothesis import strategies as st

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Planner


def encode(prompts):
    return [[1, 2, 3] for _ in prompts]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seeded(tmp_path: Path, cells: dict[str, int], unlocated: int = 0) -> tuple[Planner, dict]:
    """A planner whose index holds ``n`` texts per cell (texts ``cell-i``), none emitted."""
    p = Planner(WorkStore(tmp_path), WEBSITE, "fp")
    p.register(["polygons/a.parquet"])
    where = {}
    rows = []
    for cell, n in cells.items():
        for i in range(n):
            where[sha(f"{cell}-{i}")] = cell
            rows.append((sha(f"{cell}-{i}"), f"{cell}-{i}", 0))
    for i in range(unlocated):
        where[sha(f"none-{i}")] = None
        rows.append((sha(f"none-{i}"), f"none-{i}", 0))
    with p.db:
        p.db.executemany("INSERT INTO texts (sha, text, file_idx) VALUES (?, ?, ?)", rows)
    p.set_cells((s, c) for s, c in where.items() if c)
    return p, where


def chunks_in_plan_order(tmp_path: Path) -> list[list[str]]:
    store = WorkStore(tmp_path)
    lines = store.read_jsonl(f"plans/{WEBSITE}/fp/chunks.jsonl")
    return [
        pq.read_table(store.path(f"chunks/{r['chunk_id']}.parquet"), columns=["text_sha256"])
        .column("text_sha256")
        .to_pylist()
        for r in lines
    ]


def test_the_first_chunks_take_an_equal_share_from_every_cell(tmp_path):
    p, where = seeded(tmp_path, {"A": 6, "B": 3, "C": 3})
    p.emit_uniform(encode, "S: {}", chunk_size=3)
    chunks = chunks_in_plan_order(tmp_path)
    shares = [Counter(where[s] for s in c) for c in chunks]
    assert shares[0] == shares[1] == shares[2] == Counter({"A": 1, "B": 1, "C": 1})
    assert shares[3] == Counter({"A": 3})  # only the biggest cell is left


def test_every_pending_text_is_planned_exactly_once(tmp_path):
    p, where = seeded(tmp_path, {"A": 7, "B": 2}, unlocated=5)
    p.emit_uniform(encode, "S: {}", chunk_size=4)
    planned = [s for c in chunks_in_plan_order(tmp_path) for s in c]
    assert sorted(planned) == sorted(where)


def test_unlocated_texts_are_spread_through_the_order(tmp_path):
    p, where = seeded(tmp_path, {"A": 30, "B": 30}, unlocated=30)
    p.emit_uniform(encode, "S: {}", chunk_size=10)
    order = [s for c in chunks_in_plan_order(tmp_path) for s in c]
    first_half = sum(where[s] is None for s in order[:45])
    assert 8 <= first_half <= 22  # neither all first nor all last (30 in total)


def test_released_chunks_and_done_texts_are_handled(tmp_path):
    p, where = seeded(tmp_path, {"A": 4, "B": 4})
    p.emit_uniform(encode, "S: {}", chunk_size=4)
    first = chunks_in_plan_order(tmp_path)
    done = set(first[0])  # one chunk fully generated, the other untouched
    p.mark_done(done)
    released = p.release_unfinished()
    assert len(released) == 1
    kept = chunks_in_plan_order(tmp_path)
    assert kept == [first[0]]  # the finished chunk stays, the open one left the plan
    p.emit_uniform(encode, "S: {}", chunk_size=4)
    planned = [s for c in chunks_in_plan_order(tmp_path) for s in c]
    assert sorted(planned) == sorted(where)  # the released texts are planned again


@settings(max_examples=40, deadline=None)
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=6),
    chunk=st.integers(min_value=1, max_value=5),
)
def test_any_prefix_is_balanced_across_cells_not_yet_exhausted(tmp_path_factory, sizes, chunk):
    tmp = tmp_path_factory.mktemp("geo")
    cells = {f"c{i}": n for i, n in enumerate(sizes)}
    p, where = seeded(tmp, cells)
    p.emit_uniform(encode, "S: {}", chunk_size=chunk)
    order = [s for c in chunks_in_plan_order(tmp) for s in c]
    total = Counter(where.values())
    for end in range(chunk, len(order) + 1, chunk):
        seen = Counter(where[s] for s in order[:end])
        live = [seen[c] for c in total if seen[c] < total[c]]
        # chunks are sorted by prompt length inside, so check whole-chunk prefixes: cells
        # that still have texts left differ by at most one round
        assert not live or max(live) - min(live) <= 1


def test_replan_job_keeps_finished_chunks_and_reorders_the_rest(tmp_path):
    import json

    from landuse_filter.adapters.remote import DirRemote
    from landuse_filter.application import remote_plan

    inputs = Path(__file__).parents[1] / "fixtures" / "inputs"
    remote = DirRemote(tmp_path / "bucket")
    first = WorkStore(tmp_path / "first-plan")
    planner = Planner(first, WEBSITE, "fp")
    planner.register(["polygons/a.parquet"])
    planner.scan(lambda _: inputs / "website.parquet")
    planner.emit(encode, "S: {}", chunk_size=10, final=True)
    lines = first.read_jsonl(f"plans/{WEBSITE}/fp/chunks.jsonl")
    finished = pq.read_table(first.path(f"chunks/{lines[0]['chunk_id']}.parquet")).column(
        "text_sha256"
    )
    remote_plan.checkpoint(remote, planner, first, WEBSITE)
    remote_plan.publish_new_chunks(remote, first, WEBSITE, "fp")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"text_sha256s": finished.to_pylist()}))
    remote.put([(manifest, f"parts/fp/{lines[0]['chunk_id']}/p1.json")])

    cells = {}

    def locate(dataset, path, local):
        for i, (sha_, *_rest) in enumerate(
            planner.db.execute("SELECT sha FROM texts ORDER BY sha").fetchall()
        ):
            cells[sha_] = "A" if i % 2 else "B"
        return list(cells.items())

    report = remote_plan.run_replan(
        remote,
        WorkStore(tmp_path / "replan-node"),
        WEBSITE,
        "fp",
        remote_plan.PlanInputs(
            files=["polygons/a.parquet"],
            fetch=lambda _: inputs / "website.parquet",
            encode=encode,
            template="S: {}",
            chunk_size=10,
        ),
        locate=locate,
    )
    new_lines = WorkStore(tmp_path / "check")
    remote.get(
        [(f"plans/{WEBSITE}/fp/chunks.jsonl", new_lines.path(f"plans/{WEBSITE}/fp/chunks.jsonl"))]
    )
    ids = [r["chunk_id"] for r in new_lines.read_jsonl(f"plans/{WEBSITE}/fp/chunks.jsonl")]
    assert ids[0] == lines[0]["chunk_id"]  # the finished chunk keeps its place
    assert report["released_chunks"] == len(lines) - 1
    total_unique = planner.report().unique_texts
    assert report["replanned_texts"] == total_unique - len(finished)


def test_locator_places_website_texts_from_the_input_file_itself():
    from landuse_filter.application.locate import locator

    inputs = Path(__file__).parents[1] / "fixtures" / "inputs"
    pairs = list(
        locator(
            WEBSITE, fetch=lambda _p: inputs, cell_of=lambda lon, lat: f"{round(lon)}/{round(lat)}"
        )(WEBSITE, "polygons/a.parquet", inputs / "website.parquet")
    )
    assert pairs
    assert all("/" in cell for _, cell in pairs)


def test_locator_refuses_a_dataset_without_coordinates():
    import pytest

    from landuse_filter.application.locate import locator

    with pytest.raises(ValueError, match="no coordinates"):
        locator("osm-polygon-description-tag", fetch=lambda _p: Path(), cell_of=lambda *_: "x")
