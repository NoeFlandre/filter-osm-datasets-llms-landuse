"""Exact-value tests that pin the Planner's index, plan lines and uniform order."""

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import huggingface_hub
import pyarrow.parquet as pq
import pytest

from landuse_filter.adapters.readers import SOURCES, WEBSITE
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import plan
from landuse_filter.application.plan import Planner, PlanReport
from landuse_filter.domain.sentences import SentenceRef

RANK = 2  # the website dataset's rank


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encode(prompts):
    return [[1, 2, 3] for _ in prompts]


def ref(text, *, unsplit=False):
    return SentenceRef(WEBSITE, "f", (("k", text),), text, unsplit=unsplit)


class Script:
    """A fake source: per input path, the refs it yields; records every read call."""

    def __init__(self, files):
        self.files = files
        self.reads = []

    def __call__(self, local, path):
        self.reads.append((local, path))
        yield from self.files[path]


def make(tmp_path, files=None):
    p = Planner(WorkStore(tmp_path), WEBSITE, "fp")
    script = Script(files or {})
    p.source = replace(SOURCES[WEBSITE], read=script)
    p.register(list(files or {}))
    return p, script


def test_cell_key_is_a_truncated_hash_and_none_means_empty():
    assert plan._cell_key("A") == sha("A")[:16]
    assert plan._cell_key(None) == sha("")[:16]
    assert plan._cell_key("") == sha("")[:16]


def test_scan_reads_each_file_with_its_fetched_path_and_counts_per_file(tmp_path):
    files = {
        "a": [ref("u1", unsplit=True), ref("x"), ref("y"), ref("x"), ref("u2", unsplit=True)],
        "b": [ref("y"), ref("z")],
    }
    p, script = make(tmp_path, files)
    fetched = []

    def fetch(path):
        fetched.append(path)
        return Path(f"local-{path}")

    assert p.scan(fetch, limit=None) == 2
    assert fetched == ["a", "b"]
    assert script.reads == [(Path("local-a"), "a"), (Path("local-b"), "b")]
    rows = p.db.execute(
        "SELECT idx, path, done, sentences, unsplit, new_unique FROM files ORDER BY idx"
    ).fetchall()
    assert rows == [(0, "a", 1, 5, 2, 2), (1, "b", 1, 2, 0, 1)]
    assert p.report() == PlanReport(2, 2, 7, 2, 3, 0)


def test_scan_limit_stops_after_that_many_files(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x")], "b": [ref("y")], "c": [ref("z")]})
    assert p.scan(Path, limit=2) == 2
    assert p.scan(Path) == 1


def test_empty_index_reports_zeros(tmp_path):
    p, _ = make(tmp_path)
    assert p.report() == PlanReport(0, 0, 0, 0, 0, 0)


def test_emit_only_takes_texts_of_files_before_the_first_unscanned_one(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x"), ref("y")], "b": [], "c": []})
    p.scan(Path, limit=1)
    with p.db:  # a later file scanned out of order must not leak into the plan
        p.db.execute("INSERT INTO texts (sha, text, file_idx) VALUES ('late', 'late', 2)")
    assert p.emit(encode, "S: {}", chunk_size=1, final=True) == 2
    planned = {r[0] for r in p.db.execute("SELECT sha FROM texts WHERE chunk_id IS NOT NULL")}
    assert planned == {sha("x"), sha("y")}


def test_emit_counts_every_chunk_and_orders_by_file_then_sha(tmp_path):
    p, _ = make(tmp_path, {"a": [ref(t) for t in "pqrstu"]})
    p.scan(Path)
    assert p.emit(encode, "S: {}", chunk_size=2, final=False) == 3
    assert p.emit(encode, "S: {}", chunk_size=2, final=True) == 0


def test_plan_line_and_chunk_table_are_exact(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x"), ref("y")]})
    p.scan(Path)
    assert p.emit(encode, "S: {}", chunk_size=2, final=True) == 1
    store = WorkStore(tmp_path)
    (line,) = store.read_jsonl(p.plan_path)
    assert line == {
        "chunk_id": line["chunk_id"],
        "order": [RANK, 0],
        "size": 2,
        "prompt_tokens": 6,
    }
    table = store.read_chunk(line["chunk_id"])
    assert table.schema == CHUNK
    assert sorted(table.column("text").to_pylist()) == ["x", "y"]
    assert table.column("input_ids").to_pylist() == [[1, 2, 3], [1, 2, 3]]


def test_recover_plan_lines_rebuilds_lost_lines_exactly(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x")], "b": [ref("y")]})
    p.scan(Path)
    p.emit(encode, "S: {}", chunk_size=1, final=True)
    original = {r["chunk_id"]: r for r in WorkStore(tmp_path).read_jsonl(p.plan_path)}
    assert p.recover_plan_lines() == 0  # nothing lost
    p.store.path(p.plan_path).unlink()
    assert p.recover_plan_lines() == 2
    recovered = {r["chunk_id"]: r for r in WorkStore(tmp_path).read_jsonl(p.plan_path)}
    assert recovered.keys() == original.keys()
    assert sorted(
        (tuple(r["order"]), r["size"], r["prompt_tokens"]) for r in recovered.values()
    ) == [((RANK, 0), 1, 0), ((RANK, 1), 1, 0)]


def test_release_without_a_plan_file_still_frees_the_texts(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x")]})
    p.scan(Path)
    with p.db:
        p.db.execute("UPDATE texts SET chunk_id = 'lost'")
    assert p.release_unfinished() == {"lost"}
    assert p.db.execute("SELECT chunk_id FROM texts").fetchall() == [(None,)]


def test_release_with_nothing_unfinished_touches_nothing(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x")]})
    p.scan(Path)
    p.emit(encode, "S: {}", chunk_size=1, final=True)
    p.mark_done([sha("x")])
    assert p.release_unfinished() == set()
    assert len(WorkStore(tmp_path).read_jsonl(p.plan_path)) == 1


def test_first_location_wins_and_done_texts_are_left_alone(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x"), ref("y")]})
    p.scan(Path)
    p.set_cells([(sha("x"), "A"), (sha("y"), "B")])
    p.set_cells([(sha("x"), "C")])
    cells = dict(p.db.execute("SELECT sha, cell FROM texts"))
    assert cells == {sha("x"): "A", sha("y"): "B"}
    p.mark_done([sha("x")])
    assert dict(p.db.execute("SELECT sha, done FROM texts")) == {sha("x"): 1, sha("y"): 0}


def _plan_order(tmp_path) -> list[str]:
    store = WorkStore(tmp_path)
    order = []
    for line in store.read_jsonl(f"plans/{WEBSITE}/fp/chunks.jsonl"):
        order += store.read_chunk(line["chunk_id"]).column("text_sha256").to_pylist()
    return order


def test_uniform_order_is_rounds_then_cell_key_then_sha(tmp_path):
    p, _ = make(tmp_path, {"a": []})
    cells = {"A": 4, "B": 2, "C": 1}
    rows, located = [], {}
    for cell, n in cells.items():
        for i in range(n):
            text = f"{cell}-{i}"
            rows.append((sha(text), text, 0))
            located[sha(text)] = cell
    free = [f"free-{i}" for i in range(30)]
    rows += [(sha(t), t, 0) for t in free]
    with p.db:
        p.db.executemany("INSERT INTO texts (sha, text, file_idx) VALUES (?, ?, ?)", rows)
    p.set_cells(located.items())
    assert p.emit_uniform(encode, "S: {}", chunk_size=1) == len(rows)

    top = max(cells.values()) - 1
    keyed = []
    for cell in cells:
        for rnd, s in enumerate(sorted(s for s, c in located.items() if c == cell)):
            keyed.append((rnd, sha(cell)[:16], s))
    keyed += [(int(sha(t)[:12], 16) % (top + 1), "", sha(t)) for t in free]
    assert _plan_order(tmp_path) == [s for _, _, s in sorted(keyed)]


def test_emit_uniform_counts_each_chunk(tmp_path):
    p, _ = make(tmp_path, {"a": [ref(str(i)) for i in range(7)]})
    p.scan(Path)
    assert p.emit_uniform(encode, "S: {}", chunk_size=3) == 3


def test_list_input_files_filters_by_pattern_and_sorts(monkeypatch):
    calls = []

    class FakeApi:
        def list_repo_files(self, repo_id, *, repo_type, revision):
            calls.append((repo_id, repo_type, revision))
            return ["polygons/b.parquet", "README.md", "polygons/a.parquet", "x/polygons/c.txt"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    source = SOURCES[WEBSITE]
    assert plan.list_input_files(source, "rev1") == ["polygons/a.parquet", "polygons/b.parquet"]
    assert calls == [(source.repo_id, "dataset", "rev1")]


def test_download_passes_repo_path_type_and_revision(monkeypatch):
    calls = []

    def fake(repo_id, path, *, repo_type, revision):
        calls.append((repo_id, path, repo_type, revision))
        return "/cache/file"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    source = SOURCES[WEBSITE]
    assert plan.download(source, "polygons/a.parquet", "rev1") == Path("/cache/file")
    assert calls == [(source.repo_id, "polygons/a.parquet", "dataset", "rev1")]


def test_a_legacy_index_gains_the_replanning_columns(tmp_path):
    path = tmp_path / "index" / f"{WEBSITE}.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE texts (sha TEXT PRIMARY KEY, text TEXT, file_idx INTEGER, chunk_id TEXT)"
        )
    Planner(WorkStore(tmp_path), WEBSITE, "fp")
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(texts)")}
    assert {"done", "cell", "chunk_id"} <= columns


def test_unknown_dataset_is_refused(tmp_path):
    with pytest.raises(KeyError):
        Planner(WorkStore(tmp_path), "nope", "fp")


def test_chunk_files_are_readable_parquet(tmp_path):
    p, _ = make(tmp_path, {"a": [ref("x")]})
    p.scan(Path)
    p.emit(encode, "S: {}", chunk_size=1, final=True)
    (line,) = WorkStore(tmp_path).read_jsonl(p.plan_path)
    assert pq.read_table(tmp_path / "chunks" / f"{line['chunk_id']}.parquet").num_rows == 1
