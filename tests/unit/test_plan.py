from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter.adapters.readers import WEBSITE
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.plan import Planner

INPUTS = Path(__file__).parents[1] / "fixtures" / "inputs"


def encode(prompt):
    return [len(w) for w in prompt.split()]


def planner(tmp_path):
    p = Planner(WorkStore(tmp_path), WEBSITE, "fp")
    p.register(["polygons/a.parquet", "polygons/b.parquet"])
    return p


def test_scan_is_resumable_and_dedupes_across_files(tmp_path):
    p = planner(tmp_path)
    assert p.scan(lambda _: INPUTS / "website.parquet", limit=1) == 1
    first = p.report()
    assert (first.files_done, first.files_total) == (1, 2)
    again = Planner(WorkStore(tmp_path), WEBSITE, "fp")  # a restarted process
    assert again.scan(lambda _: INPUTS / "website.parquet") == 1
    final = again.report()
    assert final.files_done == 2
    assert final.unique_texts == first.unique_texts  # the same file twice adds nothing
    assert final.sentences == 2 * first.sentences


def test_emit_holds_back_partial_chunks_until_final(tmp_path):
    p = planner(tmp_path)
    p.scan(lambda _: INPUTS / "website.parquet")
    unique = p.report().unique_texts
    full = p.emit(encode, "S: {}", chunk_size=50, final=False)
    assert full == unique // 50
    rest = p.emit(encode, "S: {}", chunk_size=50, final=True)
    assert full + rest == -(-unique // 50)
    assert p.emit(encode, "S: {}", chunk_size=50, final=True) == 0  # idempotent
    store = WorkStore(tmp_path)
    lines = store.read_jsonl(f"plans/{WEBSITE}/fp/chunks.jsonl")
    total = sum(pq.read_table(store.path(f"chunks/{row['chunk_id']}.parquet")).num_rows for row in lines)
    assert total == unique


def test_emit_waits_for_unscanned_earlier_files(tmp_path):
    p = planner(tmp_path)
    assert p.emit(encode, "S: {}", chunk_size=1, final=True) == 0
