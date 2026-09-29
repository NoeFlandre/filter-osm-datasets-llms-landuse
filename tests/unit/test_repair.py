import pyarrow as pa
import pyarrow.parquet as pq

from landuse_filter.application.repair import dedupe_generations


def write(tmp_path, name, ids):
    path = tmp_path / name
    pq.write_table(pa.table({"generation_id": ids, "raw_output": [f"o-{i}" for i in ids]}), path)
    return path


def test_first_occurrence_is_kept_and_empty_files_are_dropped(tmp_path):
    a = write(tmp_path, "batch-00001-part-00000.parquet", ["x", "y"])
    b = write(tmp_path, "batch-00002-part-00000.parquet", ["y", "z"])
    c = write(tmp_path, "batch-00003-part-00000.parquet", ["x", "z"])
    changed = dedupe_generations([c, b, a])  # order of arguments does not matter
    assert a not in changed  # already clean
    assert changed[b].column("generation_id").to_pylist() == ["z"]
    assert changed[c] is None


def test_clean_input_changes_nothing(tmp_path):
    files = [write(tmp_path, "a.parquet", ["x"]), write(tmp_path, "b.parquet", ["y"])]
    assert dedupe_generations(files) == {}
