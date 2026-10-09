import hashlib

import pyarrow as pa
import pytest

from landuse_filter import config
from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import PROVENANCE, generation_table
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import resolution_sync as rs
from landuse_filter.application.results import canonical_generations
from landuse_filter.application.sync import upload_part
from landuse_filter.domain.parsing import PARSER_VERSION, THINK_CLOSE
from landuse_filter.domain.records import Generation

FP = config.GENERATION_FP


class CountingRemote(DirRemote):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.fetched: list[str] = []

    def get(self, files) -> None:
        self.fetched += [src for src, _ in files]
        super().get(files)


def _table(rows: list[tuple[str, str]]) -> pa.Table:
    prov = {name: "p" for name, _ in PROVENANCE}
    gens = [Generation(s, o, 5, 3, "stop", False, None, None, None, None) for s, o in rows]
    return generation_table(gens, prov)


def _part(store: WorkStore, remote: DirRemote, chunk: str, rows: list[tuple[str, str]]) -> str:
    """Write a result part (sha, raw_output) to the store and the bucket; returns its key."""
    part = store.write_part(FP, chunk, _table(rows))
    upload_part(remote, store, FP, chunk, part_id=part, shas=[s for s, _ in rows])
    return f"{chunk}/{part}"


def _parts_fetched(remote: CountingRemote) -> list[str]:
    return [p for p in remote.fetched if p.startswith("parts/")]


def test_second_restore_reads_no_part_and_a_new_part_only(tmp_path):
    remote = CountingRemote(tmp_path / "bucket")
    src = WorkStore(tmp_path / "src")
    first = _part(src, remote, "c1", [("a", "x</think>yes"), ("b", "x</think>no")])
    index = rs.restore_resolution(remote, WorkStore(tmp_path / "n1"), FP, workers=1)
    assert index.get("a")[0] == "yes"
    assert index.get("b")[0] == "no"
    assert index.parts() == {first}
    remote.fetched.clear()

    again = rs.restore_resolution(remote, WorkStore(tmp_path / "n2"), FP, workers=1)
    assert again.get("a")[0] == "yes"
    assert _parts_fetched(remote) == []  # the snapshot alone

    second = _part(src, remote, "c2", [("c", "x</think>yes")])
    remote.fetched.clear()
    third = rs.restore_resolution(remote, WorkStore(tmp_path / "n3"), FP, workers=1)
    assert third.parts() == {first, second}
    assert _parts_fetched(remote) == [f"parts/{FP}/{second}.parquet"]
    assert third.get("c")[0] == "yes"


def test_parts_are_deleted_from_the_scratch_once_indexed(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    _part(WorkStore(tmp_path / "src"), remote, "c1", [("a", "x</think>yes")])
    scratch = WorkStore(tmp_path / "n")
    rs.restore_resolution(remote, scratch, FP, workers=1)
    assert list(scratch.part_paths(FP)) == []


def test_smallest_part_wins_whatever_the_order(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    src = WorkStore(tmp_path / "src")
    _part(src, remote, "c2", [("a", "x</think>no")])
    rs.restore_resolution(remote, WorkStore(tmp_path / "n1"), FP, workers=1)
    _part(src, remote, "c1", [("a", "x</think>yes")])  # a smaller part id arrives later
    index = rs.restore_resolution(remote, WorkStore(tmp_path / "n2"), FP, workers=1)
    assert index.get("a")[0] == "yes"
    assert index.parts_of(["a", "zz"]).keys() == {"a"}
    assert index.parts_of(["a"])["a"].startswith("c1/")


def test_a_snapshot_from_another_parser_is_rebuilt(tmp_path):
    remote = CountingRemote(tmp_path / "bucket")
    _part(WorkStore(tmp_path / "src"), remote, "c1", [("a", "x</think>yes")])
    rs.restore_resolution(remote, WorkStore(tmp_path / "n1"), FP, workers=1)
    stale = ResolutionIndex(remote.root / rs.snapshot_path(FP))
    stale.set_meta("parser", PARSER_VERSION + "-old")
    stale.db.close()
    remote.fetched.clear()
    index = rs.restore_resolution(remote, WorkStore(tmp_path / "n2"), FP, workers=1)
    assert index.meta("parser") == PARSER_VERSION
    assert len(_parts_fetched(remote)) == 1  # read again


def test_a_corrupt_part_is_skipped_and_not_recorded(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    bad = remote.root / f"parts/{FP}/c1/{'0' * 64}.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a part")
    index = rs.restore_resolution(remote, WorkStore(tmp_path / "n"), FP, workers=1)
    assert index.parts() == set()


def test_a_stop_request_ends_the_catch_up_but_keeps_the_snapshot(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    _part(WorkStore(tmp_path / "src"), remote, "c1", [("a", "x</think>yes")])
    index = rs.restore_resolution(
        remote, WorkStore(tmp_path / "n"), FP, should_stop=lambda: True, workers=1
    )
    assert index.get("a") is None
    assert rs.snapshot_path(FP) in remote.ls("index/")


def test_fetch_parts_downloads_only_what_is_missing(tmp_path):
    remote = CountingRemote(tmp_path / "bucket")
    key = _part(WorkStore(tmp_path / "src"), remote, "c1", [("a", "x</think>yes")])
    remote.fetched.clear()
    scratch = WorkStore(tmp_path / "n")
    rs.fetch_parts(remote, scratch, FP, [key, key])
    rs.fetch_parts(remote, scratch, FP, [key])
    assert remote.fetched == [f"parts/{FP}/{key}.parquet"]
    assert [r["text_sha256"] for r in canonical_generations(scratch, FP)] == ["a"]


def test_part_key_round_trips():
    path = f"parts/{FP}/c1/abc.parquet"
    assert rs.part_key(path) == "c1/abc"
    assert rs.part_path(FP, "c1/abc") == path


class PutLog(CountingRemote):
    def __init__(self, root) -> None:
        super().__init__(root)
        self.puts: list[str] = []

    def put(self, files) -> None:
        self.puts += [dst for _, dst in files]
        super().put(files)


def _five_parts(tmp_path, remote):
    src = WorkStore(tmp_path / "src")
    return [_part(src, remote, f"c{i}", [(f"s{i}", "x</think>yes")]) for i in range(5)]


def test_catch_up_reads_every_part_across_several_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "BATCH", 2)
    remote = CountingRemote(tmp_path / "bucket")
    keys = _five_parts(tmp_path, remote)
    remote.fetched.clear()
    index = rs.restore_resolution(remote, WorkStore(tmp_path / "n"), FP, workers=1)
    assert index.parts() == set(keys)
    assert sorted(_parts_fetched(remote)) == sorted(f"parts/{FP}/{k}.parquet" for k in keys)


def test_a_stop_after_the_first_batch_keeps_that_batch_and_uploads_the_snapshot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rs, "BATCH", 2)
    remote = PutLog(tmp_path / "bucket")
    keys = _five_parts(tmp_path, remote)
    remote.puts.clear()
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 1  # let exactly one batch through

    index = rs.restore_resolution(
        remote, WorkStore(tmp_path / "n"), FP, should_stop=stop, workers=1
    )
    assert len(index.parts()) == 2
    assert index.parts() < set(keys)
    assert remote.puts == [rs.snapshot_path(FP)]


def test_a_zero_checkpoint_interval_uploads_after_every_batch_and_not_again_at_the_end(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rs, "BATCH", 2)
    remote = PutLog(tmp_path / "bucket")
    _five_parts(tmp_path, remote)
    remote.puts.clear()
    rs.restore_resolution(remote, WorkStore(tmp_path / "n"), FP, workers=1, checkpoint_seconds=0.0)
    assert remote.puts == [rs.snapshot_path(FP)] * 3  # one per batch (2 + 2 + 1 parts)


def test_a_long_checkpoint_interval_uploads_once_at_the_end(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "BATCH", 2)
    remote = PutLog(tmp_path / "bucket")
    _five_parts(tmp_path, remote)
    remote.puts.clear()
    rs.restore_resolution(
        remote, WorkStore(tmp_path / "n"), FP, workers=1, checkpoint_seconds=10**9
    )
    assert remote.puts == [rs.snapshot_path(FP)]


def test_an_up_to_date_snapshot_is_not_uploaded_again(tmp_path):
    remote = PutLog(tmp_path / "bucket")
    _part(WorkStore(tmp_path / "src"), remote, "c1", [("a", "x</think>yes")])
    rs.restore_resolution(remote, WorkStore(tmp_path / "n1"), FP, workers=1)
    remote.puts.clear()
    rs.restore_resolution(remote, WorkStore(tmp_path / "n2"), FP, workers=1)
    assert remote.puts == []


def test_a_failing_download_still_uploads_the_snapshot_and_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "BATCH", 1)
    remote = PutLog(tmp_path / "bucket")
    _five_parts(tmp_path, remote)
    remote.puts.clear()
    real = rs._download
    n = []

    def flaky(*a):
        n.append(1)
        if len(n) == 2:
            raise OSError("boom")
        return real(*a)

    monkeypatch.setattr(rs, "_download", flaky)
    with pytest.raises(OSError, match="boom"):
        rs.restore_resolution(remote, WorkStore(tmp_path / "n"), FP, workers=1)
    assert remote.puts == [rs.snapshot_path(FP)]


def test_read_verdicts_keeps_the_first_row_per_hash_and_parses_it(tmp_path):
    store = WorkStore(tmp_path / "s")
    rows = [
        ("h1", f"{THINK_CLOSE}\nyes"),
        ("h1", f"{THINK_CLOSE}\nno"),
        ("h2", f"{THINK_CLOSE}\nno"),
    ]
    part = store.write_part(FP, "c", _table(rows))
    verdicts = rs.read_verdicts(str(store.path(rs.part_path(FP, f"c/{part}"))))
    assert [(sha, decision) for sha, decision, *_ in verdicts] == [("h1", "yes"), ("h2", "no")]


def test_read_verdicts_rejects_a_part_whose_bytes_do_not_match_its_name(tmp_path):
    part = tmp_path / ("0" * 64 + ".parquet")
    part.write_bytes(b"not parquet")
    assert rs.read_verdicts(str(part)) is None


def test_read_verdicts_rejects_an_unreadable_part_with_a_matching_name(tmp_path):
    data = b"not parquet"
    part = tmp_path / (hashlib.sha256(data).hexdigest() + ".parquet")
    part.write_bytes(data)
    assert rs.read_verdicts(str(part)) is None
