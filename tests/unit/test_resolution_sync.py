import pyarrow as pa

from landuse_filter import config
from landuse_filter.adapters.indexes import ResolutionIndex
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import PROVENANCE, generation_table
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application import resolution_sync as rs
from landuse_filter.application.results import canonical_generations
from landuse_filter.application.sync import upload_part
from landuse_filter.domain.parsing import PARSER_VERSION
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
