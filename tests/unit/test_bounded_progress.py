import json
from pathlib import Path

from landuse_filter.adapters.indexes import ProgressIndex
from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.staging import Transport
from landuse_filter.application.sync import fetch_manifests
from landuse_filter.application.work_progress import WorkProgress


def manifest(root: Path, chunk: str, part: str, shas: list[str]) -> Path:
    path = root / "parts" / "fp" / chunk / f"{part}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text_sha256s": shas}))
    return path


def progress_for(store: WorkStore) -> WorkProgress:
    return WorkProgress(
        store,
        plan_fp="fp",
        work_fp="fp",
        datasets=["a"],
        complete_log="complete.jsonl",
        log=lambda m: None,
    )


def test_seen_manifests_are_keyed_by_their_location_not_the_local_root(tmp_path):
    index = ProgressIndex(tmp_path / "idx.sqlite")
    assert index.ingest([manifest(tmp_path / "one", "c1", "p", ["x"])]) == 1
    assert index.ingest([manifest(tmp_path / "two", "c1", "p", ["x"])]) == 0  # same place
    assert index.has_seen("parts/fp/c1/p.json")
    assert not index.has_seen("parts/fp/c1/other.json")


def test_forget_drops_a_chunks_hashes_and_keeps_the_seen_manifests(tmp_path):
    index = ProgressIndex(tmp_path / "idx.sqlite")
    index.ingest([manifest(tmp_path, "c1", "p", ["x", "y"]), manifest(tmp_path, "c2", "q", ["z"])])
    assert index.forget(["c1"]) == 2
    assert (index.count("c1"), index.count("c2")) == (0, 1)
    assert index.has_seen("parts/fp/c1/p.json")  # never fetched again
    index.compact()  # does not raise, does not lose c2
    assert index.count("c2") == 1


def test_fetch_manifests_skips_the_ones_the_index_has_seen(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    src = WorkStore(tmp_path / "src")
    for part in ("p1", "p2"):
        path = manifest(src.root, "c1", part, ["x"])
        remote.put([(path, f"parts/fp/c1/{part}.json")])
    store = WorkStore(tmp_path / "ctl")
    new = fetch_manifests(remote, store, "parts/fp/", seen=lambda p: p.endswith("/p1.json"))
    assert new == ["parts/fp/c1/p2.json"]


def test_pull_ingests_then_deletes_the_local_manifest_and_never_refetches_it(tmp_path):
    remote = DirRemote(tmp_path / "bucket")
    src = WorkStore(tmp_path / "src")
    path = manifest(src.root, "c1", "p1", ["x", "y"])
    remote.put([(path, "parts/fp/c1/p1.json")])
    store = WorkStore(tmp_path / "ctl")
    progress = progress_for(store)
    transport = Transport(store, remote, "bucket", log=lambda m: None)
    transport.pull(progress)
    assert progress.index("fp").count("c1") == 2
    assert not store.exists("parts/fp/c1/p1.json")  # the SSD keeps no manifest
    transport.pull(progress)  # would re-download it if "seen" were the local file
    assert not store.exists("parts/fp/c1/p1.json")


def test_completed_chunks_are_forgotten_so_the_index_stays_bounded(tmp_path):
    store = WorkStore(tmp_path)
    store.append_jsonl(
        "plans/a/fp/chunks.jsonl", [{"chunk_id": "c1", "size": 2}, {"chunk_id": "c2", "size": 2}]
    )
    manifest(tmp_path, "c1", "p", ["x", "y"])  # c1 complete
    manifest(tmp_path, "c2", "p", ["z"])  # c2 half done
    progress = progress_for(store)
    assert progress.pending() == [("c2", 2)]
    index = progress.index("fp")
    assert index.count("c1") == 0  # forgotten: only in-flight chunks stay in the index
    assert index.count("c2") == 1
    assert store.read_jsonl("complete.jsonl") == [{"chunk_id": "c1"}]


def test_a_corrupt_manifest_is_dropped_and_never_marked_seen(tmp_path):
    """Regression: an empty manifest left by an interrupted download made every controller cycle
    raise JSONDecodeError at ingest, so nothing was counted or submitted for an hour."""
    index = ProgressIndex(tmp_path / "idx.sqlite")
    good = tmp_path / "fp" / "c1" / "good.json"
    good.parent.mkdir(parents=True)
    good.write_text('{"text_sha256s": ["a", "b"]}')
    empty = tmp_path / "fp" / "c1" / "empty.json"
    empty.write_text("")
    wrong = tmp_path / "fp" / "c1" / "wrong.json"
    wrong.write_text('{"other": 1}')
    assert index.ingest([empty, good, wrong]) == 1  # only the good one counts as new
    assert index.count("c1") == 2
    assert not empty.exists()
    assert not wrong.exists()  # removed, so the next pull downloads them again
    assert not index.has_seen(str(empty))
    assert good.exists()
