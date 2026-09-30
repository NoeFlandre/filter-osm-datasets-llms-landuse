from landuse_filter.adapters.store import WorkStore
from landuse_filter.cli import g5k as cli


def test_only_parts_whose_result_is_already_here_count_as_synced(tmp_path):
    store = WorkStore(tmp_path)
    store.path("parts/fp/c1").mkdir(parents=True)
    store.path("parts/fp/c1/p1.parquet").write_bytes(b"x")  # the parquet is here
    store.path("parts/fp/c2").mkdir(parents=True)
    store.path("parts/fp/c2/p2.json").write_text("{}")  # only its manifest is here
    listing = [
        ("luf/work/parts/fp/c1/p1.parquet", 1.0),
        ("luf/work/parts/fp/c2/p2.parquet", 1.0),
        ("luf/work/parts/fp/c3/p3.parquet", 1.0),  # never seen here
        ("luf/code/abc", 1.0),  # not a part
    ]
    assert cli._synced_parts(store, listing) == {"fp/c1/p1.parquet", "fp/c2/p2.parquet"}


def test_live_commits_are_those_of_live_jobs_and_the_current_main(tmp_path, monkeypatch):
    from landuse_filter.application import controller

    store = WorkStore(tmp_path)
    store.write_json(
        "assignments/a.json", {"state": "submitted", "provenance": {"code_commit": "c1"}}
    )
    store.write_json("assignments/b.json", {"state": "ended", "provenance": {"code_commit": "c2"}})
    monkeypatch.setattr(controller, "commit", lambda: "main-head")
    assert cli._live_commits(store) == {"c1", "main-head"}
