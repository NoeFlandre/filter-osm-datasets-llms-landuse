from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

from landuse_filter.adapters import hub
from landuse_filter.adapters.remote import BucketRemote, DirRemote


class FakeApi:
    commits: list = []
    batches: list = []

    def create_repo(self, repo_id, **kw):
        self.commits.append(("create", repo_id, kw))

    def list_repo_files(self, repo_id, **kw):
        return ["b.txt", "a.txt"]

    def create_commit(self, repo_id, ops, **kw):
        self.commits.append(("commit", repo_id, len(ops)))

    def create_bucket(self, bucket_id, **kw):
        self.batches.append(("create", kw))

    def batch_bucket_files(self, bucket_id, **kw):
        self.batches.append(kw)

    def list_bucket_tree(self, bucket_id, prefix=None, recursive=None):
        return [
            SimpleNamespace(type="file", path=f"{prefix}x.json"),
            SimpleNamespace(type="directory", path=prefix),
        ]

    def download_bucket_files(self, bucket_id, files, **kw):
        for _, dst in files:
            Path(dst).write_text("{}")


def install(monkeypatch):
    FakeApi.commits, FakeApi.batches = [], []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "BATCH", 2)


def test_hub_upload_commits_in_batches_and_lists(monkeypatch, tmp_path):
    install(monkeypatch)
    hub.ensure_dataset("u/d")
    files = [(tmp_path / f"{i}", f"p{i}") for i in range(5)]
    for f, _ in files:
        f.write_text("x")
    hub.upload("u/d", files, "msg")
    assert [c[2] for c in FakeApi.commits if c[0] == "commit"] == [2, 2, 1]
    assert FakeApi.commits[0][2]["private"] is False
    assert hub.list_files("u/d", "rev") == ["a.txt", "b.txt"]
    assert hub.remote_files("u/d") == {"a.txt", "b.txt"}


def test_bucket_remote_put_get_ls_delete(monkeypatch, tmp_path):
    install(monkeypatch)
    b = BucketRemote("u/work")
    b.BATCH = 1
    b.ensure()
    src = tmp_path / "f"
    src.write_text("x")
    b.put([(src, "a"), (src, "b")])
    assert len([x for x in FakeApi.batches if isinstance(x, dict) and "add" in x]) == 2
    assert b.ls("parts/") == ["parts/x.json"]  # directories are skipped
    b.get([("parts/x.json", tmp_path / "out.json")])
    assert (tmp_path / "out.json").read_text() == "{}"
    b.get([])
    b.delete(["a", "b"])
    assert FakeApi.batches[-1] == {"delete": ["b"]}


def test_dir_remote_round_trip(tmp_path):
    r = DirRemote(tmp_path / "bucket")
    src = tmp_path / "f"
    src.write_text("hello")
    r.put([(src, "x/y.txt")])
    assert r.ls("x/") == ["x/y.txt"]
    assert r.ls("x/y.txt") == ["x/y.txt"]
    assert r.ls("missing/") == []
    r.get([("x/y.txt", tmp_path / "copy.txt")])
    assert (tmp_path / "copy.txt").read_text() == "hello"
    r.delete(["x/y.txt"])
    assert r.ls("x/") == []


def test_upload_gate_shares_the_admission_file_through_the_bucket(tmp_path):
    from landuse_filter.adapters.remote import DirRemote
    from landuse_filter.adapters.store import WorkStore
    from landuse_filter.application.sync import upload_gate

    store = WorkStore(tmp_path / "w")
    store.write_json("gates/admission/l40s.json", {"status": "admitted", "gate": {}})
    remote = DirRemote(tmp_path / "bucket")
    upload_gate(remote, store, "l40s")
    assert remote.ls("gates/admission/") == ["gates/admission/l40s.json"]
