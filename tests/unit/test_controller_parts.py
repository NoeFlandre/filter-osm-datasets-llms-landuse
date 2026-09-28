import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyarrow as pa

from landuse_filter.adapters.remote import DirRemote
from landuse_filter.adapters.schema import CHUNK
from landuse_filter.adapters.store import WorkStore
from landuse_filter.application.inventory import admission, eligible, load_clusters, profile_for
from landuse_filter.application.memory import ClusterMemory
from landuse_filter.application.staging import Transport
from landuse_filter.application.work_progress import WorkProgress
from landuse_filter.domain.capacity import Cluster
from landuse_filter.domain.gpu import Admission

NOW = datetime(2026, 9, 28, 22, tzinfo=ZoneInfo("Europe/Paris"))
GRES = Cluster("nancy", "gres", "L40S", 46068, (8, 9), 2, 2, ("abaca",), exotic=False)


def test_memory_back_off_expires_and_besteffort_is_remembered(tmp_path):
    m = ClusterMemory(WorkStore(tmp_path))
    assert not m.backed_off(GRES, NOW)
    m.back_off("nancy", "gres", NOW + timedelta(minutes=30))
    assert m.backed_off(GRES, NOW)
    assert not m.backed_off(GRES, NOW + timedelta(minutes=31))
    assert not m.besteffort_only(GRES)
    m.remember_besteffort_only("nancy", "gres")
    assert m.besteffort_only(GRES)


def test_inventory_loading_eligibility_admission_and_profiles(tmp_path):
    store = WorkStore(tmp_path)
    store.write_json(
        "inventory.json",
        [
            {
                "site": "nancy",
                "cluster": "gres",
                "gpu": "L40S",
                "memory_mib": 46068,
                "compute_capability": [8, 9],
                "gpus_per_node": 2,
                "nodes": 2,
                "queues": ["abaca"],
                "exotic": False,
            },
            {
                "site": "lyon",
                "cluster": "hydra",
                "gpu": "GH200",
                "memory_mib": 97871,
                "compute_capability": [9, 0],
                "gpus_per_node": 1,
                "nodes": 4,
                "queues": ["default"],
                "exotic": True,
                "arch": "aarch64",
            },
        ],
    )
    clusters = load_clusters(store)
    assert [eligible(c) for c in clusters] == [True, False]
    assert admission(store, "NVIDIA L40S") is Admission.PENDING
    store.write_json("gates/admission/l40s.json", {"status": "admitted"})
    assert admission(store, "NVIDIA L40S") is Admission.ADMITTED
    assert profile_for(store, "NVIDIA L40S").max_running_requests == 16
    store.write_json("profiles/l40s.json", {"gpu": "l40s", "max_running_requests": 64})
    assert profile_for(store, "NVIDIA L40S").max_running_requests == 64


def test_progress_dedupes_plan_lines_and_marks_complete(tmp_path):
    store = WorkStore(tmp_path)
    store.append_jsonl(
        "plans/a/fp/chunks.jsonl", [{"chunk_id": "c1", "size": 2}, {"chunk_id": "c2", "size": 1}]
    )
    store.append_jsonl("plans/b/fp/chunks.jsonl", [{"chunk_id": "c1", "size": 2}])
    manifest = store.path("parts/fp/c1/p.json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"text_sha256s": ["x", "y"]}))
    progress = WorkProgress(
        store,
        plan_fp="fp",
        work_fp="fp",
        datasets=["a", "b"],
        complete_log="complete.jsonl",
        log=lambda m: None,
    )
    assert [r["chunk_id"] for r in progress.plan_lines()] == ["c1", "c2"]
    assert progress.pending() == [("c2", 1)]
    assert store.read_jsonl("complete.jsonl") == [{"chunk_id": "c1"}]


def test_transport_uploads_chunks_once_and_frees_local_copies(tmp_path):
    store = WorkStore(tmp_path / "w")
    for c in ("c1", "c2"):
        store.write_chunk(
            c, pa.table({"text_sha256": ["s"], "text": ["t"], "input_ids": [[1]]}, schema=CHUNK)
        )
    remote = DirRemote(tmp_path / "bucket")
    remote.put([(store.path("chunks/c1.parquet"), "chunks/c1.parquet")])
    Transport(store, "bucket", log=lambda m: None).upload_chunks(remote, ["c1", "c2"])
    assert remote.ls("chunks/") == ["chunks/c1.parquet", "chunks/c2.parquet"]
    assert not store.exists("chunks/c1.parquet")
    assert not store.exists("chunks/c2.parquet")
