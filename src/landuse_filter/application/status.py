"""Progress, throughput, ETA and drift alerts from the controller's small local tree."""

from collections import Counter

from landuse_filter.adapters.store import WorkStore
from landuse_filter.config import GENERATION_FP
from landuse_filter.domain.gpu import Profile, gpu_key
from landuse_filter.domain.progress import alerts, eta


def _json_files(store: WorkStore, pattern: str) -> list[dict]:
    return [
        store.read_json(str(p.relative_to(store.root))) for p in sorted(store.root.glob(pattern))
    ]


def _rate(store: WorkStore, gpu: str) -> float:
    path = f"profiles/{gpu_key(gpu)}.json"
    return (
        Profile(**store.read_json(path)).sentences_per_second
        if store.exists(path)
        else Profile("x").sentences_per_second
    )


def _dataset_status(store: WorkStore, dataset: str, complete: set[str], live: list[dict]) -> dict:
    rows = {
        r["chunk_id"]: r for r in store.read_jsonl(f"plans/{dataset}/{GENERATION_FP}/chunks.jsonl")
    }
    done = sum(r["size"] for cid, r in rows.items() if cid in complete)
    texts = sum(r["size"] for r in rows.values())
    mine = [a for a in live if set(a["chunks"]) & rows.keys()]
    estimate = eta(texts - done, [_rate(store, a["gpu"]) for a in mine])
    hours = estimate.hours
    return {
        "chunks": len(rows),
        "chunks_complete": sum(c in complete for c in rows),
        "texts": texts,
        "texts_complete": done,
        "live_jobs": len(mine),
        "sentences_per_second": round(estimate.sentences_per_second, 2),
        "eta_hours": None if hours is None else round(hours, 1),
    }


def _throughput_by_gpu(jobs: list[dict]) -> dict[str, float]:
    by_gpu: dict[str, list[float]] = {}
    for j in jobs:
        by_gpu.setdefault(j["gpu"], []).append(j["sentences_per_second"])
    return {g: round(sum(v) / len(v), 3) for g, v in by_gpu.items()}


def summarize(store: WorkStore, datasets: list[str]) -> dict:
    complete = {r["chunk_id"] for r in store.read_jsonl("complete.jsonl")}
    assignments = _json_files(store, "assignments/*.json")
    live = [
        a
        for a in assignments
        if a.get("state") in ("submitting", "submitted") and a.get("fp") == GENERATION_FP
    ]
    jobs = _json_files(store, "jobs/*/*.json")
    return {
        "config_fingerprint": GENERATION_FP,
        **{d: _dataset_status(store, d, complete, live) for d in datasets},
        "assignments": dict(Counter(a.get("state") for a in assignments)),
        "throughput_by_gpu": _throughput_by_gpu(jobs),
        "alerts": alerts(
            sum(j.get("completed", 0) for j in jobs), sum(j.get("failed", 0) for j in jobs)
        ),
    }
