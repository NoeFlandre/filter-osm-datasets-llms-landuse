"""Progress summary from the local work tree."""

from collections import Counter

from landuse_filter.adapters.store import WorkStore
from landuse_filter.config import GENERATION_FP


def summarize(store: WorkStore, datasets: list[str]) -> dict:
    complete = {r["chunk_id"] for r in store.read_jsonl("complete.jsonl")}
    out: dict = {"config_fingerprint": GENERATION_FP}
    for dataset in datasets:
        rows = {r["chunk_id"]: r for r in store.read_jsonl(f"plans/{dataset}/{GENERATION_FP}/chunks.jsonl")}
        done = [r for cid, r in rows.items() if cid in complete]
        out[dataset] = {
            "chunks": len(rows), "chunks_complete": len(done),
            "texts": sum(r["size"] for r in rows.values()), "texts_complete": sum(r["size"] for r in done),
        }
    assignments = [store.read_json(f"assignments/{p.name}") for p in sorted(store.path("assignments").glob("*.json"))]
    out["assignments"] = dict(Counter(a.get("state") for a in assignments))
    jobs = [store.read_json(str(p.relative_to(store.root))) for p in sorted(store.path("jobs").glob("*/*.json"))]
    by_gpu: dict[str, list[float]] = {}
    for j in jobs:
        by_gpu.setdefault(j["gpu"], []).append(j["sentences_per_second"])
    out["throughput_by_gpu"] = {g: round(sum(v) / len(v), 3) for g, v in by_gpu.items()}
    return out
