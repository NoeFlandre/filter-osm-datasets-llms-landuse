"""Moving work between the local tree and the remote store.

A part ``parts/<fp>/<chunk>/<part>.parquet`` always travels with a manifest
``parts/<fp>/<chunk>/<part>.json`` (its text hashes): the controller tracks progress
from manifests alone and never downloads raw outputs.
"""

import json
from pathlib import Path

from landuse_filter.adapters.remote import Remote
from landuse_filter.adapters.store import WorkStore


def manifest_path(fp: str, chunk_id: str, part_id: str) -> str:
    return f"parts/{fp}/{chunk_id}/{part_id}.json"


def upload_part(
    remote: Remote, store: WorkStore, fp: str, chunk_id: str, *, part_id: str, shas: list[str]
) -> None:
    """Upload a part, then its manifest (manifest last: its presence implies the part)."""
    parquet = f"parts/{fp}/{chunk_id}/{part_id}.parquet"
    manifest = manifest_path(fp, chunk_id, part_id)
    store.write_json(manifest, {"part_id": part_id, "text_sha256s": sorted(shas)})
    remote.put([(store.path(parquet), parquet)])
    remote.put([(store.path(manifest), manifest)])


def fetch(remote: Remote, store: WorkStore, paths: list[str]) -> None:
    missing = [p for p in paths if not store.exists(p)]
    remote.get([(p, store.path(p)) for p in missing])


def fetch_manifests(remote: Remote, store: WorkStore, prefix: str) -> list[str]:
    """Download manifests under ``prefix`` that are not local yet; returns their paths."""
    wanted = [p for p in remote.ls(prefix) if p.endswith(".json") and not store.exists(p)]
    fetch(remote, store, wanted)
    return wanted


def manifest_shas(store: WorkStore, fp: str, chunk_id: str) -> set[str]:
    shas: set[str] = set()
    for path in sorted(store.path(f"parts/{fp}/{chunk_id}").glob("*.json")):
        shas.update(json.loads(path.read_text(encoding="utf-8"))["text_sha256s"])
    return shas


def prune_local_part(store: WorkStore, fp: str, chunk_id: str, part_id: str) -> None:
    """Drop a part's bulk bytes once uploaded; its manifest stays for progress."""
    Path(store.path(f"parts/{fp}/{chunk_id}/{part_id}.parquet")).unlink(missing_ok=True)
