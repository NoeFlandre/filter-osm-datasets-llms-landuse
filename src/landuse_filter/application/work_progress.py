"""Which chunks of a namespace are still pending (plan lines + progress index)."""

from collections.abc import Callable
from pathlib import Path

from landuse_filter.adapters.indexes import ProgressIndex
from landuse_filter.adapters.store import CorruptPartError, WorkStore


class WorkProgress:
    def __init__(
        self,
        store: WorkStore,
        *,
        plan_fp: str,
        work_fp: str,
        datasets: list[str],
        complete_log: str,
        log: Callable[[str], None],
    ) -> None:
        self.store = store
        self.plan_fp = plan_fp
        self.work_fp = work_fp
        self.datasets = datasets
        self.complete_log = complete_log
        self.log = log
        self._indexes: dict[str, ProgressIndex] = {}
        self._compacted = False

    def plan_lines(self) -> list[dict]:
        seen: set[str] = set()
        lines = []
        for dataset in self.datasets:
            for row in self.store.read_jsonl(f"plans/{dataset}/{self.plan_fp}/chunks.jsonl"):
                if row["chunk_id"] not in seen:
                    seen.add(row["chunk_id"])
                    lines.append(row)
        return lines

    def index(self, fp: str) -> ProgressIndex:
        """Per-namespace index of generated hashes, backfilled once from local manifests."""
        if fp not in self._indexes:
            index = ProgressIndex(self.store.path(f"index/progress-{fp}.sqlite"))
            local = sorted(self.store.path(f"parts/{fp}").glob("*/*.json"))
            index.ingest(local)
            for path in local:  # ingested: the manifest is no longer needed on disk
                path.unlink(missing_ok=True)
            self._indexes[fp] = index
        return self._indexes[fp]

    def done_count(self, chunk_id: str, fp: str | None = None) -> int:
        """Distinct texts of a chunk already generated (manifest index + local parts)."""
        fp = fp or self.work_fp
        index = self.index(fp)
        local = list(self.store.part_paths(fp, chunk_id))
        if not local:
            return index.count(chunk_id)
        shas = index.shas(chunk_id)
        for path in local:
            try:
                shas.update(self.store.read_part(path).column("text_sha256").to_pylist())
            except CorruptPartError:
                self.quarantine(path)
        return len(shas)

    def quarantine(self, path: Path) -> None:
        target = self.store.path("quarantine") / path.relative_to(self.store.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        self.log(f"quarantined corrupt part {path.name}")

    def pending(self) -> list[tuple[str, int]]:
        complete = {r["chunk_id"] for r in self.store.read_jsonl(self.complete_log)}
        if not self._compacted:  # once per process: drop what earlier runs kept for finished chunks
            self._compacted = True
            if self.index(self.work_fp).forget(complete):
                self.index(self.work_fp).compact()
        pending, newly = [], []
        for row in self.plan_lines():
            cid = row["chunk_id"]
            if cid in complete:
                continue
            if self.done_count(cid) >= row["size"]:
                newly.append({"chunk_id": cid})
            else:
                pending.append((cid, row["size"]))
        if newly:
            self.store.append_jsonl(self.complete_log, newly)
            self.index(self.work_fp).forget(c["chunk_id"] for c in newly)
        return pending
