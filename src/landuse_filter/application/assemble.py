"""Build the ``labels/`` and ``generations/`` tables of a ``-landuse`` output dataset.

``labels/<input path>`` holds one row per in-scope sentence position of that input
file, keyed by ``label_id`` and the dataset's natural join keys; ``generations/``
holds one row per unique text (raw output, token counts, provenance) keyed by
``generation_id``, referenced from labels. Nothing is dropped: an unsplit text is
``skipped_unsplit`` with no generation; a failed parse is ``failed`` with its reason.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from landuse_filter.adapters.readers import SOURCES
from landuse_filter.adapters.store import WorkStore, write_atomic, table_bytes
from landuse_filter.application.results import canonical_generations, verdict_of
from landuse_filter.domain.hashing import sha256_parts
from landuse_filter.domain.sentences import Decision, SentenceRef

LABEL_FIELDS = [
    ("label_id", pa.string()),
    ("text_sha256", pa.string()),
    ("decision", pa.string()),
    ("parse_mode", pa.string()),
    ("failure_reason", pa.string()),
    ("generation_id", pa.string()),
    ("language", pa.string()),
    ("config_fingerprint", pa.string()),
    ("input_revision", pa.string()),
]

JOIN_KEYS = {
    "osm-polygon-description-tag": [("description_identity", pa.string()), ("tag_key", pa.string()), ("sentence_index", pa.int32())],
    "osm-polygon-wikidata-and-wikipedia": [("sentence_id", pa.string())],
    "osm-polygon-website-tag": [("polygon_id", pa.string()), ("field", pa.string()), ("sentence_index", pa.int32())],
}


def generation_id(text_sha256: str, fp: str) -> str:
    return sha256_parts(text_sha256, fp)[:32]


@dataclass(frozen=True, slots=True)
class Resolved:
    decision: str
    mode: str | None
    failure: str | None


class Missing(LookupError):
    """A non-skipped sentence has no generation yet: the file is not publishable."""


def resolve_all(store: WorkStore, fp: str) -> dict[str, Resolved]:
    out = {}
    for row in canonical_generations(store, fp):
        v = verdict_of(row)
        out[row["text_sha256"]] = Resolved(v.decision.value, v.mode and v.mode.value, v.failure and v.failure.value)
    return out


def label_rows(refs: Iterable[SentenceRef], resolved: dict[str, Resolved], fp: str, revision: str) -> list[dict]:
    rows = []
    for ref in refs:
        keys = dict(ref.locator)
        base = {"label_id": ref.label_id, **keys, "text_sha256": ref.text_sha256, "language": ref.language,
                "config_fingerprint": fp, "input_revision": revision}
        if ref.unsplit:
            rows.append({**base, "decision": Decision.SKIPPED_UNSPLIT.value, "parse_mode": None,
                         "failure_reason": "unsplit_upstream", "generation_id": None})
            continue
        r = resolved.get(ref.text_sha256)
        if r is None:
            raise Missing(ref.text_sha256)
        rows.append({**base, "decision": r.decision, "parse_mode": r.mode, "failure_reason": r.failure,
                     "generation_id": generation_id(ref.text_sha256, fp)})
    return rows


def labels_table(dataset: str, rows: list[dict]) -> pa.Table:
    schema = pa.schema([LABEL_FIELDS[0], *JOIN_KEYS[dataset], *LABEL_FIELDS[1:]])
    return pa.Table.from_pylist(rows, schema=schema)


def build_labels(dataset: str, input_path: str, local: Path, resolved: dict[str, Resolved],
                 fp: str, revision: str, out: Path) -> int:
    """Write ``out/labels/<input_path>``; raises ``Missing`` if any text is unresolved."""
    refs = SOURCES[dataset].read(local, input_path)
    table = labels_table(dataset, label_rows(refs, resolved, fp, revision))
    write_atomic(out / "labels" / input_path, table_bytes(table))
    return table.num_rows


def build_generations(store: WorkStore, fp: str, out: Path, shas: set[str], rows_per_file: int = 50_000) -> int:
    """Write the canonical generations of ``shas`` as ``generations/<fp>/part-NNNNN.parquet``."""
    batch, n, written = [], 0, 0
    for row in canonical_generations(store, fp):
        if row["text_sha256"] not in shas:
            continue
        batch.append({"generation_id": generation_id(row["text_sha256"], fp), **row})
        if len(batch) >= rows_per_file:
            _write_generations(out, fp, n, batch)
            n, written, batch = n + 1, written + len(batch), []
    if batch:
        _write_generations(out, fp, n, batch)
        written += len(batch)
    return written


def _write_generations(out: Path, fp: str, n: int, rows: list[dict]) -> None:
    write_atomic(out / "generations" / fp / f"part-{n:05d}.parquet", table_bytes(pa.Table.from_pylist(rows)))
