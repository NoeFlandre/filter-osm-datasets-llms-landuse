"""Build the ``labels/`` and ``generations/`` tables of a ``-landuse`` output dataset.

``labels/<input path>`` holds one row per in-scope sentence position of that input
file, keyed by ``label_id`` and the dataset's natural join keys; ``generations/``
holds one row per unique text (raw output, token counts, provenance) keyed by
``generation_id``, referenced from labels. Nothing is dropped: an unsplit text is
``skipped_unsplit`` with no generation; a failed parse is ``failed`` with its reason.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Protocol

import pyarrow as pa

from landuse_filter.adapters.readers import SOURCES
from landuse_filter.adapters.store import WorkStore, table_bytes, write_atomic
from landuse_filter.application.results import canonical_generations
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

VIEWER_COLUMNS = ["sentence", "label", "language", "region"]  # what a dataset-viewer reader needs
PENDING = "pending"  # a sentence of a partly labelled file whose text has no generation yet

JOIN_KEYS = {
    "osm-polygon-description-tag": [
        ("description_identity", pa.string()),
        ("tag_key", pa.string()),
        ("sentence_index", pa.int32()),
    ],
    "osm-polygon-wikidata-and-wikipedia": [("sentence_id", pa.string())],
    "osm-polygon-website-tag": [
        ("polygon_id", pa.string()),
        ("field", pa.string()),
        ("sentence_index", pa.int32()),
    ],
}


def generation_id(text_sha256: str, fp: str) -> str:
    return sha256_parts(text_sha256, fp)[:32]


class Resolved(NamedTuple):
    decision: str
    mode: str | None
    failure: str | None


class Lookup(Protocol):
    """Text hash -> (decision, mode, failure); a dict or an on-disk ResolutionIndex."""

    def get(self, sha: str, /) -> tuple[str, str | None, str | None] | None: ...


class MissingGenerationError(LookupError):
    """A non-skipped sentence has no generation yet: the file is not publishable."""


def label_rows(
    refs: Iterable[SentenceRef],
    resolved: Lookup,
    fp: str,
    revision: str,
    *,
    allow_pending: bool = False,
) -> list[dict]:
    """One row per sentence; with ``allow_pending`` an unresolved text becomes ``pending``."""
    rows = []
    for ref in refs:
        keys = dict(ref.locator)
        base = {
            "label_id": ref.label_id,
            **keys,
            "text_sha256": ref.text_sha256,
            "language": ref.language,
            "config_fingerprint": fp,
            "input_revision": revision,
        }
        if ref.unsplit:
            rows.append(
                {
                    **base,
                    "decision": Decision.SKIPPED_UNSPLIT.value,
                    "parse_mode": None,
                    "failure_reason": "unsplit_upstream",
                    "generation_id": None,
                }
            )
            continue
        r = resolved.get(ref.text_sha256)
        if r is None and allow_pending:
            rows.append(
                {
                    **base,
                    "decision": PENDING,
                    "parse_mode": None,
                    "failure_reason": None,
                    "generation_id": None,
                }
            )
            continue
        if r is None:
            raise MissingGenerationError(ref.text_sha256)
        r = Resolved(*r)
        rows.append(
            {
                **base,
                "decision": r.decision,
                "parse_mode": r.mode,
                "failure_reason": r.failure,
                "generation_id": generation_id(ref.text_sha256, fp),
            }
        )
    return rows


def labels_table(dataset: str, rows: list[dict]) -> pa.Table:
    schema = pa.schema([LABEL_FIELDS[0], *JOIN_KEYS[dataset], *LABEL_FIELDS[1:]])
    return pa.Table.from_pylist(rows, schema=schema)


def viewer_table(refs: list[SentenceRef], rows: list[dict], region: str) -> pa.Table:
    """The sentence next to its label: a plain table for the dataset viewer."""
    return pa.table(
        {
            "sentence": pa.array([r.text for r in refs], pa.large_string()),
            "label": pa.array([row["decision"] for row in rows], pa.string()),
            "language": pa.array([r.language for r in refs], pa.string()),
            "region": pa.array([region] * len(refs), pa.string()),
        }
    )


def _write_viewer(out: Path, input_path: str, refs: list[SentenceRef], rows: list[dict]) -> None:
    table = viewer_table(refs, rows, Path(input_path).stem)
    write_atomic(out / "viewer" / input_path, table_bytes(table))


def build_labels(  # noqa: PLR0913 - one call builds both tables of a file
    dataset: str,
    input_path: str,
    local: Path,
    *,
    resolved: Lookup,
    fp: str,
    revision: str,
    out: Path,
    allow_pending: bool = False,
) -> int:
    """Write ``out/labels/<input_path>`` and its ``out/viewer/<input_path>`` companion.

    Raises ``MissingGenerationError`` if any text is unresolved, unless ``allow_pending``
    (partial publication: those sentences are labelled ``pending``).
    """
    refs = list(SOURCES[dataset].read(local, input_path))
    rows = label_rows(refs, resolved, fp, revision, allow_pending=allow_pending)
    table = labels_table(dataset, rows)
    write_atomic(out / "labels" / input_path, table_bytes(table))
    _write_viewer(out, input_path, refs, rows)
    return table.num_rows


def build_viewer(dataset: str, input_path: str, local: Path, *, resolved: Lookup, out: Path) -> int:
    """Write only ``out/viewer/<input_path>`` (files whose labels are already published)."""
    refs = list(SOURCES[dataset].read(local, input_path))
    rows = label_rows(refs, resolved, "", "")
    _write_viewer(out, input_path, refs, rows)
    return len(rows)


def build_generations(
    store: WorkStore, fp: str, out: Path, shas: set[str], rows_per_file: int = 50_000
) -> int:
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
    write_atomic(
        out / "generations" / fp / f"part-{n:05d}.parquet", table_bytes(pa.Table.from_pylist(rows))
    )
