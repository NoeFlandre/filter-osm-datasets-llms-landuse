"""Streaming readers turning input parquet shards into ``SentenceRef`` rows.

Each reader projects only the columns it needs and fails loudly on a segmentation
status it does not know, so a schema change upstream cannot silently drop sentences.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter.domain.sentences import SentenceRef

BATCH_ROWS = 4096


class UnknownStatusError(ValueError):
    """An upstream segmentation status this reader was not written for."""


def _batches(path: Path, columns: list[str]) -> Iterator[dict]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
        yield from batch.to_pylist()


def _check(status: str | None, known: frozenset[str], where: str) -> None:
    if status not in known:
        raise UnknownStatusError(f"{where}: unknown segmentation status {status!r}")


# --- osm-polygon-description-tag (config language-v1) ------------------------------

DESCRIPTION = "osm-polygon-description-tag"
_DESC_SPLIT = frozenset({"split"})
_DESC_UNSPLIT = frozenset({"unsupported_language", "not_detected"})


def read_description(path: Path, source_file: str) -> Iterator[SentenceRef]:
    cols = [
        "description_identity",
        "tag_key",
        "original_text",
        "language_code",
        "split_status",
        "sentences",
    ]
    for row in _batches(path, cols):
        status = row["split_status"]
        _check(status, _DESC_SPLIT | _DESC_UNSPLIT, f"{source_file}:{row['description_identity']}")
        base = (("description_identity", row["description_identity"]), ("tag_key", row["tag_key"]))
        if status in _DESC_UNSPLIT:
            yield SentenceRef(
                DESCRIPTION,
                source_file,
                (*base, ("sentence_index", 0)),
                row["original_text"],
                row["language_code"],
                unsplit=True,
            )
            continue
        for i, text in enumerate(row["sentences"]):
            yield SentenceRef(
                DESCRIPTION, source_file, (*base, ("sentence_index", i)), text, row["language_code"]
            )


# --- osm-polygon-wikidata-and-wikipedia (wikipedia/ and wikivoyage/ sentences) -----

WIKI = "osm-polygon-wikidata-and-wikipedia"
_WIKI_SPLIT = frozenset({"split"})
_WIKI_UNSPLIT = frozenset({"unsupported_language"})


def read_wiki(path: Path, source_file: str) -> Iterator[SentenceRef]:
    for row in _batches(path, ["sentence_id", "language", "text", "segmentation_status"]):
        status = row["segmentation_status"]
        _check(status, _WIKI_SPLIT | _WIKI_UNSPLIT, f"{source_file}:{row['sentence_id']}")
        yield SentenceRef(
            WIKI,
            source_file,
            (("sentence_id", row["sentence_id"]),),
            row["text"],
            row["language"],
            unsplit=status in _WIKI_UNSPLIT,
        )


# --- osm-polygon-website-tag (polygons/) --------------------------------------------

WEBSITE = "osm-polygon-website-tag"
WEBSITE_FIELDS = ("website", "contact_website")
_WEB_SPLIT = frozenset({"success"})
_WEB_UNSPLIT = frozenset({"unsupported_language"})
# No sentences to label: no text fetched, or an empty page (regression: bayern-latest,
# way/365496611, status "empty_text", website planning job 2070649).
_WEB_ABSENT = frozenset({"absent", "empty_text"})


def read_website(path: Path, source_file: str) -> Iterator[SentenceRef]:
    cols = ["polygon_id"] + [
        f"{f}_{c}"
        for f in WEBSITE_FIELDS
        for c in ("text", "language", "sentences", "sentence_status")
    ]
    for row in _batches(path, cols):
        for field in WEBSITE_FIELDS:
            yield from _website_field(row, field, source_file)


def _website_field(row: dict, field: str, source_file: str) -> Iterator[SentenceRef]:
    status = row[f"{field}_sentence_status"]
    _check(
        status,
        _WEB_SPLIT | _WEB_UNSPLIT | _WEB_ABSENT,
        f"{source_file}:{row['polygon_id']}:{field}",
    )
    base = (("polygon_id", row["polygon_id"]), ("field", field))
    language = row[f"{field}_language"]
    if status in _WEB_UNSPLIT:
        yield SentenceRef(
            WEBSITE,
            source_file,
            (*base, ("sentence_index", 0)),
            row[f"{field}_text"],
            language,
            unsplit=True,
        )
    elif status in _WEB_SPLIT:
        for i, text in enumerate(row[f"{field}_sentences"]):
            yield SentenceRef(WEBSITE, source_file, (*base, ("sentence_index", i)), text, language)


# --- registry ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Source:
    """An input dataset: where its in-scope shards live and how to read them."""

    dataset: str
    repo_id: str
    patterns: tuple[str, ...]
    read: Callable[[Path, str], Iterator[SentenceRef]]


SOURCES = {
    DESCRIPTION: Source(
        DESCRIPTION, f"NoeFlandre/{DESCRIPTION}", ("language-v1/data/*.parquet",), read_description
    ),
    WIKI: Source(
        WIKI,
        f"NoeFlandre/{WIKI}",
        ("wikipedia/sentences/*.parquet", "wikivoyage/sentences/*.parquet"),
        read_wiki,
    ),
    WEBSITE: Source(WEBSITE, f"NoeFlandre/{WEBSITE}", ("polygons/*.parquet",), read_website),
}
DATASET_RANK = {DESCRIPTION: 0, WIKI: 1, WEBSITE: 2}
