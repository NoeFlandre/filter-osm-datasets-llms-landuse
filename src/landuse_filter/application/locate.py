"""Where the texts of an input file are: ``(text sha, H3 cell)`` pairs for geographic planning."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter.adapters.readers import WEBSITE, WIKI, read_website, read_wiki
from landuse_filter.application.geo import CellOf, website_text_cells, wiki_text_cells

Fetch = Callable[[str], Path]  # path in the input repo -> local file


def locator(
    dataset: str, fetch: Fetch, cell_of: CellOf
) -> Callable[[str, str, Path], Iterator[tuple[str, str]]]:
    """``(dataset, path, local file) -> (sha, cell)`` for the datasets that have coordinates.

    ``fetch`` downloads sibling files of the same input repo (the wiki polygon tables).
    """

    def website(_dataset: str, path: str, local: Path) -> Iterator[tuple[str, str]]:
        polygons = pq.read_table(local, columns=["polygon_id", "lat", "lon"])
        yield from website_text_cells(read_website(local, path), polygons, cell_of)

    def wiki(_dataset: str, path: str, local: Path) -> Iterator[tuple[str, str]]:
        region = Path(path).stem
        yield from wiki_text_cells(
            read_wiki(local, path),
            pq.read_table(local, columns=["sentence_id", "document_id"]),
            pq.read_table(
                fetch(f"polygon_document_links/{region}.parquet"),
                columns=["polygon_id", "document_id"],
            ),
            pq.read_table(
                fetch(f"polygons/{region}.parquet"), columns=["polygon_id", "lat", "lon"]
            ),
            cell_of,
        )

    try:
        return {WEBSITE: website, WIKI: wiki}[dataset]
    except KeyError:
        raise ValueError(f"{dataset}: no coordinates to order by") from None
