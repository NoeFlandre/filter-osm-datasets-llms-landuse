"""Where the rows of an input dataset are: locators for planning and for the published map.

Planning locators yield ``(text sha, H3 cell)`` pairs of an input file; map locators count
the ``yes``/``no`` rows of a published labels file per cell. Which dataset has which is
declared once, in :mod:`landuse_filter.application.datasets`.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter.adapters.hub import Hub
from landuse_filter.adapters.readers import read_website, read_wiki
from landuse_filter.application import published_stats
from landuse_filter.application.geo import (
    BBOX,
    CellOf,
    Located,
    description_cells,
    website_cells,
    website_text_cells,
    wiki_cells,
    wiki_text_cells,
)

Fetch = Callable[[str], Path]  # path in the input repo -> local file
PlanningLocator = Callable[[str, str, Path], Iterator[tuple[str, str]]]  # dataset, path, file
PlanningFactory = Callable[[Fetch, CellOf], PlanningLocator]
MapFactory = Callable[[str, Hub, str], published_stats.Locator]  # input repo, hub, revision


def website_planning(fetch: Fetch, cell_of: CellOf) -> PlanningLocator:  # noqa: ARG001
    def locate(_dataset: str, path: str, local: Path) -> Iterator[tuple[str, str]]:
        polygons = pq.read_table(local, columns=["polygon_id", "lat", "lon"])
        yield from website_text_cells(read_website(local, path), polygons, cell_of)

    return locate


def wiki_planning(fetch: Fetch, cell_of: CellOf) -> PlanningLocator:
    """``fetch`` downloads the sibling polygon tables of the same input repo."""

    def locate(_dataset: str, path: str, local: Path) -> Iterator[tuple[str, str]]:
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

    return locate


def _cell_of() -> CellOf:
    from landuse_filter.adapters import hexmap

    return hexmap.cell_of


def website_map(input_repo: str, hub: Hub, revision: str) -> published_stats.Locator:
    def locate(path: str, labels: published_stats.Source) -> Located:
        rel = path.removeprefix("labels/")
        return website_cells(
            pq.read_table(labels, columns=["polygon_id", "decision"]),
            pq.read_table(
                hub.open_file(input_repo, rel, revision), columns=["polygon_id", "lat", "lon"]
            ),
            _cell_of(),
        )

    return locate


def description_map(input_repo: str, hub: Hub, revision: str) -> published_stats.Locator:
    def locate(path: str, labels: published_stats.Source) -> Located:
        rel = path.removeprefix("labels/")
        language_file = hub.open_file(input_repo, rel, revision)
        polygon_file = hub.open_file(input_repo, f"data/{Path(rel).name}", revision)
        return description_cells(
            pq.read_table(labels, columns=["description_identity", "decision"]),
            pq.read_table(language_file, columns=["description_identity", "osm_type", "osm_id"]),
            pq.read_table(polygon_file, columns=["osm_type", "osm_id", *BBOX]),
            _cell_of(),
        )

    return locate


def wiki_map(input_repo: str, hub: Hub, revision: str) -> published_stats.Locator:
    def locate(path: str, labels: published_stats.Source) -> Located:
        rel = path.removeprefix("labels/")
        region = Path(rel).stem
        return wiki_cells(
            pq.read_table(labels, columns=["sentence_id", "decision"]),
            pq.read_table(
                hub.open_file(input_repo, rel, revision), columns=["sentence_id", "document_id"]
            ),
            pq.read_table(
                hub.open_file(input_repo, f"polygon_document_links/{region}.parquet", revision),
                columns=["polygon_id", "document_id"],
            ),
            pq.read_table(
                hub.open_file(input_repo, f"polygons/{region}.parquet", revision),
                columns=["polygon_id", "lat", "lon"],
            ),
            _cell_of(),
        )

    return locate
