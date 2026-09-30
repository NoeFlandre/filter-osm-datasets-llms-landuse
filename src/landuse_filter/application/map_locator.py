"""Where the rows of a published labels file are, for datasets whose input has coordinates."""

from pathlib import Path

import pyarrow.parquet as pq

from landuse_filter.adapters.hub import Hub
from landuse_filter.adapters.readers import DESCRIPTION, SOURCES, WEBSITE
from landuse_filter.application import published_stats
from landuse_filter.application.geo import BBOX, Located, description_cells, website_cells


def map_locator(dataset: str, revision: str, hub: Hub) -> published_stats.Locator | None:
    """A locator reading the pinned input through ``hub``; ``None`` without coordinates."""
    if dataset not in (DESCRIPTION, WEBSITE):
        return None
    input_repo = SOURCES[dataset].repo_id

    def locate(path: str, labels_source: published_stats.Source) -> Located:
        from landuse_filter.adapters import hexmap

        rel = path.removeprefix("labels/")
        if dataset == WEBSITE:
            return website_cells(
                pq.read_table(labels_source, columns=["polygon_id", "decision"]),
                pq.read_table(
                    hub.open_file(input_repo, rel, revision), columns=["polygon_id", "lat", "lon"]
                ),
                hexmap.cell_of,
            )
        language_file = hub.open_file(input_repo, rel, revision)
        polygon_file = hub.open_file(input_repo, f"data/{Path(rel).name}", revision)
        return description_cells(
            pq.read_table(labels_source, columns=["description_identity", "decision"]),
            pq.read_table(language_file, columns=["description_identity", "osm_type", "osm_id"]),
            pq.read_table(polygon_file, columns=["osm_type", "osm_id", *BBOX]),
            hexmap.cell_of,
        )

    return locate
