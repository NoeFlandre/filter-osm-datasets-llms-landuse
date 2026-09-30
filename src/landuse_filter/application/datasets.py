"""Everything that differs between the input datasets, declared once.

``SPECS[dataset]`` bundles the reader and shards (:class:`~landuse_filter.adapters.readers.Source`),
the join keys of the labels table, how the dataset card joins and places the labels, and the
location capabilities (planning order, published map). Adding a dataset is one entry here.
"""

from dataclasses import dataclass

import pyarrow as pa

from landuse_filter.adapters.readers import DESCRIPTION, SOURCES, WEBSITE, WIKI, Source
from landuse_filter.application.locators import (
    MapFactory,
    PlanningFactory,
    description_map,
    website_map,
    website_planning,
    wiki_map,
    wiki_planning,
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    source: Source
    join_keys: tuple[tuple[str, pa.DataType], ...]  # identify a sentence in the labels table
    card_using: str  # SQL join condition between the input and labels on the card
    card_keys: str  # the join keys, as the card words them
    placement: str  # completes "holding the yes/no sentences ..." on the card's map
    text_license: str | None = None
    planning_locator: PlanningFactory | None = None  # orders the plan geographically
    map_locator: MapFactory | None = None  # draws the published yes-share map


SPECS = {
    DESCRIPTION: DatasetSpec(
        SOURCES[DESCRIPTION],
        (
            ("description_identity", pa.string()),
            ("tag_key", pa.string()),
            ("sentence_index", pa.int32()),
        ),
        "USING (description_identity, tag_key)",
        "description_identity, tag_key, sentence_index",
        "of the polygons whose bounding-box centre falls in it",
        map_locator=description_map,
    ),
    WIKI: DatasetSpec(
        SOURCES[WIKI],
        (("sentence_id", pa.string()),),
        "USING (sentence_id)",
        "sentence_id",
        "whose document's first linked polygon (smallest `polygon_id` with lat/lon) falls in it",
        text_license=(
            "Wikipedia and Wikivoyage text is CC BY-SA 4.0; attribution columns are kept "
            "in the mirrored input."
        ),
        planning_locator=wiki_planning,
        map_locator=wiki_map,
    ),
    WEBSITE: DatasetSpec(
        SOURCES[WEBSITE],
        (
            ("polygon_id", pa.string()),
            ("field", pa.string()),
            ("sentence_index", pa.int32()),
        ),
        "USING (polygon_id)",
        "polygon_id, field (website | contact_website), sentence_index",
        "of the polygons whose `lat`/`lon` falls in it",
        planning_locator=website_planning,
        map_locator=website_map,
    ),
}
