"""Where the labelled sentences are: label rows -> polygon -> bounding-box centre -> H3 cell."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import pyarrow as pa

from landuse_filter.domain.geomap import bbox_centre

CellOf = Callable[[float, float], str]  # (lon, lat) -> cell id
BBOX = ("bbox_min_x", "bbox_min_y", "bbox_max_x", "bbox_max_y")


@dataclass(frozen=True, slots=True)
class Located:
    cells: dict[str, list[int]]  # cell -> [yes, no]
    labelled: int  # yes+no rows seen
    located: int  # of which placed on the map


def description_cells(
    labels: pa.Table, language: pa.Table, polygons: pa.Table, cell_of: CellOf
) -> Located:
    """Bin the ``yes``/``no`` rows of a description labels file.

    ``labels`` carries ``description_identity`` and ``decision``; ``language`` (the input
    ``language-v1`` file) maps an identity to ``osm_type``/``osm_id``; ``polygons`` (the input
    ``data`` file of the same region) holds the bounding boxes.
    """
    polygon_of = {
        i: (t, n)
        for i, t, n in zip(
            *[
                language.column(c).to_pylist()
                for c in ("description_identity", "osm_type", "osm_id")
            ],
            strict=True,
        )
    }
    cell_of_polygon: dict[tuple[str, int], str] = {}
    columns = [polygons.column(c).to_pylist() for c in ("osm_type", "osm_id", *BBOX)]
    for osm_type, osm_id, *box in zip(*columns, strict=True):
        if None not in box:
            cell_of_polygon[(osm_type, osm_id)] = cell_of(*bbox_centre(*box))
    counts: dict[str, Counter[str]] = {}
    labelled = located = 0
    identities = labels.column("description_identity").to_pylist()
    for identity, decision in zip(identities, labels.column("decision").to_pylist(), strict=True):
        if decision not in ("yes", "no"):
            continue
        labelled += 1
        cell = cell_of_polygon.get(polygon_of.get(identity, ("", -1)))
        if cell is None:
            continue
        located += 1
        counts.setdefault(cell, Counter())[decision] += 1
    cells = {c: [n["yes"], n["no"]] for c, n in sorted(counts.items())}
    return Located(cells, labelled, located)
