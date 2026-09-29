"""Where the labelled sentences are: label rows -> polygon -> bounding-box centre -> H3 cell."""

from collections import Counter
from collections.abc import Callable, Iterable
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


def _bin(pairs: Iterable[tuple[str, str | None]]) -> Located:
    """Count ``yes``/``no`` rows per cell; ``(decision, None)`` rows are not located."""
    counts: dict[str, Counter[str]] = {}
    labelled = located = 0
    for decision, cell in pairs:
        if decision not in ("yes", "no"):
            continue
        labelled += 1
        if cell is None:
            continue
        located += 1
        counts.setdefault(cell, Counter())[decision] += 1
    return Located({c: [n["yes"], n["no"]] for c, n in sorted(counts.items())}, labelled, located)


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
    identities = labels.column("description_identity").to_pylist()
    decisions = labels.column("decision").to_pylist()
    return _bin(
        (d, cell_of_polygon.get(polygon_of.get(i, ("", -1))))
        for i, d in zip(identities, decisions, strict=True)
    )


def website_cells(labels: pa.Table, polygons: pa.Table, cell_of: CellOf) -> Located:
    """Bin the ``yes``/``no`` rows of a website labels file at the polygon's ``lat``/``lon``.

    ``labels`` carries ``polygon_id`` and ``decision``; ``polygons`` is the input file of the
    same path with ``polygon_id``, ``lat`` and ``lon``.
    """
    cell_of_polygon = {
        i: cell_of(lon, lat)
        for i, lat, lon in zip(
            *[polygons.column(c).to_pylist() for c in ("polygon_id", "lat", "lon")], strict=True
        )
        if lat is not None and lon is not None
    }
    ids = labels.column("polygon_id").to_pylist()
    return _bin(
        (d, cell_of_polygon.get(i))
        for i, d in zip(ids, labels.column("decision").to_pylist(), strict=True)
    )
