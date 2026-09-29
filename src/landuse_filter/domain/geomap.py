"""Pure aggregation behind the card's world map: yes/no counts per H3 cell."""

from collections.abc import Iterable, Mapping, Sequence

H3_RESOLUTION = 3  # same resolution as the input dataset's density map
MIN_SENTENCES = 10  # cells with fewer yes+no sentences are drawn grey

Counts = tuple[int, int]  # (yes, no)


def bbox_centre(min_x: float, min_y: float, max_x: float, max_y: float) -> tuple[float, float]:
    """(lon, lat) of the centre of a bounding box."""
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def merge(records: Iterable[Mapping[str, Sequence[int]]]) -> dict[str, Counts]:
    """Sum per-file ``{cell: [yes, no]}`` records."""
    total: dict[str, list[int]] = {}
    for record in records:
        for cell, (yes, no) in record.items():
            acc = total.setdefault(cell, [0, 0])
            acc[0] += yes
            acc[1] += no
    return {cell: (yes, no) for cell, (yes, no) in total.items()}


def sentences(cells: Mapping[str, Counts]) -> int:
    return sum(yes + no for yes, no in cells.values())


def global_share(cells: Mapping[str, Counts]) -> float:
    """Share of ``yes`` among all yes+no sentences of the map."""
    total = sentences(cells)
    return sum(yes for yes, _ in cells.values()) / total if total else 0.0


def cell_share(counts: Counts) -> float | None:
    """Share of ``yes`` in a cell, or ``None`` when it holds too few sentences."""
    yes, no = counts
    return yes / (yes + no) if yes + no >= MIN_SENTENCES else None
