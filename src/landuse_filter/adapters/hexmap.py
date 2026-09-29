"""H3 cells and the yes-share world map (h3 and matplotlib are imported lazily)."""

import json
from collections.abc import Mapping
from pathlib import Path

from landuse_filter.domain.geomap import H3_RESOLUTION, MIN_SENTENCES, cell_share, global_share

LAND = Path(__file__).parent / "data" / "land_110m.json"
OCEAN, LAND_COLOUR, NO_DATA = "#cfe2f3", "#ece6d8", "#b5b5b5"
SIZE, DPI = (16, 8), 100
HALF_TURN = 180  # a ring wider than this in longitude crosses the antimeridian


def cell_of(lon: float, lat: float) -> str:
    import h3

    return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)


def _outline(cell: str) -> list[list[tuple[float, float]]]:
    """(lon, lat) ring of a cell; one copy per side when it crosses the antimeridian."""
    import h3

    ring = [(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]
    lons = [lon for lon, _ in ring]
    if max(lons) - min(lons) <= HALF_TURN:
        return [ring]
    east = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in ring]
    return [east, [(lon - 360, lat) for lon, lat in east]]


def render(cells: Mapping[str, tuple[int, int]], out: Path, *, title: str) -> None:
    """Write the map: one hexagon per cell, coloured by its share of ``yes``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Patch

    centre = global_share(cells)
    norm = (
        TwoSlopeNorm(vcenter=centre, vmin=0.0, vmax=1.0)
        if 0.0 < centre < 1.0
        else Normalize(0.0, 1.0)
    )
    cmap = plt.get_cmap("RdBu")
    polygons, colours = [], []
    for cell in sorted(cells):
        share = cell_share(cells[cell])
        colour = NO_DATA if share is None else cmap(norm(share))
        for ring in _outline(cell):
            polygons.append(ring)
            colours.append(colour)
    fig, ax = plt.subplots(figsize=SIZE, dpi=DPI)
    ax.set_facecolor(OCEAN)
    land = json.loads(LAND.read_text(encoding="utf-8"))["rings"]
    ax.add_collection(PolyCollection(land, facecolors=LAND_COLOUR, edgecolors="none"))
    ax.add_collection(PolyCollection(polygons, facecolors=colours, edgecolors="none"))
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude", title=title)
    ax.set_aspect("equal")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.02)
    bar.set_label(f"share of yes among yes+no sentences (dataset-wide {centre:.1%})")
    ax.legend(
        handles=[Patch(color=NO_DATA, label=f"fewer than {MIN_SENTENCES} yes+no sentences")],
        loc="lower left",
    )
    fig.savefig(out, format="png", metadata={"Software": None})
    plt.close(fig)
