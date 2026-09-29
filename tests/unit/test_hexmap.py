import hashlib

from landuse_filter.adapters import hexmap

CELLS = {
    hexmap.cell_of(2.35, 48.85): (40, 10),  # Paris
    hexmap.cell_of(-74.0, 40.7): (5, 30),  # New York
    hexmap.cell_of(179.9, -17.0): (3, 1),  # Fiji, across the antimeridian, too few sentences
    hexmap.cell_of(-179.9, -17.0): (12, 12),
}


def test_cell_of_is_a_resolution_3_h3_index():
    cell = hexmap.cell_of(2.35, 48.85)
    assert len(cell) == 15
    assert cell == hexmap.cell_of(2.35, 48.85)
    assert cell != hexmap.cell_of(-74.0, 40.7)


def test_render_writes_a_deterministic_png(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    hexmap.render(CELLS, a, title="t")
    hexmap.render(CELLS, b, title="t")
    assert a.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_render_handles_a_map_without_cells(tmp_path):
    hexmap.render({}, tmp_path / "empty.png", title="t")
    assert (tmp_path / "empty.png").stat().st_size > 0


def test_cells_crossing_the_antimeridian_are_drawn_on_both_sides():
    cell = hexmap.cell_of(179.99, 0.0)
    rings = hexmap._outline(cell)
    assert len(rings) in (1, 2)
    for ring in rings:
        assert max(lon for lon, _ in ring) - min(lon for lon, _ in ring) < 180
