import pyarrow as pa

from landuse_filter.application.geo import description_cells


def cell_of(lon, lat):
    return f"{round(lon)}/{round(lat)}"


def tables():
    labels = pa.table(
        {
            "description_identity": ["i1", "i1", "i2", "i3", "i4", "i5"],
            "decision": ["yes", "no", "yes", "failed", "skipped_unsplit", "yes"],
        }
    )
    language = pa.table(
        {
            "description_identity": ["i1", "i2", "i3", "i5"],
            "osm_type": ["way", "way", "way", "relation"],
            "osm_id": [1, 2, 3, 9],
        }
    )
    polygons = pa.table(
        {
            "osm_type": ["way", "way", "way"],
            "osm_id": [1, 2, 3],
            "bbox_min_x": [0.0, 10.0, 20.0],
            "bbox_min_y": [0.0, 40.0, 50.0],
            "bbox_max_x": [2.0, 12.0, 22.0],
            "bbox_max_y": [2.0, 42.0, 52.0],
        }
    )
    return labels, language, polygons


def test_only_yes_and_no_rows_are_binned_at_the_bbox_centre():
    result = description_cells(*tables(), cell_of)
    assert result.cells == {"1/1": [1, 1], "11/41": [1, 0]}
    assert result.labelled == 4  # i1 yes, i1 no, i2 yes, i5 yes


def test_rows_of_polygons_without_a_bbox_are_counted_but_not_located():
    result = description_cells(*tables(), cell_of)
    assert result.located == 3  # i5's polygon (relation 9) has no bbox row
    assert result.labelled - result.located == 1


def test_polygons_with_missing_bbox_values_are_skipped():
    labels, language, polygons = tables()
    polygons = polygons.set_column(2, "bbox_min_x", pa.array([None, 10.0, 20.0], pa.float64()))
    result = description_cells(labels, language, polygons, cell_of)
    assert result.cells == {"11/41": [1, 0]}


def test_website_rows_are_placed_at_the_polygon_lat_lon():
    from landuse_filter.application.geo import website_cells

    labels = pa.table(
        {
            "polygon_id": ["p1", "p1", "p2", "p3", "p4"],
            "decision": ["yes", "no", "yes", "failed", "yes"],
        }
    )
    polygons = pa.table(
        {"polygon_id": ["p1", "p2", "p4"], "lat": [48.0, 10.0, None], "lon": [2.0, 20.0, 5.0]}
    )
    result = website_cells(labels, polygons, cell_of)
    assert result.cells == {"2/48": [1, 1], "20/10": [1, 0]}
    assert (result.labelled, result.located) == (4, 3)  # p4 has no coordinates


def test_website_texts_get_the_cell_of_their_polygon():
    from landuse_filter.application.geo import website_text_cells
    from landuse_filter.domain.sentences import SentenceRef

    def ref(polygon, text, unsplit=False):
        return SentenceRef(
            "w", "f", (("polygon_id", polygon), ("field", "website")), text, "en", unsplit=unsplit
        )

    refs = [ref("p1", "a"), ref("p2", "b"), ref("p4", "c"), ref("p1", "u", unsplit=True)]
    polygons = pa.table(
        {"polygon_id": ["p1", "p2", "p4"], "lat": [48.0, 10.0, None], "lon": [2.0, 20.0, 5.0]}
    )
    pairs = dict(website_text_cells(refs, polygons, cell_of))
    assert pairs == {
        refs[0].text_sha256: "2/48",
        refs[1].text_sha256: "20/10",
    }  # no cell: p4, unsplit


def test_wiki_texts_get_the_cell_of_the_first_polygon_linked_to_their_document():
    from landuse_filter.application.geo import wiki_text_cells
    from landuse_filter.domain.sentences import SentenceRef

    def ref(sentence_id, text, unsplit=False):
        return SentenceRef(
            "wiki", "f", (("sentence_id", sentence_id),), text, "en", unsplit=unsplit
        )

    refs = [ref("s1", "a"), ref("s2", "b"), ref("s3", "c"), ref("s4", "d"), ref("s5", "e", True)]
    documents = pa.table(
        {
            "sentence_id": ["s1", "s2", "s3", "s4", "s5"],
            "document_id": ["d1", "d1", "d2", "d3", "d1"],
        }
    )
    links = (
        pa.table(  # d1 is linked to two polygons: the first (by id) without coordinates is skipped
            {"polygon_id": ["p2", "p1", "p3", "p9"], "document_id": ["d1", "d1", "d2", "d3"]}
        )
    )
    polygons = pa.table(
        {"polygon_id": ["p1", "p2", "p3"], "lat": [None, 10.0, 48.0], "lon": [None, 20.0, 2.0]}
    )
    pairs = dict(wiki_text_cells(refs, documents, links, polygons, cell_of))
    assert pairs == {
        refs[0].text_sha256: "20/10",
        refs[1].text_sha256: "20/10",
        refs[2].text_sha256: "2/48",
    }  # d3's polygon has no row; the unsplit sentence is not sent to the model
