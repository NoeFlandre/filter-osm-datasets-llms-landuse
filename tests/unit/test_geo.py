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
