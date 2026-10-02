"""Exact-string tests for the dataset card helpers, written to pin them under mutation testing."""

from dataclasses import replace

from landuse_filter.application import card
from landuse_filter.application.card import CardFacts

GATE = {
    "delta_f1": -0.0018,
    "delta_mcc": -0.006,
    "mcc_lower": -0.0123,
    "delta_accuracy": 0.0016,
    "delta_failed_rate": 0.0123,
}

BASE = CardFacts(
    dataset="osm-polygon-description-tag",
    revision="b4706eb1234",
    model="org/model@0123456789",
    draft="org/draft",
    max_new_tokens=4096,
    fingerprint="fp",
    labelled_files=1,
    total_files=2,
    decisions={"yes": 1},
    failures={},
    unique_texts=0,
    gpu_rows={},
    admitted={},
)


def test_from_one_glob_is_a_quoted_path_and_several_make_read_parquet():
    assert card._from(("a/*.parquet",)) == "'a/*.parquet'"
    assert card._from(("a/*.parquet",), "labels/") == "'labels/a/*.parquet'"
    assert card._from(("a", "b"), "p/") == "read_parquet(['p/a', 'p/b'])"


def test_model_reference_shows_a_seven_character_revision_only_when_given():
    assert card._ref("org/name") == "`org/name`"
    assert card._ref("org/name@0123456789") == "`org/name` (revision `0123456`)"
    assert card._ref("org/na@me@0123456789") == "`org/na` (revision `me@0123`)"


def test_gate_row_is_formatted_to_the_published_precision():
    assert card._gate_row("l40s", 12345, 0.25, GATE) == (
        "| `l40s` | 12,345 (25.0%) | -0.0018 | -0.0060 (-0.0123) | +0.0016 | +1.23 pp |"
    )


def test_decision_table_lists_absent_decisions_as_zero_and_hides_empty_pending():
    assert card._decision_table({"yes": 1, "no": 3}) == "\n".join(
        [
            "| `yes` | 1 | 25.0% | relevant to land use / land cover |",
            "| `no` | 3 | 75.0% | not relevant |",
            "| `failed` | 0 | 0.0% | the model output could not be parsed into yes/no |",
            "| `skipped_unsplit` | 0 | 0.0% | text not segmented upstream; not sent to the model |",
            "| **total** | **4** | | |",
        ]
    )
    with_pending = card._decision_table({"yes": 1, "pending": 1})
    assert (
        "| `pending` | 1 | 50.0% | in a partly labelled file; no model answer yet |" in with_pending
    )


def test_failure_table_sorts_by_count_then_name_and_leaves_unknown_reasons_blank():
    assert card._failure_table({"weird": 1, "truncated": 2, "empty": 1}) == "\n".join(
        [
            "| `truncated` | 2 | 50.0% | hit the token limit before answering |",
            "| `empty` | 1 | 25.0% | nothing after the reasoning |",
            "| `weird` | 1 | 25.0% |  |",
        ]
    )


def test_generations_config_only_exists_once_there_are_generations():
    assert card._generations_config(BASE) == ""
    assert card._generations_config(replace(BASE, unique_texts=1)) == (
        '- config_name: generations\n  data_files: "generations/**/*.parquet"\n'
    )


def test_texts_clause_depends_on_published_generations():
    zero = "generations are published with each completed input file"
    assert card._texts_clause(BASE, "in_progress") == zero
    assert card._texts_clause(BASE, "complete") == zero
    many = replace(BASE, unique_texts=1234)
    assert card._texts_clause(many, "complete") == "1,234 unique texts sent to the model"
    partial = card._texts_clause(many, "in_progress")
    assert partial.startswith("1,234 unique texts have their full model outputs published so far")
    assert "first complete file" in partial
    assert "`labels/`" in partial
    assert "sent to the model" not in partial


def test_intro_shows_a_seven_character_dataset_revision():
    assert "(revision `b4706eb`)" in card._intro(BASE, "in_progress", 1)


def test_failed_section_shares_count_failed_rows_and_defaults_to_zero():
    none = card._failed_section(BASE, 1)
    assert none == ""
    text = card._failed_section(replace(BASE, failures={"truncated": 1}), 1)
    assert "1 rows (0.0% of all rows) have no" in text
    text = card._failed_section(
        replace(BASE, decisions={"yes": 3, "failed": 1}, failures={"truncated": 1}), 4
    )
    assert "1 rows (25.0% of all rows) have no" in text


def test_tables_section_explains_pending_rows_only_for_partial_publications():
    note = (
        "\n* `pending` rows belong to partly labelled files and are only in "
        "`labels/`, never in the viewer; their labels are refreshed as the "
        "model works, and the `generations/` rows of a file are published "
        "once the file is complete."
    )
    assert note in card._tables_section(replace(BASE, partial_files=1))
    assert "pending" not in card._tables_section(BASE)
