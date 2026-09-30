from pathlib import Path

import pytest

from landuse_filter.application.card import CardFacts, MapFacts, render_card
from landuse_filter.application.datasets import SPECS

GOLDEN = Path(__file__).parents[1] / "fixtures" / "card_description.md"


def facts(dataset="osm-polygon-description-tag", labelled=3, total=386):
    return CardFacts(
        dataset=dataset,
        revision="b4706eb",
        model="LiquidAI/LFM2.5-2.6B@654f",
        draft="LiquidAI/LFM2.5-2.6B-DSpark@458c",
        max_new_tokens=4096,
        fingerprint="71dd8471f52321ab",
        labelled_files=labelled,
        total_files=total,
        decisions={"yes": 10, "no": 5, "failed": 3, "skipped_unsplit": 2},
        failures={"truncated": 2, "no_label": 1},
        unique_texts=12,
        gpu_rows={"a100_sxm4_40gb": 9, "l40s": 3},
        admitted={
            "a100_sxm4_40gb": {
                "delta_f1": -0.0,
                "delta_mcc": -0.0014,
                "mcc_lower": -0.008,
                "delta_accuracy": 0.0016,
                "delta_failed_rate": -0.0033,
            },
            "l40s": {
                "delta_f1": -0.0018,
                "delta_mcc": -0.0060,
                "mcc_lower": -0.0123,
                "delta_accuracy": -0.0018,
                "delta_failed_rate": -0.0016,
            },
            "a40": {  # admitted but never used: must not appear
                "delta_f1": -0.001,
                "delta_mcc": -0.001,
                "mcc_lower": -0.016,
                "delta_accuracy": -0.001,
                "delta_failed_rate": 0.0,
            },
        },
    )


def test_card_matches_golden():
    assert render_card(facts()) == GOLDEN.read_text(encoding="utf-8")


def test_status_follows_coverage():
    assert "dataset_status: in_progress" in render_card(facts())
    assert "dataset_status: complete" in render_card(facts(labelled=386))


@pytest.mark.parametrize("dataset", sorted(SPECS))
def test_every_dataset_has_a_join_recipe(dataset):
    card = render_card(facts(dataset=dataset))
    assert SPECS[dataset].card_using in card


def test_wiki_card_states_cc_by_sa():
    assert "CC BY-SA" in render_card(facts(dataset="osm-polygon-wikidata-and-wikipedia"))


def test_card_numbers_come_from_the_facts_and_percentages_are_exact():
    card = render_card(facts())
    assert "| `yes` | 10 | 50.0% |" in card
    assert "| **total** | **20** |" in card
    assert "12 unique texts" in card


def test_card_lists_only_gpu_types_that_generated_rows():
    card = render_card(facts())
    assert "`a100_sxm4_40gb` | 9 (75.0%)" in card
    assert "`l40s` | 3 (25.0%)" in card
    assert "`a40`" not in card


def test_card_is_deterministic():
    assert render_card(facts()) == render_card(facts())


def test_card_without_failures_has_no_failed_section():
    f = facts()
    f = CardFacts(**{**{k: getattr(f, k) for k in f.__slots__}, "failures": {}})
    assert "### Failed rows" not in render_card(f)


@pytest.mark.parametrize("dataset", sorted(SPECS))
def test_join_recipe_reads_every_input_table_of_the_dataset(dataset):
    """Wiki labels cover wikipedia/ and wikivoyage/ sentences: the recipe must read both."""
    from landuse_filter.adapters.readers import SOURCES

    card = render_card(facts(dataset=dataset))
    for pattern in SOURCES[dataset].patterns:
        assert f"'{pattern}'" in card
        assert f"'labels/{pattern}'" in card


def test_map_section_shows_computed_figures_and_is_optional():
    f = facts()
    with_map = CardFacts(
        **{k: getattr(f, k) for k in f.__slots__ if k != "world_map"},
        world_map=MapFacts(cells=1234, located=950, labelled=1000, share=0.6861),
    )
    card = render_card(with_map)
    assert "![Share of yes among yes/no sentences per H3 cell](assets/yes_share_map.png)" in card
    assert "dataset-wide share (68.6%)" in card
    assert "950 of 1,000 `yes`/`no` sentences (95.0%) are placed, in 1,234 cells" in card
    assert "Where the labels are" not in render_card(facts())


def test_the_sentence_viewer_is_the_default_and_first_config():
    """The dataset viewer opens the first config marked default: sentence + label only."""
    front = render_card(facts()).split("---")[1]
    configs = front[front.index("configs:") :]
    assert configs.index("config_name: sentences") < configs.index("config_name: labels")
    assert "default: true" in configs.split("config_name: labels")[0]
    assert 'data_files: "viewer/**/*.parquet"' in configs


def test_partial_publication_is_stated_and_pending_rows_are_counted():
    import dataclasses

    partial = dataclasses.replace(
        facts(),
        decisions={"yes": 10, "no": 5, "failed": 3, "skipped_unsplit": 2, "pending": 7},
        partial_files=4,
    )
    card = render_card(partial)
    assert "| `pending` | 7 |" in card
    assert "3 of 386 input files fully labelled" in card
    assert "4 more partially" in card
    assert "published once the file is complete" in card


def test_no_pending_row_when_nothing_is_pending():
    assert "`pending`" not in render_card(facts())


def test_wiki_map_section_says_how_sentences_are_placed():
    f = facts(dataset="osm-polygon-wikidata-and-wikipedia")
    card = render_card(
        CardFacts(
            **{k: getattr(f, k) for k in f.__slots__ if k != "world_map"},
            world_map=MapFacts(cells=2, located=3, labelled=4, share=0.5),
        )
    )
    assert "first linked polygon" in card


def test_generations_config_and_count_only_when_generations_are_published():
    import dataclasses

    none = render_card(dataclasses.replace(facts(), unique_texts=0, gpu_rows={}, admitted={}))
    assert "config_name: generations" not in none  # an empty glob breaks the Hub viewer config
    assert "0 unique texts" not in none
    assert "generations are published with each completed input file" in none
    assert "config_name: generations" in render_card(facts())
