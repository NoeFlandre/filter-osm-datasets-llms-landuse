from pathlib import Path

import pytest

from landuse_filter.application.card import JOINS, CardFacts, render_card

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


@pytest.mark.parametrize("dataset", sorted(JOINS))
def test_every_dataset_has_a_join_recipe(dataset):
    card = render_card(facts(dataset=dataset))
    assert JOINS[dataset][1] in card


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
