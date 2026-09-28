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
        decisions={"yes": 10, "no": 5, "failed": 1, "skipped_unsplit": 2},
        admitted={
            "a100_sxm4_40gb": {
                "delta_f1": -0.0,
                "delta_mcc": -0.0014,
                "mcc_lower": -0.008,
                "delta_accuracy": 0.0016,
                "delta_failed_rate": -0.0033,
            }
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
