from hypothesis import given
from hypothesis import strategies as st

from landuse_filter.domain.parsing import parse_answer, parse_generation
from landuse_filter.domain.sentences import Decision

reasoning = st.text().filter(lambda s: "</think>" not in s)
label = st.sampled_from(["yes", "no"])
padding = st.text(alphabet=" \n\t.*\"'!", max_size=5)


@given(reasoning, label, padding, padding)
def test_clear_final_answer_always_parsed(think, answer, before, after):
    verdict = parse_generation(f"{think}</think>{before}{answer}{after}", truncated=False)
    assert verdict.decision == Decision(answer)


@given(st.text())
def test_truncated_is_always_failed(raw):
    assert parse_generation(raw, truncated=True).decision is Decision.FAILED


@given(st.text(), st.text())
def test_both_labels_without_lead_never_decide(a, b):
    answer = f"so {a} yes {b} no"
    verdict = parse_answer(answer)
    if verdict.decision is not Decision.FAILED:
        # Only an exact or leading match may decide, and "so" prevents both.
        assert verdict.mode is not None
        assert verdict.mode.value != "last"


@given(st.text())
def test_parser_is_total(raw):
    assert parse_generation(raw, truncated=False).decision in set(Decision) - {
        Decision.SKIPPED_UNSPLIT
    }
