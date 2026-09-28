import pytest

from landuse_filter.domain.parsing import Failure, ParseMode, parse_answer, parse_generation
from landuse_filter.domain.sentences import Decision


@pytest.mark.parametrize(
    ("raw", "decision", "mode"),
    [
        ("reasoning says yes and no</think>yes", Decision.YES, ParseMode.EXACT),
        ("x</think>\n\nno", Decision.NO, ParseMode.EXACT),
        ("x</think>**Yes**.", Decision.YES, ParseMode.EXACT),
        ('x</think>"no"', Decision.NO, ParseMode.EXACT),
        ("x</think>yes, it shows fields", Decision.YES, ParseMode.LEADING),
        ("x</think>The answer is no.", Decision.NO, ParseMode.LAST),
        ("a</think>b</think>no", Decision.NO, ParseMode.EXACT),
    ],
)
def test_parses_answer_after_last_think_close(raw, decision, mode):
    verdict = parse_generation(raw, truncated=False)
    assert (verdict.decision, verdict.mode, verdict.failure) == (decision, mode, None)


@pytest.mark.parametrize(
    ("raw", "truncated", "failure"),
    [
        ("</think>yes", True, Failure.TRUNCATED),
        ("still thinking about yes", False, Failure.UNCLOSED_THINK),
        ("x</think>   ", False, Failure.EMPTY),
        ("x</think>The answer is yes or no.", False, Failure.AMBIGUOUS),
        ("x</think>oui", False, Failure.NON_ENGLISH_TOKEN),
        ("x</think>maybe", False, Failure.NO_LABEL),
    ],
)
def test_failures_are_explicit(raw, truncated, failure):
    verdict = parse_generation(raw, truncated=truncated)
    assert verdict.decision is Decision.FAILED
    assert verdict.failure is failure
    assert verdict.mode is None


def test_rubric_mentions_in_reasoning_never_decide():
    assert (
        parse_generation("Answer yes for ... answer no for ...", truncated=False).decision
        is Decision.FAILED
    )


def test_label_inside_word_is_not_a_label():
    assert parse_answer("nobody knows").failure is Failure.NO_LABEL


def test_reproduces_every_reference_decision(reference_sample):
    for row in reference_sample:
        verdict = parse_generation(row["tail"], truncated=row["truncated"])
        expected = row["predicted"] or "failed"
        assert verdict.decision == expected, row["item_id"]
