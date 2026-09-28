import pytest

from landuse_filter.domain.metrics import confusion, scores


def test_failed_is_its_own_bucket_and_counts_against_accuracy():
    c = confusion([("yes", "yes"), ("no", "no"), ("yes", None), ("no", "yes")])
    assert (c.tp, c.tn, c.fn, c.fp, c.failed) == (1, 1, 0, 1, 1)
    s = scores(c)
    assert s.accuracy == 0.5
    assert s.recall == 1.0  # failed items are outside the recall denominator, as in the benchmark
    assert s.failed_rate == 0.25


def test_degenerate_scores_are_zero():
    s = scores(confusion([("yes", None)]))
    assert (s.precision, s.recall, s.f1, s.mcc) == (0.0, 0.0, 0.0, 0.0)


def test_empty_refused():
    with pytest.raises(ValueError, match="empty"):
        scores(confusion([]))


def test_reproduces_published_english_metrics(reference_sample):
    # Golden values of en/LiquidAI__LFM2.5-2.6B+DSpark-throughput-b16.json are checked on
    # the full corpus in the integration suite; here the sample must at least be scored.
    s = scores(confusion([(r["expected"], r["predicted"]) for r in reference_sample]))
    assert 0.6 < s.f1 < 0.95


def test_scores_exact_values():
    outcomes = [("yes", "yes")] * 3 + [("yes", "no")] + [("no", "no")] * 4 + [("no", "yes")] * 2
    c = confusion(outcomes)
    assert (c.tp, c.fn, c.tn, c.fp, c.failed) == (3, 1, 4, 2, 0)
    s = scores(c)
    assert s.precision == pytest.approx(3 / 5)
    assert s.recall == pytest.approx(3 / 4)
    assert s.balanced_accuracy == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert s.mcc == pytest.approx(10 / 600**0.5)


def test_empty_scores_message():
    from landuse_filter.domain.metrics import Confusion

    with pytest.raises(ValueError, match=r"^cannot score an empty set of outcomes$"):
        scores(Confusion(0, 0, 0, 0, 0))
