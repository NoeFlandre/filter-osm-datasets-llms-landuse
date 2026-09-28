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
