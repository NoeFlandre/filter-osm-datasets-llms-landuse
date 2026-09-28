import random

import pytest

from landuse_filter.domain.noninferiority import PairedItem, evaluate_gate


def items(n_lang=3, n=120, seed=1):
    rng = random.Random(seed)
    out = []
    for lang in range(n_lang):
        for _ in range(n):
            exp = rng.choice(["yes", "no"])
            pred = exp if rng.random() < 0.8 else ("no" if exp == "yes" else "yes")
            out.append(PairedItem(f"l{lang}", exp, pred, pred))
    return out


def test_identical_predictions_pass_with_zero_delta():
    result = evaluate_gate(items(), resamples=500)
    assert result.passed
    assert result.delta_f1 == 0.0
    assert result.agreement == 1.0


def test_systematic_degradation_fails():
    base = items()
    worse = [
        PairedItem(
            i.language,
            i.expected,
            i.reference,
            None if k % 5 == 0 and i.reference == i.expected else i.candidate,
        )
        for k, i in enumerate(base)
    ]
    result = evaluate_gate(worse, resamples=500)
    assert not result.passed
    assert result.reasons


def test_single_language_collapse_is_blocking():
    base = items()
    worse = [
        PairedItem(
            i.language, i.expected, i.reference, "yes" if i.language == "l0" else i.candidate
        )
        for i in base
    ]
    result = evaluate_gate(worse, resamples=300)
    assert result.worst_language == "l0"
    assert not result.passed


def test_deterministic_given_seed():
    a = evaluate_gate(items(), resamples=200, seed=3)
    b = evaluate_gate(items(), resamples=200, seed=3)
    assert a == b


def test_empty_refused():
    with pytest.raises(ValueError, match="no paired"):
        evaluate_gate([])


def test_failing_hard_items_cannot_game_f1():
    """Regression: the benchmark's F1/MCC ignore failed items, so failing on *wrong*
    reference answers raises them. The failed-rate guard must block that."""
    base = items(n=300)
    gamed = [
        PairedItem(
            i.language, i.expected, i.reference, None if i.reference != i.expected else i.candidate
        )
        for i in base
    ]
    result = evaluate_gate(gamed, resamples=300)
    assert result.delta_f1 > 0
    assert result.delta_accuracy == 0
    assert not result.passed
    assert any("failed-rate" in r for r in result.reasons)


def test_failing_correct_items_is_caught_by_accuracy():
    base = items(n=300)
    worse = [
        PairedItem(
            i.language,
            i.expected,
            i.reference,
            None if k % 12 == 0 and i.reference == i.expected else i.candidate,
        )
        for k, i in enumerate(base)
    ]
    result = evaluate_gate(worse, resamples=300)
    assert any("accuracy" in r for r in result.reasons)


def _cells(expected, predicted):
    import numpy as np

    from landuse_filter.domain.noninferiority import _codes

    return _codes(np.array([e == "yes" for e in expected]), predicted)


def _metrics_of(expected, predicted):
    import numpy as np

    from landuse_filter.domain.noninferiority import _metrics

    idx = np.arange(len(expected))[None, :]
    return tuple(float(m[0]) for m in _metrics(_cells(expected, predicted), idx))


def test_metrics_exact_values():
    # tp=3 fn=2 tn=4 fp=3 failed=2
    expected = ["yes"] * 5 + ["no"] * 7 + ["yes", "no"]
    predicted = ["yes"] * 3 + ["no"] * 2 + ["no"] * 4 + ["yes"] * 3 + [None, None]
    f1, mcc, failed, accuracy = _metrics_of(expected, predicted)
    precision, recall = 3 / 6, 3 / 5
    assert f1 == pytest.approx(2 * precision * recall / (precision + recall))
    assert mcc == pytest.approx((12 - 6) / (6 * 5 * 7 * 6) ** 0.5)
    assert failed == pytest.approx(2 / 14)
    assert accuracy == pytest.approx(7 / 14)


def test_metrics_small_counts():
    assert _metrics_of(["yes", "no"], ["yes", "no"]) == (1.0, 1.0, 0.0, 1.0)
    assert _metrics_of(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"]) == (
        0.5,
        0.0,
        0.0,
        0.5,
    )


def test_metrics_degenerate_cells_are_zero_without_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _metrics_of(["no", "no"], ["no", "no"]) == (0.0, 0.0, 0.0, 1.0)


def test_reasons_margins_are_inclusive_and_language_drop_strict():
    import numpy as np

    from landuse_filter.domain.noninferiority import _reasons

    at_margin = _reasons(
        np.array([-0.01, -0.01, 0.0, -0.01]), np.array([0.0, 0.0, 0.005, 0.0]), ("de", -0.05)
    )
    assert len(at_margin) == 4
    assert not any("language" in r for r in at_margin)
    inside = _reasons(
        np.array([-0.009, -0.009, 0.0, -0.009]), np.array([0, 0, 0.0049, 0]), ("de", 0)
    )
    assert inside == []


def test_reasons_messages():
    import numpy as np

    from landuse_filter.domain.noninferiority import _reasons

    reasons = _reasons(
        np.array([-0.02, -0.03, 0.5, -0.04]), np.array([0.9, 0.9, 0.006, 0.7]), ("de", -0.06)
    )
    assert reasons == [
        "macro-F1 lower bound -0.0200 <= -0.01",
        "macro-MCC lower bound -0.0300 <= -0.01",
        "macro-accuracy lower bound -0.0400 <= -0.01",
        "failed-rate upper bound 0.0060 >= +0.005",
        "language de F1 drop -0.0600 > 0.05",
    ]


def uniform(language, expected, reference, candidate, n=4):
    return [PairedItem(language, expected, reference, candidate)] * n


def test_gate_exact_result_on_bootstrap_invariant_groups():
    # Every item of a language is identical, so each resample equals the point estimate.
    gate_items = uniform("a", "yes", "yes", None) + uniform("b", "no", "yes", "no")
    result = evaluate_gate(gate_items, resamples=20)
    assert result.delta_f1 == -0.5
    assert result.delta_mcc == 0.0
    assert result.delta_failed_rate == 0.5
    assert result.delta_accuracy == 0.0
    assert result.f1_lower == pytest.approx(-0.5)
    assert result.mcc_lower == 0.0
    assert result.accuracy_lower == pytest.approx(0.0)
    assert result.failed_rate_upper == pytest.approx(0.5)
    assert result.agreement == 0.0
    assert result.worst_language == "a"
    assert result.worst_language_delta_f1 == -1.0
    assert not result.passed
    assert len(result.reasons) == 3


def test_worst_language_is_by_f1_and_first_on_ties():
    tie = uniform("a", "yes", "yes", None) + uniform("b", "yes", "yes", None)
    assert evaluate_gate(tie, resamples=5).worst_language == "a"
    f1_drop = [PairedItem("a", "yes", "yes", "yes"), PairedItem("a", "no", None, "yes")]
    mcc_drop = [PairedItem("b", "yes", "yes", "yes"), PairedItem("b", "no", "no", None)]
    result = evaluate_gate(f1_drop + mcc_drop, resamples=5)
    assert result.worst_language == "a"
    assert result.worst_language_delta_f1 == pytest.approx(-1 / 3)


def test_single_item_language_is_resampled():
    result = evaluate_gate([PairedItem("a", "yes", "yes", "yes")], resamples=5)
    assert result.passed


def test_default_seed_is_reproducible_for_differing_predictions():
    base = items(n=40)
    changed = [
        PairedItem(i.language, i.expected, i.reference, None if k % 3 == 0 else i.candidate)
        for k, i in enumerate(base)
    ]
    assert evaluate_gate(changed, resamples=50) == evaluate_gate(changed, resamples=50)


def test_empty_refused_message():
    with pytest.raises(ValueError, match=r"^no paired items$"):
        evaluate_gate([])
