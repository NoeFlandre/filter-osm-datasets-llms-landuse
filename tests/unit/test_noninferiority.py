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
