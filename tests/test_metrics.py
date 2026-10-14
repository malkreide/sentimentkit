from __future__ import annotations

import pytest

from sentimentkit.eval.metrics import (
    accuracy,
    cohen_kappa,
    confusion,
    expected_calibration_error,
    macro_f1,
)


def test_accuracy_and_f1_perfect():
    g = ["a", "b", "a", "c"]
    assert accuracy(g, g) == 1.0
    assert macro_f1(g, g) == 1.0
    assert cohen_kappa(g, g) == 1.0


def test_macro_f1_penalises_missing_class():
    gold = ["pos", "neg", "pos", "neg"]
    pred = ["pos", "pos", "pos", "pos"]
    assert accuracy(gold, pred) == 0.5
    assert macro_f1(gold, pred) == pytest.approx((2 / 3) / 2)


def test_kappa_zero_for_chance():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "a", "b"]
    assert cohen_kappa(gold, pred) == pytest.approx(0.0)


def test_confusion_shape():
    m = confusion(["a", "a", "b"], ["a", "b", "b"])
    assert m == {"a": {"a": 1, "b": 1}, "b": {"b": 1}}


def test_ece_zero_when_calibrated():
    conf = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    correct = [True] * 9 + [False]
    # bucket mean confidence 0.95 vs accuracy 0.9 → 0.05
    assert expected_calibration_error(conf, correct) == pytest.approx(0.05)


def test_length_mismatch():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])
