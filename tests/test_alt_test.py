from __future__ import annotations

import random

import pytest

from sentimentkit.eval.alt_test import _t_cdf, alt_test


def test_t_cdf_matches_known_values():
    # t = 0 → 0.5; large df → normal; t=1.96, df=1000 → ≈0.975
    assert _t_cdf(0.0, 10) == pytest.approx(0.5)
    assert _t_cdf(1.96, 1000) == pytest.approx(0.975, abs=2e-3)
    assert _t_cdf(-2.228, 10) == pytest.approx(0.025, abs=2e-3)  # t crit, df=10


def test_model_as_good_as_humans_passes():
    random.seed(1)
    n = 60
    truth = [random.choice(["pos", "neg", "neu"]) for _ in range(n)]

    def noisy(p):
        return [t if random.random() < p else random.choice(["pos", "neg", "neu"]) for t in truth]

    human = {"a1": noisy(0.8), "a2": noisy(0.8), "a3": noisy(0.8)}
    model = noisy(0.95)
    res = alt_test(human, model)
    assert res.passed and res.omega >= 0.5


def test_bad_model_fails():
    random.seed(2)
    n = 60
    truth = [random.choice(["pos", "neg", "neu"]) for _ in range(n)]

    def noisy(p):
        return [t if random.random() < p else random.choice(["pos", "neg", "neu"]) for t in truth]

    human = {"a1": noisy(0.9), "a2": noisy(0.9), "a3": noisy(0.9)}
    model = noisy(0.3)
    res = alt_test(human, model)
    assert not res.passed and res.omega == 0.0


def test_needs_two_annotators():
    with pytest.raises(ValueError):
        alt_test({"a1": ["pos"]}, ["pos"])


def test_skipped_instances_are_ignored():
    human = {"a1": ["pos", None, "neg", "pos"], "a2": ["pos", "neg", "neg", "pos"]}
    res = alt_test(human, ["pos", "neg", "neg", "pos"])
    assert res.per_annotator["a1"]["n"] == 3
