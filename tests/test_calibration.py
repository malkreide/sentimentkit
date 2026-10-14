from __future__ import annotations

import random
from pathlib import Path

import pytest

from sentimentkit.eval import evaluate, load_gold
from sentimentkit.eval.calibration import (
    T_MAX,
    T_MIN,
    Sample,
    answer_set,
    calibrate,
    sample,
    scaled_confidence,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _samples(n: int, confidence: float, accuracy: float, seed: int = 1) -> list[Sample]:
    rng = random.Random(seed)
    rest = (1 - confidence) / 2
    out = []
    for _ in range(n):
        dist = {"positive": confidence, "neutral": rest, "negative": rest}
        out.append(sample(dist, "positive", confidence, rng.random() < accuracy))
    return out


def test_overconfidence_is_softened():
    cal = calibrate(_samples(200, 0.98, 0.6))
    assert cal is not None and cal.temperature > 1
    assert cal.ece_after < cal.ece_before


def test_underconfidence_is_sharpened():
    cal = calibrate(_samples(200, 0.5, 0.95))
    assert cal is not None and cal.temperature < 1
    assert cal.ece_after < cal.ece_before


def test_temperature_stays_within_its_bounds():
    always_wrong = calibrate(_samples(50, 0.9, 0.0))
    always_right = calibrate(_samples(50, 0.6, 1.0))
    assert always_wrong is not None and always_wrong.temperature <= T_MAX
    assert always_right is not None and always_right.temperature >= T_MIN


def test_rule_based_confidence_is_not_rescaled():
    # lexicon-like: the confidence is not the distribution's top probability
    s = sample({"positive": 0.3, "neutral": 0.7}, "positive", 0.8, True)
    assert s.distribution is None
    assert scaled_confidence(s, 5.0, ["positive", "neutral"]) == 0.8
    assert calibrate([s, s, s]) is None


def test_options_outside_the_answer_set_are_left_out():
    samples = [
        sample({"positive": 0.9, "neutral": 0.1, "unclear": 0.0}, "positive", 0.9, True),
        sample({"positive": 0.0, "neutral": 1.0, "unclear": 0.0}, "neutral", 1.0, False),
    ]
    assert answer_set(samples) == ["neutral", "positive"]  # 'unclear' never > 0
    # a rounded zero inside the answer set still gives way under a high T
    assert scaled_confidence(samples[1], 10.0, answer_set(samples)) < 1.0


def test_evaluate_calibrates_only_softmax_backends(lexicon):
    gold = load_gold(FIX / "de_synthetic_gold.jsonl")
    assert evaluate(lexicon, gold).calibration is None


@pytest.mark.parametrize("t", [0.5, 1.0, 3.0])
def test_scaling_keeps_the_winner(t):
    s = sample({"positive": 0.6, "neutral": 0.3, "negative": 0.1}, "positive", 0.6, True)
    keys = ["negative", "neutral", "positive"]
    others = [
        scaled_confidence(sample(s.distribution, k, s.distribution[k], True), t, keys)
        for k in ("neutral", "negative")
    ]
    assert all(scaled_confidence(s, t, keys) > o for o in others)
