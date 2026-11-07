"""Tools for the stage-2 gate: polar macro-F1, paired bootstrap, splits,
dev-fitted calibration and the review figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sentimentkit import backends
from sentimentkit.backends.base import Backend, Decision
from sentimentkit.cli import main
from sentimentkit.eval import (
    beats_baseline,
    evaluate,
    gain_interval,
    load_gold,
    review_figures,
    select_split,
)
from sentimentkit.eval.metrics import POLAR, macro_f1, paired_bootstrap
from sentimentkit.models import Polarity

FIX = Path(__file__).resolve().parents[1] / "fixtures"
GOLD = FIX / "de_synthetic_gold.jsonl"


class KeywordSoftmax(Backend):
    """A softmax backend in miniature: sure of itself, often wrong."""

    name = "encoder"
    softmax_distribution = True

    @property
    def model_version(self) -> str:
        return "keyword-softmax"

    def decide(self, text, *, aspect="overall", aspect_terms=(), lang=None) -> Decision:
        low = text.lower()
        top = "negative" if any(w in low for w in ("nicht", "chaos", "kein")) else "positive"
        dist = {"positive": 0.02, "neutral": 0.02, "negative": 0.02, "unclear": 0.0}
        dist[top] = 0.96
        return Decision(Polarity(top), 0.0, 0.96, dist)


def _split_copy(tmp_path: Path) -> Path:
    """The fixture with every third line in dev, the rest in test."""
    out = tmp_path / "gold_split.jsonl"
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for i, row in enumerate(rows):
        row["split"] = "dev" if i % 3 == 0 else "test"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return out


# ------------------------------------------------------------------ metrics
def test_polar_macro_f1_leaves_out_the_escape_classes():
    gold = ["positive", "negative", "neutral", "unclear", "not_mentioned"]
    pred = ["positive", "negative", "neutral", "positive", "not_mentioned"]
    assert macro_f1(gold, pred) < macro_f1(gold, pred, POLAR)
    # a polar gold answered `unclear` is still a miss for that class
    assert macro_f1(["positive"], ["unclear"], POLAR) == 0.0


def test_paired_bootstrap_is_about_the_difference():
    gold = ["positive", "negative", "neutral"] * 20
    same = paired_bootstrap(gold, gold, gold, macro_f1, resamples=200)
    assert same == (0.0, 0.0, 0.0)
    worse = ["neutral"] * len(gold)
    gain, low, high = paired_bootstrap(gold, worse, gold, macro_f1, resamples=500)
    assert gain > 0 and 0 < low <= gain <= high
    assert paired_bootstrap(gold, worse, gold, macro_f1, resamples=50, seed=3) == paired_bootstrap(
        gold, worse, gold, macro_f1, resamples=50, seed=3
    )


# ------------------------------------------------------------------- splits
def test_select_split_filters_and_explains(tmp_path):
    gold = load_gold(_split_copy(tmp_path))
    test = select_split(gold, "test")
    assert test and all(g.split == "test" for g in test)
    with pytest.raises(ValueError, match="candidates_to_gold"):
        select_split(load_gold(GOLD), "test")
    with pytest.raises(ValueError, match="unknown split"):
        select_split(gold, "train")


def test_temperature_from_dev_is_measured_on_test(tmp_path):
    gold = load_gold(_split_copy(tmp_path))
    dev, test = select_split(gold, "dev"), select_split(gold, "test")
    res = evaluate(KeywordSoftmax(), test, calibration_gold=dev)
    cal = res.calibration
    assert cal["method"] == "dev" and cal["folds"] == 0
    assert cal["n_fit"] == len(dev) and cal["n_scaled"] == len(test)
    assert evaluate(KeywordSoftmax(), test).calibration["method"] == "cross-fitted"


# ------------------------------------------------------------------- review
def test_review_figures_count_caught_errors(policy):
    sure = Decision(Polarity.POSITIVE, 0.9, 0.95, {"positive": 0.95, "negative": 0.05})
    unsure = Decision(Polarity.NEGATIVE, -0.1, 0.4, {"positive": 0.35, "negative": 0.4})
    figures = review_figures([sure, unsure, sure], [False, False, True], policy)
    assert figures["n_review"] == 1 and figures["errors"] == 2
    assert figures["errors_caught"] == 1 and figures["capture"] == 0.5
    assert review_figures([sure], [True], policy)["capture"] is None


def test_evaluate_reports_review_figures_with_a_policy(lexicon, policy):
    res = evaluate(lexicon, load_gold(GOLD), policy=policy)
    r = res.review
    assert r["errors"] == len(res.failures)
    assert 0 <= r["errors_caught"] <= r["errors"] and 0 <= r["review_rate"] <= 1
    assert evaluate(lexicon, load_gold(GOLD)).review is None


# --------------------------------------------------------------------- gate
def test_gate_metric_and_gain_interval(lexicon):
    gold = load_gold(GOLD)
    base, cand = evaluate(lexicon, gold), evaluate(KeywordSoftmax(), gold)
    assert beats_baseline(base, cand) == (base.macro_f1 >= cand.macro_f1 + 0.05)
    assert beats_baseline(base, cand, metric="macro_f1_polar") == (
        base.macro_f1_polar >= cand.macro_f1_polar + 0.05
    )
    with pytest.raises(ValueError, match="unknown gate metric"):
        beats_baseline(base, cand, metric="accuracy")
    gain, low, high = gain_interval(cand, base, resamples=200)
    assert gain == pytest.approx(cand.macro_f1 - base.macro_f1, abs=1e-3)
    assert low <= gain <= high
    with pytest.raises(ValueError, match="same gold"):
        gain_interval(cand, evaluate(lexicon, gold[:10]))


# ---------------------------------------------------------------------- CLI
@pytest.fixture
def keyword_encoder(monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "encoder", KeywordSoftmax)


def test_cli_compare_prints_interval_and_review(keyword_encoder, tmp_path):
    gold = str(_split_copy(tmp_path))
    args = ["eval", gold, "--compare", "lexicon,encoder", "--split", "test"]
    r = CliRunner().invoke(main, [*args, "--gate-metric", "macro_f1_polar"])
    assert r.exit_code == 0, r.output
    assert "split test" in r.output and "n=20" in r.output
    assert "95% CI [" in r.output and "macro_f1_polar" in r.output
    assert "encoder: review gate:" in r.output
    assert "T fitted on 10 dev judgements" in r.output
    payload = json.loads(CliRunner().invoke(main, [*args, "--json"]).output)
    low, high = payload["gain"]["encoder"]["ci95"]
    assert low <= payload["gain"]["encoder"]["gain"] <= high
    assert payload["gate_metric"] == "macro_f1"


def test_cli_split_needs_a_split_gold_set():
    r = CliRunner().invoke(main, ["eval", str(GOLD), "--split", "test"])
    assert r.exit_code == 1 and "candidates_to_gold" in r.output
