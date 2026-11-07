"""Eval harness: gold-set format, metrics, alt-test and the backend gate.

Gold set format (JSONL), one line per instance:

    {"text_id": "g001", "text": "…", "aspect": "overall",
     "gold": "negative",                      # consensus label
     "annotators": {"a1": "negative", "a2": "negative"},   # optional
     "tags": ["negation", "satzklammer"]}     # optional, for slicing

Every backend is evaluated the same way; `evaluate()` returns the figures the
README table and the ROADMAP gates refer to.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import Backend, Decision, decide_grouped
from ..gate import needs_review
from ..models import Polarity
from ..policy import Policy
from .alt_test import AltTestResult, alt_test
from .calibration import Calibration, Sample, calibrate, sample
from .metrics import (
    POLAR,
    accuracy,
    cohen_kappa,
    confusion,
    expected_calibration_error,
    macro_f1,
    paired_bootstrap,
)

SPLITS = ("dev", "test")


@dataclass
class GoldInstance:
    text_id: str
    text: str
    gold: str
    aspect: str = "overall"
    aspect_terms: tuple[str, ...] = ()
    lang: str | None = None
    annotators: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    split: str | None = None  # dev (calibration, tuning) or test (the gate)


def load_gold(path: str | Path) -> list[GoldInstance]:
    out: list[GoldInstance] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        Polarity(row["gold"])  # validates the label
        out.append(
            GoldInstance(
                text_id=row["text_id"],
                text=row["text"],
                gold=row["gold"],
                aspect=row.get("aspect", "overall"),
                aspect_terms=tuple(row.get("aspect_terms", [])),
                lang=row.get("lang"),
                annotators=dict(row.get("annotators", {})),
                tags=list(row.get("tags", [])),
                split=row.get("split"),
            )
        )
    return out


def select_split(gold: list[GoldInstance], split: str) -> list[GoldInstance]:
    """The instances of one split; an error when the gold set has none."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; known: {', '.join(SPLITS)}")
    chosen = [g for g in gold if g.split == split]
    if not chosen:
        raise ValueError(
            f"the gold set has no {split!r} instances; build it with "
            "scripts/candidates_to_gold.py, which assigns dev and test"
        )
    return chosen


@dataclass
class EvalResult:
    backend: str
    model_version: str
    n: int
    accuracy: float
    macro_f1: float
    macro_f1_polar: float  # mean F1 over positive, neutral, negative only
    kappa_vs_gold: float
    ece: float
    confusion: dict[str, dict[str, int]]
    by_tag: dict[str, dict[str, float]]
    alt_test: dict[str, Any] | None = None
    # temperature scaling, for backends whose distribution is a softmax
    calibration: dict[str, Any] | None = None
    # what the review gate in policy.yaml would route, and how many errors it catches
    review: dict[str, Any] | None = None
    gold_labels: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)  # in gold order, for paired tests
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decide(backend: Backend, gold: list[GoldInstance]) -> list[Decision]:
    return decide_grouped(backend, [(g.text, g.aspect, g.aspect_terms, g.lang) for g in gold])


def _samples(decisions: list[Decision], gold: list[GoldInstance]) -> list[Sample]:
    return [
        sample(d.distribution, d.polarity.value, d.confidence, d.polarity.value == g.gold)
        for d, g in zip(decisions, gold)
    ]


def review_figures(
    decisions: list[Decision], correct: list[bool], policy: Policy
) -> dict[str, Any]:
    """Share of judgements the gate routes to review, and of errors it catches."""
    flagged = [
        needs_review(d.polarity, d.confidence, d.distribution or None, policy) for d in decisions
    ]
    errors = sum(not ok for ok in correct)
    caught = sum(f and not ok for f, ok in zip(flagged, correct))
    return {
        "review_threshold": policy.review_threshold,
        "review_margin": policy.review_margin,
        "n_review": sum(flagged),
        "review_rate": round(sum(flagged) / len(decisions), 4),
        "errors": errors,
        "errors_caught": caught,
        "capture": round(caught / errors, 4) if errors else None,
    }


def evaluate(
    backend: Backend,
    gold: list[GoldInstance],
    *,
    with_alt_test: bool = True,
    calibration_gold: list[GoldInstance] | None = None,
    policy: Policy | None = None,
) -> EvalResult:
    """Score `backend` on `gold`. `calibration_gold` (the dev split) is where a
    softmax backend's temperature is fitted; without it the fit is
    cross-fitted on `gold`. `policy` adds what its review gate would catch."""
    preds: list[str] = []
    confs: list[float] = []
    failures: list[dict[str, Any]] = []
    decisions = _decide(backend, gold)
    for g, d in zip(gold, decisions):
        preds.append(d.polarity.value)
        confs.append(d.confidence)
        if d.polarity.value != g.gold:
            failures.append(
                {
                    "text_id": g.text_id,
                    "gold": g.gold,
                    "pred": d.polarity.value,
                    "confidence": d.confidence,
                    "evidence": d.evidence,
                    "tags": g.tags,
                }
            )
    golds = [g.gold for g in gold]
    correct = [p == gl for p, gl in zip(preds, golds)]

    cal: Calibration | None = None
    if backend.softmax_distribution:
        fit_on = None
        if calibration_gold:
            fit_on = _samples(_decide(backend, calibration_gold), calibration_gold)
        cal = calibrate(_samples(decisions, gold), fit_on=fit_on)

    by_tag: dict[str, dict[str, float]] = {}
    idx_by_tag: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gold):
        for t in g.tags:
            idx_by_tag[t].append(i)
    for t, idxs in sorted(idx_by_tag.items()):
        by_tag[t] = {
            "n": len(idxs),
            "accuracy": round(sum(correct[i] for i in idxs) / len(idxs), 4),
        }

    alt: AltTestResult | None = None
    annotator_ids = sorted({a for g in gold for a in g.annotators})
    if with_alt_test and len(annotator_ids) >= 2:
        human = {a: [g.annotators.get(a) for g in gold] for a in annotator_ids}
        alt = alt_test(human, preds)

    return EvalResult(
        backend=backend.name,
        model_version=backend.model_version,
        n=len(gold),
        accuracy=round(accuracy(golds, preds), 4),
        macro_f1=round(macro_f1(golds, preds), 4),
        macro_f1_polar=round(macro_f1(golds, preds, POLAR), 4),
        kappa_vs_gold=round(cohen_kappa(golds, preds), 4),
        ece=round(expected_calibration_error(confs, correct), 4),
        confusion=confusion(golds, preds),
        by_tag=by_tag,
        alt_test=asdict(alt) if alt else None,
        calibration=asdict(cal) if cal else None,
        review=review_figures(decisions, correct, policy) if policy else None,
        gold_labels=golds,
        predictions=preds,
        failures=failures,
    )


MIN_GAIN = 0.05  # macro-F1 points a backend must add over the baseline (ROADMAP stage 2)
GATE_METRICS = {"macro_f1": None, "macro_f1_polar": POLAR}  # metric -> labels averaged


def beats_baseline(
    candidate: EvalResult,
    baseline: EvalResult,
    *,
    min_gain: float = MIN_GAIN,
    metric: str = "macro_f1",
) -> bool:
    """Roadmap gate for stage 2: a backend becomes default only when `metric`
    exceeds the lexicon baseline by `min_gain` on the same gold set."""
    if metric not in GATE_METRICS:
        raise ValueError(f"unknown gate metric {metric!r}; known: {', '.join(GATE_METRICS)}")
    return (
        candidate.n == baseline.n
        and getattr(candidate, metric) >= getattr(baseline, metric) + min_gain
    )


def gain_interval(
    candidate: EvalResult,
    baseline: EvalResult,
    *,
    metric: str = "macro_f1",
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(gain, low, high): the candidate's gain in `metric` over the baseline
    and its paired-bootstrap 95 % interval, on the same gold instances."""
    if candidate.gold_labels != baseline.gold_labels:
        raise ValueError("both results must come from the same gold instances")
    labels = GATE_METRICS[metric]

    def score(g, p):
        return macro_f1(g, p, labels)

    return paired_bootstrap(
        candidate.gold_labels,
        baseline.predictions,
        candidate.predictions,
        score,
        resamples=resamples,
        seed=seed,
    )


__all__ = [
    "GoldInstance",
    "EvalResult",
    "load_gold",
    "evaluate",
    "beats_baseline",
    "gain_interval",
    "select_split",
    "review_figures",
    "GATE_METRICS",
    "SPLITS",
    "MIN_GAIN",
    "alt_test",
    "calibrate",
    "accuracy",
    "macro_f1",
    "cohen_kappa",
    "confusion",
    "expected_calibration_error",
]
