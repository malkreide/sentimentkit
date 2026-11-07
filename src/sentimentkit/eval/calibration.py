"""Temperature scaling for backends whose distribution is a softmax.

One parameter T rescales a judgement's distribution: softmax(log p / T). It
never changes which option wins, only how sure the backend claims to be,
which is what the review gate in policy.yaml reads. Two choices keep it
honest on a small gold set:

- T is fitted on the log-loss of the top label being right, the quantity ECE
  and the gate look at. A gold label outside a backend's answer set (the
  encoder has no `unclear`) cannot pull T towards infinity that way.
- The reported ECE is never measured on the judgements T was fitted on:
  with a dev set (`fit_on`), T comes from the dev set and the ECE from the
  evaluated set; without one it is cross-fitted, every judgement rescaled
  with a T fitted on the other folds.

Judgements whose confidence is not their distribution's top (rule-based
answers such as an encoder's `not_mentioned` from clause scoping) keep their
confidence. Options that are zero in every rescaled judgement are outside the
backend's answer set and are left out; other zeros are rounding (four
decimals) and count as EPS.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .metrics import expected_calibration_error

EPS = 5e-5  # half the rounding unit of a judgement's distribution
T_MIN, T_MAX = 0.05, 100.0
FOLDS = 5


@dataclass(frozen=True)
class Sample:
    distribution: Mapping[str, float] | None  # None: not a softmax, confidence stays
    predicted: str
    confidence: float
    correct: bool


@dataclass(frozen=True)
class Calibration:
    temperature: float  # the value to configure: fitted on the dev set, else on all samples
    ece_before: float
    ece_after: float  # with T from the dev set, else cross-fitted
    folds: int  # 0 when T came from a dev set
    n_scaled: int
    method: str = "cross-fitted"  # or "dev"
    n_fit: int = 0  # judgements T was fitted on


def sample(
    distribution: Mapping[str, float] | None, predicted: str, confidence: float, correct: bool
) -> Sample:
    """A judgement as calibration sees it; a distribution only where the
    confidence is that distribution's top probability."""
    softmax = bool(distribution) and distribution.get(predicted) == confidence == max(
        distribution.values()
    )
    return Sample(dict(distribution) if softmax else None, predicted, confidence, correct)


def scaled_confidence(s: Sample, t: float, keys: Sequence[str]) -> float:
    if s.distribution is None:
        return s.confidence
    logs = {k: math.log(max(s.distribution.get(k, 0.0), EPS)) / t for k in keys}
    top = max(logs.values())
    z = sum(math.exp(v - top) for v in logs.values())
    return math.exp(logs[s.predicted] - top) / z


def _log_loss(samples: Sequence[Sample], t: float, keys: Sequence[str]) -> float:
    loss = 0.0
    for s in samples:
        c = min(max(scaled_confidence(s, t, keys), 1e-12), 1 - 1e-12)
        loss -= math.log(c) if s.correct else math.log(1 - c)
    return loss


def fit_temperature(samples: Sequence[Sample], keys: Sequence[str]) -> float:
    """Log-spaced grid over [T_MIN, T_MAX], then a finer grid around the best."""
    scalable = [s for s in samples if s.distribution is not None]
    if not scalable:
        return 1.0
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    grid = [lo + (hi - lo) * i / 200 for i in range(201)]
    best = min(grid, key=lambda g: _log_loss(scalable, math.exp(g), keys))
    step = (hi - lo) / 200
    fine = [min(hi, max(lo, best - step + 2 * step * i / 40)) for i in range(41)]
    return math.exp(min(fine, key=lambda g: _log_loss(scalable, math.exp(g), keys)))


def answer_set(samples: Sequence[Sample]) -> list[str]:
    keys = {k for s in samples if s.distribution for k, p in s.distribution.items() if p > 0}
    return sorted(keys)


def calibrate(
    samples: Sequence[Sample],
    *,
    fit_on: Sequence[Sample] | None = None,
    folds: int = FOLDS,
    seed: int = 0,
) -> Calibration | None:
    """ECE before and after temperature scaling; None when no judgement is a
    softmax (e.g. the lexicon backend). With `fit_on` (the dev set), T is
    fitted there and applied to `samples`; otherwise T is fitted on all
    samples and the ECE after scaling is cross-fitted."""
    keys = answer_set([*samples, *(fit_on or [])])
    n_scaled = sum(s.distribution is not None for s in samples)
    if n_scaled < 2 or len(keys) < 2:
        return None
    correct = [s.correct for s in samples]
    before = round(expected_calibration_error([s.confidence for s in samples], correct), 4)
    n_fit = sum(s.distribution is not None for s in fit_on or [])
    if fit_on is not None and n_fit >= 2:
        t = fit_temperature(fit_on, keys)
        after = [scaled_confidence(s, t, keys) for s in samples]
        return Calibration(
            temperature=round(t, 3),
            ece_before=before,
            ece_after=round(expected_calibration_error(after, correct), 4),
            folds=0,
            n_scaled=n_scaled,
            method="dev",
            n_fit=n_fit,
        )
    folds = max(2, min(folds, n_scaled))
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    fold_of = {i: pos % folds for pos, i in enumerate(order)}
    after = [0.0] * len(samples)
    for f in range(folds):
        train = [s for i, s in enumerate(samples) if fold_of[i] != f]
        t = fit_temperature(train, keys)
        for i, s in enumerate(samples):
            if fold_of[i] == f:
                after[i] = scaled_confidence(s, t, keys)
    return Calibration(
        temperature=round(fit_temperature(samples, keys), 3),
        ece_before=before,
        ece_after=round(expected_calibration_error(after, correct), 4),
        folds=folds,
        n_scaled=n_scaled,
        n_fit=n_scaled,
    )
