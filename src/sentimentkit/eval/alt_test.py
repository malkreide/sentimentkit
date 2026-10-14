"""Alternative Annotator Test (alt-test) — pure-Python implementation.

Reference: Calderon, Reichart & Dror (2025), "The Alternative Annotator Test
for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators
with LLMs", arXiv:2501.10970.

Idea in one sentence: an LLM (or any backend) may replace human annotators if,
for at least half of the annotators, it agrees with the *other* annotators at
least as well as that annotator does — allowing a cost margin epsilon that
favours the human. This file implements that criterion with a one-sided
paired t-test per annotator and reports the winning rate omega.

Requires >= 2 human annotators per instance (the comparison is always
against the other annotators). The numbers here follow the paper's
defaults (epsilon = 0.1, alpha = 0.05); re-check against the full text before
citing a result — the summary in the knowledge base is itself unverified.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class AltTestResult:
    omega: float  # winning rate
    passed: bool
    epsilon: float
    alpha: float
    n_instances: int
    per_annotator: dict[str, dict[str, float | bool]] = field(default_factory=dict)


def alt_test(
    human: Mapping[str, Sequence[str | None]],
    model: Sequence[str],
    *,
    epsilon: float = 0.1,
    alpha: float = 0.05,
) -> AltTestResult:
    """`human` maps annotator id -> labels (None where that annotator skipped
    the instance); `model` is the backend's label per instance."""
    annotators = list(human)
    if len(annotators) < 2:
        raise ValueError("alt-test needs at least two human annotators")
    n = len(model)
    for a, labs in human.items():
        if len(labs) != n:
            raise ValueError(f"annotator {a!r}: {len(labs)} labels, expected {n}")

    wins = 0
    per: dict[str, dict[str, float | bool]] = {}
    for j in annotators:
        others = [a for a in annotators if a != j]
        diffs: list[float] = []
        for i in range(n):
            ref = [human[a][i] for a in others if human[a][i] is not None]
            if not ref or human[j][i] is None:
                continue
            s_model = sum(1.0 for r in ref if r == model[i]) / len(ref)
            s_human = sum(1.0 for r in ref if r == human[j][i]) / len(ref)
            diffs.append(s_model - s_human)
        if len(diffs) < 2:
            per[j] = {"n": len(diffs), "mean_diff": float("nan"), "p": 1.0, "win": False}
            continue
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        sd = math.sqrt(var)
        # H0: model is worse than annotator j by more than epsilon
        #     mean(d) <= -epsilon   vs   H1: mean(d) > -epsilon
        if sd == 0.0:
            p = 0.0 if mean > -epsilon else 1.0
        else:
            t = (mean + epsilon) / (sd / math.sqrt(len(diffs)))
            p = 1.0 - _t_cdf(t, len(diffs) - 1)
        win = p <= alpha
        wins += int(win)
        per[j] = {"n": len(diffs), "mean_diff": round(mean, 4), "p": round(p, 4), "win": win}

    omega = wins / len(annotators)
    return AltTestResult(
        omega=omega,
        passed=omega >= 0.5,
        epsilon=epsilon,
        alpha=alpha,
        n_instances=n,
        per_annotator=per,
    )


# --- Student t CDF via the regularised incomplete beta function ------------


def _t_cdf(t: float, df: int) -> float:
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
