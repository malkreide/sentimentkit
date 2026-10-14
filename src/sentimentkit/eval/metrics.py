"""Classification metrics without numpy/scipy — the harness must run on a
Raspberry Pi and in a bare CI container."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    _check(gold, pred)
    labels = sorted(set(gold) | set(pred))
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    _check(a, b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 0.0 if pe == 1.0 else (po - pe) / (1 - pe)


def confusion(gold: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, int]]:
    _check(gold, pred)
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return {g: dict(row) for g, row in m.items()}


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """ECE: how far 'confidence 0.8' is from 'right 80 % of the time'."""
    _check(confidences, correct)
    n = len(confidences)
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confidences, correct):
        idx = min(bins - 1, int(c * bins))
        buckets[idx].append((c, ok))
    ece = 0.0
    for rows in buckets.values():
        conf = sum(c for c, _ in rows) / len(rows)
        acc = sum(1 for _, ok in rows if ok) / len(rows)
        ece += (len(rows) / n) * abs(conf - acc)
    return ece


def _check(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty input")
