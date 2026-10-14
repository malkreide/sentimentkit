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

from ..backends.base import Backend
from ..models import Polarity
from .alt_test import AltTestResult, alt_test
from .metrics import accuracy, cohen_kappa, confusion, expected_calibration_error, macro_f1


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
            )
        )
    return out


@dataclass
class EvalResult:
    backend: str
    model_version: str
    n: int
    accuracy: float
    macro_f1: float
    kappa_vs_gold: float
    ece: float
    confusion: dict[str, dict[str, int]]
    by_tag: dict[str, dict[str, float]]
    alt_test: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    backend: Backend, gold: list[GoldInstance], *, with_alt_test: bool = True
) -> EvalResult:
    preds: list[str] = []
    confs: list[float] = []
    failures: list[dict[str, Any]] = []
    for g in gold:
        d = backend.decide(g.text, aspect=g.aspect, aspect_terms=g.aspect_terms, lang=g.lang)
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
        kappa_vs_gold=round(cohen_kappa(golds, preds), 4),
        ece=round(expected_calibration_error(confs, correct), 4),
        confusion=confusion(golds, preds),
        by_tag=by_tag,
        alt_test=asdict(alt) if alt else None,
        failures=failures,
    )


def beats_baseline(candidate: EvalResult, baseline: EvalResult, *, min_gain: float = 0.05) -> bool:
    """Roadmap gate for stage 2: a backend becomes default only when its
    macro-F1 exceeds the lexicon baseline by `min_gain` on the same gold set."""
    return candidate.n == baseline.n and candidate.macro_f1 >= baseline.macro_f1 + min_gain


__all__ = [
    "GoldInstance",
    "EvalResult",
    "load_gold",
    "evaluate",
    "beats_baseline",
    "alt_test",
    "accuracy",
    "macro_f1",
    "cohen_kappa",
    "confusion",
    "expected_calibration_error",
]
