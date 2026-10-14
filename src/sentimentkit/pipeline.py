"""The run: policy → backend → judgements → gate.

Kept free of I/O so it can be called from the CLI, from tests, from a
Claude skill and later from an MCP tool without change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from . import __version__
from .backends import get_backend
from .backends.base import Backend
from .gate import apply_gate
from .models import Judgement, RunMeta, TextItem
from .policy import Policy

AspectSpec = Mapping[str, tuple[str, ...]]


@dataclass
class RunResult:
    meta: RunMeta
    judgements: list[Judgement]


def run(
    items: Iterable[TextItem],
    *,
    policy: Policy,
    purpose: str,
    backend: str | Backend = "lexicon",
    aspects: AspectSpec | None = None,
    anonymized: bool = False,
    context: str | None = None,
    dimensions: Iterable[str] = (),
) -> RunResult:
    items = list(items)
    backend_obj = backend if isinstance(backend, Backend) else get_backend(backend)

    policy.check_run(
        purpose=purpose,
        backend=backend_obj.name,
        anonymized=anonymized,
        dimensions=dimensions,
        context=context,
    )
    policy.check_items(items)

    meta = RunMeta(
        sentimentkit_version=__version__,
        purpose=purpose,
        backend=backend_obj.name,
        model_version=backend_obj.model_version,
        policy_hash=policy.hash,
        anonymized=anonymized,
        n_items=len(items),
        params={"aspects": sorted(aspects) if aspects else []},
    )

    judgements: list[Judgement] = []
    for item in items:
        targets: list[tuple[str, tuple[str, ...]]] = [("overall", ())]
        if aspects:
            targets += [(name, tuple(terms)) for name, terms in aspects.items()]
        for aspect, terms in targets:
            d = backend_obj.decide(item.text, aspect=aspect, aspect_terms=terms, lang=item.lang)
            j = Judgement(
                run_id=meta.run_id,
                text_id=item.text_id,
                span=(0, len(item.text)),
                sentence=item.text if len(item.text) <= 280 else item.text[:277] + "…",
                aspect=aspect,
                polarity=d.polarity,
                intensity=d.intensity,
                confidence=d.confidence,
                distribution=d.distribution or None,
                backend=backend_obj.name,
                model_version=backend_obj.model_version,
                prompt_hash=d.prompt_hash,
                evidence=d.evidence,
            )
            judgements.append(apply_gate(j, policy))

    meta.n_judgements = len(judgements)
    meta.n_review = sum(1 for j in judgements if j.needs_review)
    return RunResult(meta=meta, judgements=judgements)
