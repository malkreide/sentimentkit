"""Backend registry.

Stage 0 ships `lexicon`. The other three names are registered as
``NotAvailable`` so that `sentimentkit backends` lists the full plan and a
typo in `--backend` fails loudly instead of silently falling back.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Backend, Decision, NotAvailable
from .lexicon import LexiconBackend

_REGISTRY: dict[str, Callable[[], Backend]] = {
    "lexicon": LexiconBackend,
    "encoder": lambda: NotAvailable(
        "encoder",
        2,
        "german-sentiment-bert / twitter-xlm-roberta-base-sentiment via transformers; "
        "CPU-capable, Raspberry Pi 5 target",
    ),
    "llm-score": lambda: NotAvailable(
        "llm-score",
        2,
        "fixed-answer scoring over typed questions via Ollama/SGLang /v1/score "
        "(Laya/Jev pattern); calibrated on the gold set",
    ),
    "llm-cloud": lambda: NotAvailable(
        "llm-cloud",
        2,
        "Claude API with typed questions and rationale; anonymised or public input only",
    ),
}


def available() -> dict[str, Backend]:
    return {name: factory() for name, factory in _REGISTRY.items()}


def get_backend(name: str) -> Backend:
    try:
        backend = _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(_REGISTRY)}") from exc
    if isinstance(backend, NotAvailable):
        raise NotImplementedError(
            f"backend {name!r} is scheduled for stage {backend.stage}: {backend.note}"
        )
    return backend


__all__ = ["Backend", "Decision", "LexiconBackend", "NotAvailable", "available", "get_backend"]
