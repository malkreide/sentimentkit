"""Backend registry.

`lexicon` (stage 0), `encoder` and `llm-score` (stage 2) are implemented; the
latter two need their optional extra (`encoder`, `llm`) at run time, not to be
listed. `llm-cloud` is registered as ``NotAvailable`` so that `sentimentkit
backends` lists the full plan and a typo in `--backend` fails loudly instead
of silently falling back.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Backend, BackendUnavailable, Decision, NotAvailable, decide_grouped
from .encoder import EncoderBackend
from .lexicon import LexiconBackend
from .llm_score import LLMScoreBackend

_REGISTRY: dict[str, Callable[[], Backend]] = {
    "lexicon": LexiconBackend,
    "encoder": EncoderBackend,
    "llm-score": LLMScoreBackend,
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


__all__ = [
    "Backend",
    "BackendUnavailable",
    "Decision",
    "EncoderBackend",
    "LLMScoreBackend",
    "LexiconBackend",
    "NotAvailable",
    "available",
    "decide_grouped",
    "get_backend",
]
