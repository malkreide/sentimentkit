"""Backend registry.

`lexicon` (stage 0) and `encoder` (stage 2) are implemented; the encoder needs
the optional extra `encoder` at run time, not to be listed. The two LLM
backends are registered as ``NotAvailable`` so that `sentimentkit backends`
lists the full plan and a typo in `--backend` fails loudly instead of
silently falling back.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Backend, BackendUnavailable, Decision, NotAvailable, decide_grouped
from .encoder import EncoderBackend
from .lexicon import LexiconBackend

_REGISTRY: dict[str, Callable[[], Backend]] = {
    "lexicon": LexiconBackend,
    "encoder": EncoderBackend,
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


__all__ = [
    "Backend",
    "BackendUnavailable",
    "Decision",
    "EncoderBackend",
    "LexiconBackend",
    "NotAvailable",
    "available",
    "decide_grouped",
    "get_backend",
]
