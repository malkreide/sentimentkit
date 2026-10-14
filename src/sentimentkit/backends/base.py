"""Backend interface. Every backend — lexicon, encoder, local LLM scoring,
cloud LLM — implements the same two methods so the eval harness can compare
them on equal terms and the CLI can swap them with one flag.

The interface is *typed-question* shaped on purpose: a backend is asked, for
one text unit and one aspect, to distribute probability over a fixed set of
options. That is what makes fixed-answer scoring (Laya/Jev pattern), encoder
softmax and lexicon scores comparable, and it keeps an escape option
(`unclear`, `not_mentioned`) in every answer set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import Polarity

DEFAULT_OPTIONS: tuple[Polarity, ...] = (
    Polarity.POSITIVE,
    Polarity.NEUTRAL,
    Polarity.NEGATIVE,
    Polarity.UNCLEAR,
)

ASPECT_OPTIONS: tuple[Polarity, ...] = DEFAULT_OPTIONS + (Polarity.NOT_MENTIONED,)


@dataclass
class Decision:
    """A backend's answer for one unit × one aspect."""

    polarity: Polarity
    intensity: float  # signed, [-1, 1]
    confidence: float  # [0, 1]
    distribution: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    prompt_hash: str | None = None


# (text, aspect, aspect_terms, lang) — one typed question
Query = tuple[str, str, tuple[str, ...], str | None]


class BackendUnavailable(RuntimeError):
    """The backend exists but cannot run here, e.g. its optional extra is missing."""


class Backend(ABC):
    name: str = "abstract"
    stage: int = 0  # roadmap stage in which the backend becomes usable
    # True when `distribution` is a softmax and the polarity its argmax:
    # then `sentimentkit eval` can fit a temperature to it.
    softmax_distribution: bool = False

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @abstractmethod
    def decide(
        self,
        text: str,
        *,
        aspect: str = "overall",
        aspect_terms: tuple[str, ...] = (),
        lang: str | None = None,
    ) -> Decision: ...

    def decide_many(
        self,
        texts: list[str],
        *,
        aspect: str = "overall",
        aspect_terms: tuple[str, ...] = (),
        lang: str | None = None,
    ) -> list[Decision]:
        """Batch hook. Backends with real batching override this."""
        return [self.decide(t, aspect=aspect, aspect_terms=aspect_terms, lang=lang) for t in texts]

    def missing_extra(self) -> str | None:
        """The pip extra this backend still needs in this environment, or None.
        Must stay cheap — `sentimentkit backends` calls it for every backend."""
        return None


def not_mentioned(aspect: str) -> Decision:
    """The escape answer when no clause mentions the aspect."""
    dist = {o.value: 0.0 for o in ASPECT_OPTIONS}
    dist[Polarity.NOT_MENTIONED.value] = 1.0
    return Decision(Polarity.NOT_MENTIONED, 0.0, 0.9, dist, [f"aspect:{aspect}:absent"])


def decide_grouped(backend: Backend, queries: Sequence[Query]) -> list[Decision]:
    """Answer mixed queries in input order with one `decide_many` call per
    (aspect, aspect_terms, lang), so that batching backends actually batch."""
    groups: dict[tuple[str, tuple[str, ...], str | None], list[int]] = {}
    for i, (_, aspect, terms, lang) in enumerate(queries):
        groups.setdefault((aspect, tuple(terms), lang), []).append(i)
    answers: dict[int, Decision] = {}
    for (aspect, terms, lang), idxs in groups.items():
        texts = [queries[i][0] for i in idxs]
        decisions = backend.decide_many(texts, aspect=aspect, aspect_terms=terms, lang=lang)
        if len(decisions) != len(idxs):
            raise RuntimeError(f"backend {backend.name!r} answered {len(decisions)} of {len(idxs)}")
        answers.update(zip(idxs, decisions))
    return [answers[i] for i in range(len(queries))]


class NotAvailable(Backend):
    """Placeholder for backends scheduled in later stages. Keeps the registry
    and the CLI honest about what exists today."""

    def __init__(self, name: str, stage: int, note: str) -> None:
        self.name = name
        self.stage = stage
        self.note = note

    @property
    def model_version(self) -> str:
        return "n/a"

    def decide(self, text: str, **_: object) -> Decision:  # pragma: no cover - trivial
        raise NotImplementedError(
            f"backend {self.name!r} is scheduled for stage {self.stage}: {self.note}"
        )
