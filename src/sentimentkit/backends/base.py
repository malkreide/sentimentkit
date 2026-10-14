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


class Backend(ABC):
    name: str = "abstract"
    stage: int = 0  # roadmap stage in which the backend becomes usable

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
