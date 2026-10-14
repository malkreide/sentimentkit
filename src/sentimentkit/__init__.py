"""sentimentkit — Sentiment toolkit with swappable backends, evidence atoms,
an eval harness and policy-as-code.

Stage 0 ships the lexicon baseline only. Encoder, local LLM scoring and
cloud LLM backends are registered as stubs so the interface is fixed before
any model is chosen (see docs/ROADMAP.md).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .models import Judgement, Polarity, RunMeta, TextItem  # noqa: E402
from .policy import Policy, PolicyViolation, load_policy  # noqa: E402

__all__ = [
    "__version__",
    "Judgement",
    "Polarity",
    "RunMeta",
    "TextItem",
    "Policy",
    "PolicyViolation",
    "load_policy",
]
