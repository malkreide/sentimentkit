"""Human gate: decision model → confidence threshold → escalation.

The thresholds live in policy.yaml, not in the backend, so they can be tuned
on the gold set and audited. A judgement that fails the gate is not dropped;
it is marked `needs_review` and routed to the review queue (io.review_queue).
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import Judgement, Polarity
from .policy import Policy


def needs_review(
    polarity: Polarity,
    confidence: float,
    distribution: Mapping[str, float] | None,
    policy: Policy,
) -> bool:
    """The gate's rule, shared by the pipeline and the eval's review figures."""
    review = polarity is Polarity.UNCLEAR or confidence < policy.review_threshold
    if distribution:
        ranked = sorted(distribution.values(), reverse=True)
        if len(ranked) >= 2 and (ranked[0] - ranked[1]) < policy.review_margin:
            # a near-tie is a review case even when confidence looks fine
            review = review or polarity is not Polarity.NOT_MENTIONED
    return review


def apply_gate(j: Judgement, policy: Policy) -> Judgement:
    j.needs_review = needs_review(j.polarity, j.confidence, j.distribution, policy)
    return j
