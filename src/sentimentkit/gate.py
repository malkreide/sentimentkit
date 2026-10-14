"""Human gate: decision model → confidence threshold → escalation.

The thresholds live in policy.yaml, not in the backend, so they can be tuned
on the gold set and audited. A judgement that fails the gate is not dropped;
it is marked `needs_review` and routed to the review queue (io.review_queue).
"""

from __future__ import annotations

from .models import Judgement, Polarity
from .policy import Policy


def apply_gate(j: Judgement, policy: Policy) -> Judgement:
    review = False
    if j.polarity is Polarity.UNCLEAR:
        review = True
    if j.confidence < policy.review_threshold:
        review = True
    if j.distribution:
        ranked = sorted(j.distribution.values(), reverse=True)
        if len(ranked) >= 2 and (ranked[0] - ranked[1]) < policy.review_margin:
            # a near-tie is a review case even when confidence looks fine
            review = review or j.polarity is not Polarity.NOT_MENTIONED
    j.needs_review = review
    return j
