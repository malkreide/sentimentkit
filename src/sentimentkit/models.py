"""Data model: the evidence atom and its surroundings.

One judgement is one evidence atom. Every number in a report must be traceable
to a list of atoms, and every atom must carry enough provenance to be re-run:
which backend, which model version, which prompt (hash), when, and whether a
human has verified it. This mirrors the evidence-atom pattern of journeykit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.1"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    NOT_MENTIONED = "not_mentioned"  # aspect asked but absent (escape option)
    UNCLEAR = "unclear"  # backend cannot decide — never silently neutral


class TextItem(BaseModel):
    """One input text. `meta` is free-form but must not carry person-level
    identifiers when the policy forbids per-person profiles (see policy.py)."""

    model_config = ConfigDict(extra="forbid")

    text_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    lang: str | None = None  # BCP-47-ish: de, fr, it, en, gsw (Swiss German)
    source: str | None = None  # e.g. "news-monitor-mcp", "survey-2026-q3"
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class Judgement(BaseModel):
    """The evidence atom."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    judgement_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    run_id: str
    text_id: str
    span: tuple[int, int] | None = None  # character offsets in the source text
    sentence: str | None = None  # the judged unit, for human review
    aspect: str = "overall"
    polarity: Polarity
    intensity: float = Field(ge=-1.0, le=1.0, default=0.0)  # signed strength
    confidence: float = Field(ge=0.0, le=1.0)  # backend's own confidence
    distribution: dict[str, float] | None = None  # per-option probabilities
    backend: str
    model_version: str
    prompt_hash: str | None = None
    evidence: list[str] = Field(default_factory=list)  # tokens / cues that fired
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    human_verified: bool = False
    human_label: Polarity | None = None
    needs_review: bool = False


class RunMeta(BaseModel):
    """Provenance of a whole run — written next to the judgements."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    schema_version: str = SCHEMA_VERSION
    sentimentkit_version: str
    purpose: str
    backend: str
    model_version: str
    policy_hash: str
    anonymized: bool = False
    n_items: int = 0
    n_judgements: int = 0
    n_review: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    params: dict[str, Any] = Field(default_factory=dict)


def stable_hash(obj: Any) -> str:
    """Deterministic short hash for prompts, policies, configs."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
