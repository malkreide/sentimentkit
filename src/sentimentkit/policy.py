"""Policy as code.

The policy file is loaded once per run, hashed into the run metadata, and
enforced at two points: before the backend runs (purpose, person identifiers,
anonymisation) and before aggregation (k-anonymity floor). Violations raise
``PolicyViolation``; the CLI maps that to exit code 2.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .models import TextItem, stable_hash

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "policy.yaml"


class PolicyViolation(RuntimeError):
    """Raised when a run would breach the policy. Never caught silently."""


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    purpose_required: bool = True
    allowed_purposes: list[str] = Field(default_factory=list)
    forbidden_dimensions: list[str] = Field(default_factory=list)
    forbidden_contexts: list[str] = Field(default_factory=list)
    per_person_profiles: bool = False
    person_identifier_keys: list[str] = Field(default_factory=list)
    aggregation_floor: int = 20
    cloud_backends: list[str] = Field(default_factory=list)
    cloud_backends_require_anonymized: bool = True
    review_threshold: float = 0.6
    review_margin: float = 0.2

    # --- derived -----------------------------------------------------------
    @property
    def hash(self) -> str:
        return stable_hash(self.model_dump())

    # --- checks ------------------------------------------------------------
    def check_run(
        self,
        *,
        purpose: str | None,
        backend: str,
        anonymized: bool,
        dimensions: Iterable[str] = (),
        context: str | None = None,
    ) -> None:
        if self.purpose_required and not purpose:
            raise PolicyViolation("purpose is required (policy.purpose_required)")
        if purpose and self.allowed_purposes and purpose not in self.allowed_purposes:
            raise PolicyViolation(
                f"purpose {purpose!r} is not allowed; allowed: {', '.join(self.allowed_purposes)}"
            )
        bad = sorted(set(dimensions) & set(self.forbidden_dimensions))
        if bad:
            raise PolicyViolation(f"forbidden dimension(s) requested: {', '.join(bad)}")
        if context and context in self.forbidden_contexts:
            raise PolicyViolation(f"context {context!r} is out of scope by policy")
        if (
            backend in self.cloud_backends
            and self.cloud_backends_require_anonymized
            and not anonymized
        ):
            raise PolicyViolation(
                f"backend {backend!r} is a cloud backend; input must be declared anonymised"
            )

    def check_items(self, items: Iterable[TextItem]) -> None:
        if self.per_person_profiles:
            return
        keys = {k.lower() for k in self.person_identifier_keys}
        for item in items:
            hit = sorted(k for k in item.meta if k.lower() in keys)
            if hit:
                raise PolicyViolation(
                    f"item {item.text_id}: person-level identifier(s) {hit} present, "
                    "but policy.per_person_profiles is false"
                )

    def check_group_size(self, label: str, n: int) -> None:
        if n < self.aggregation_floor:
            raise PolicyViolation(
                f"group {label!r} has n={n} < aggregation_floor={self.aggregation_floor}"
            )


def load_policy(path: str | Path | None = None) -> Policy:
    p = Path(path) if path else DEFAULT_POLICY_PATH
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Policy.model_validate(data)
