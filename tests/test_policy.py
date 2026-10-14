from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentimentkit.models import TextItem
from sentimentkit.policy import Policy, PolicyViolation


def test_policy_hash_is_stable(policy):
    assert len(policy.hash) == 16
    assert policy.hash == Policy.model_validate(policy.model_dump()).hash


def test_purpose_required(policy):
    with pytest.raises(PolicyViolation):
        policy.check_run(purpose=None, backend="lexicon", anonymized=False)


def test_unknown_purpose_rejected(policy):
    with pytest.raises(PolicyViolation, match="not allowed"):
        policy.check_run(purpose="marketing_profiling", backend="lexicon", anonymized=False)


def test_forbidden_dimension_rejected(policy):
    with pytest.raises(PolicyViolation, match="forbidden dimension"):
        policy.check_run(
            purpose="media_resonance",
            backend="lexicon",
            anonymized=False,
            dimensions=["political_leaning"],
        )


def test_forbidden_context_rejected(policy):
    with pytest.raises(PolicyViolation, match="out of scope"):
        policy.check_run(
            purpose="inbox_triage",
            backend="lexicon",
            anonymized=False,
            context="student_monitoring",
        )


def test_cloud_backend_requires_anonymized(policy):
    with pytest.raises(PolicyViolation, match="anonymised"):
        policy.check_run(purpose="media_resonance", backend="llm-cloud", anonymized=False)
    policy.check_run(purpose="media_resonance", backend="llm-cloud", anonymized=True)


def test_person_identifier_in_meta_rejected(policy):
    items = [TextItem(text="alles gut", meta={"author_id": "u42"})]
    with pytest.raises(PolicyViolation, match="person-level identifier"):
        policy.check_items(items)
    policy.check_items([TextItem(text="alles gut", meta={"schulkreis": "A"})])


def test_aggregation_floor(policy):
    policy.check_group_size("x", policy.aggregation_floor)
    with pytest.raises(PolicyViolation, match="aggregation_floor"):
        policy.check_group_size("x", policy.aggregation_floor - 1)


def test_policy_file_forbids_extra_keys(tmp_path):
    from sentimentkit.policy import load_policy

    p = tmp_path / "policy.yaml"
    p.write_text("version: 1\nsurprise: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_policy(p)
