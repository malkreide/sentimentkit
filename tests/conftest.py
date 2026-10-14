from __future__ import annotations

from pathlib import Path

import pytest

from sentimentkit.backends.lexicon import LexiconBackend
from sentimentkit.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


@pytest.fixture(scope="session")
def lexicon() -> LexiconBackend:
    return LexiconBackend()


@pytest.fixture(scope="session")
def policy():
    return load_policy(ROOT / "policy.yaml")
