from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


def _load_script(name: str) -> ModuleType:
    """Import a file from scripts/ (not a package) as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def build_candidates() -> ModuleType:
    return _load_script("build_gold_candidates")


@pytest.fixture(scope="session")
def to_gold() -> ModuleType:
    return _load_script("candidates_to_gold")
