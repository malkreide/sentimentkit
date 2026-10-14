from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sentimentkit.models import Judgement, Polarity

ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return json.loads((ROOT / "schema" / name).read_text(encoding="utf-8"))


def test_judgement_round_trips_through_schema():
    j = Judgement(
        run_id="run123456",
        text_id="t1",
        polarity=Polarity.NEGATIVE,
        confidence=0.8,
        backend="lexicon",
        model_version="seed:1:x",
    )
    row = json.loads(j.model_dump_json(exclude_none=True))
    jsonschema.validate(row, _schema("judgement.schema.json"))


def test_schema_rejects_unknown_backend():
    j = Judgement(
        run_id="run123456",
        text_id="t1",
        polarity=Polarity.NEUTRAL,
        confidence=0.5,
        backend="magic",
        model_version="x",
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            json.loads(j.model_dump_json(exclude_none=True)), _schema("judgement.schema.json")
        )


def test_gold_fixture_matches_schema():
    schema = _schema("gold.schema.json")
    for line in (ROOT / "fixtures" / "de_synthetic_gold.jsonl").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            jsonschema.validate(json.loads(line), schema)
