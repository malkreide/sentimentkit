from __future__ import annotations

import csv
import json
from pathlib import Path

import jsonschema
import pytest

from sentimentkit.eval import evaluate, load_gold

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema" / "gold.schema.json").read_text(encoding="utf-8"))
COLUMNS = ["text_id", "text", "source", "a1", "a2", "tags", "note"]

ROWS = [
    {"text_id": "t1", "text": "Die Tagesschule ist ein Erfolg.", "source": "m", "tags": ""},
    {"text_id": "t2", "text": "Die Betreuung ist nicht gut.", "source": "m", "tags": "negation?"},
    {"text_id": "t3", "text": "Toll, schon wieder eine Schulreform.", "source": "g", "tags": ""},
    {"text_id": "t4", "text": "Die Kreisschulbehörde tagt am Montag.", "source": "g", "tags": ""},
    {"text_id": "t5", "text": "Gemeinderätin X lobt die Schule.", "source": "g", "tags": ""},
    {"text_id": "t6", "text": "Die Volksschule wächst weiter.", "source": "m", "tags": ""},
]


def _write(path: Path, rows: list[dict], *, delimiter: str = ",", columns=COLUMNS) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter=delimiter, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def _annotated(tmp_path: Path) -> tuple[Path, Path]:
    """Two independent copies: a1 in a comma CSV, a2 in a semicolon CSV
    (Excel de-CH). t6 is not yet annotated by a2."""
    a1 = {"t1": "pos", "t2": "negative", "t3": "negative", "t4": "neutral", "t5": "positive"}
    a1["t6"] = "neutral"
    a2 = {"t1": "positiv", "t2": "Negative", "t3": "unclear", "t4": "neutral", "t5": "positive"}
    rows1, rows2 = [], []
    for r in ROWS:
        r1 = {**r, "a1": a1.get(r["text_id"], "")}
        if r["text_id"] == "t2":
            r1["tags"] = "negation"  # a1 confirms the hint
        r2 = {**r, "a2": a2.get(r["text_id"], "")}
        if r["text_id"] == "t3":
            r2["tags"] = "irony"
            r2["note"] = "Ironie, Absicht unklar"
        if r["text_id"] == "t5":
            r2["tags"] = "exclude"  # personal name
        rows1.append(r1)
        rows2.append(r2)
    return _write(tmp_path / "a1.csv", rows1), _write(tmp_path / "a2.csv", rows2, delimiter=";")


@pytest.mark.parametrize(
    ("labels", "gold"),
    [
        (["positive", "positive"], "positive"),
        (["positive", "negative"], "unclear"),
        (["negative", "negative", "neutral"], "negative"),
        (["negative", "neutral", "positive"], "unclear"),
        (["neutral", "neutral", "positive", "positive"], "unclear"),
    ],
)
def test_majority_or_unclear(to_gold, labels, gold):
    assert to_gold.majority(labels) == gold


def test_label_aliases_and_unknown_label(to_gold):
    assert to_gold.normalise_label(" POS ", "x") == "positive"
    assert to_gold.normalise_label("unklar", "x") == "unclear"
    assert to_gold.normalise_label("nm", "x") == "not_mentioned"
    assert to_gold.normalise_label("", "x") is None
    with pytest.raises(to_gold.InputError, match="unknown label"):
        to_gold.normalise_label("super", "t9 column a1")


def test_merge_and_convert_end_to_end(to_gold, tmp_path, capsys, lexicon):
    f1, f2 = _annotated(tmp_path)
    out = tmp_path / "gold.jsonl"
    assert to_gold.main([str(f1), str(f2), "-o", str(out)]) == 0

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    for inst in lines:
        jsonschema.validate(inst, SCHEMA)
    by = {i["text_id"]: i for i in lines}
    assert set(by) == {"t1", "t2", "t3", "t4"}  # t5 excluded, t6 incomplete
    assert by["t1"]["gold"] == "positive"
    assert by["t1"]["annotators"] == {"a1": "positive", "a2": "positive"}
    assert by["t2"]["tags"] == ["negation"]  # confirmed hint kept, 'negation?' dropped
    assert by["t3"]["gold"] == "unclear"
    assert by["t3"]["tags"] == ["irony", "dissent"]
    assert by["t3"]["note"] == "Ironie, Absicht unklar"
    assert "tags" not in by["t4"]
    assert by["t4"]["source"] == "g"
    assert by["t4"]["aspect"] == "schule"
    assert "volksschule" in by["t4"]["aspect_terms"]

    report = capsys.readouterr().out
    assert "excluded      -1" in report and "incomplete    -1" in report
    assert "a1 vs a2        n=4 agreement=0.750" in report
    assert "slice irony        1 (< 20)" in report

    gold = load_gold(out)
    res = evaluate(lexicon, gold)
    assert res.n == 4 and res.alt_test is not None


def test_aspect_overall_omits_aspect_fields(to_gold, tmp_path):
    f1, f2 = _annotated(tmp_path)
    out = tmp_path / "gold.jsonl"
    assert to_gold.main([str(f1), str(f2), "-o", str(out), "--aspect", "overall"]) == 0
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "aspect" not in first and "aspect_terms" not in first


def test_third_annotator_resolves_dissent(to_gold, tmp_path):
    f1, f2 = _annotated(tmp_path)
    f3 = _write(
        tmp_path / "a3.csv",
        [{"text_id": "t3", "text": ROWS[2]["text"], "a3": "negative"}],
        columns=["text_id", "text", "a3"],
    )
    out = tmp_path / "gold.jsonl"
    assert to_gold.main([str(f1), str(f2), str(f3), "-o", str(out)]) == 0
    by = {i["text_id"]: i for i in map(json.loads, out.read_text().splitlines())}
    assert by["t3"]["gold"] == "negative"
    assert by["t3"]["annotators"] == {"a1": "negative", "a2": "unclear", "a3": "negative"}
    assert "dissent" in by["t3"]["tags"]


def test_conflicting_cells_are_an_error(to_gold, tmp_path, capsys):
    f1, f2 = _annotated(tmp_path)
    same = _write(tmp_path / "same.csv", [{**ROWS[0], "a1": "positive"}])  # alias of 'pos'
    assert to_gold.main([str(f1), str(f2), str(same), "-o", str(tmp_path / "g.jsonl")]) == 0
    clash = _write(tmp_path / "clash.csv", [{**ROWS[0], "a1": "negative"}])
    assert to_gold.main([str(f1), str(clash), "-o", str(tmp_path / "g.jsonl")]) == 1
    assert "conflicts" in capsys.readouterr().err


def test_unknown_label_exits_1(to_gold, tmp_path, capsys):
    f = _write(tmp_path / "bad.csv", [{**ROWS[0], "a1": "gut", "a2": "positive"}])
    assert to_gold.main([str(f), "-o", str(tmp_path / "g.jsonl")]) == 1
    assert "unknown label 'gut'" in capsys.readouterr().err


def test_empty_text_fails_schema_validation(to_gold, tmp_path, capsys):
    f = _write(tmp_path / "bad.csv", [{**ROWS[0], "text": " ", "a1": "pos", "a2": "pos"}])
    out = tmp_path / "g.jsonl"
    assert to_gold.main([str(f), "-o", str(out)]) == 1
    assert "schema violation" in capsys.readouterr().err
    assert not out.exists()


def test_nothing_annotated_exits_1(to_gold, tmp_path, capsys):
    f = _write(tmp_path / "empty.csv", ROWS)
    assert to_gold.main([str(f), "-o", str(tmp_path / "g.jsonl")]) == 1
    assert "at least two labels" in capsys.readouterr().err


def test_person_column_is_a_policy_violation(to_gold, tmp_path, capsys):
    row = {**ROWS[0], "a1": "pos", "a2": "pos", "author": "X"}
    f = _write(tmp_path / "p.csv", [row], columns=[*COLUMNS, "author"])
    out = tmp_path / "g.jsonl"
    assert to_gold.main([str(f), "-o", str(out)]) == 2
    assert "POLICY VIOLATION" in capsys.readouterr().err
    assert not out.exists()


def test_builder_output_round_trips(build_candidates, to_gold, tmp_path):
    cand = tmp_path / "cand.csv"
    src = ROOT / "fixtures" / "sources_synthetic.jsonl"
    assert build_candidates.main([str(src), "-o", str(cand)]) == 0
    with cand.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["a1"] = r["a2"] = "neutral"
    filled = _write(tmp_path / "filled.csv", rows)
    out = tmp_path / "gold.jsonl"
    assert to_gold.main([str(filled), "-o", str(out)]) == 0
    gold = load_gold(out)
    assert len(gold) == len(rows)
    assert not any(t.endswith("?") for g in gold for t in g.tags)
