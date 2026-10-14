from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "fixtures" / "sources_synthetic.jsonl"
KEYWORDS = ROOT / "fixtures" / "keywords_schule.yaml"


def _candidates(mod, **kw):
    keywords, exclude = mod.load_keywords(KEYWORDS)
    stats = mod.Stats()
    cands = mod.extract_candidates(
        mod.read_documents(SOURCES), keywords, exclude, stats=stats, **kw
    )
    return cands, stats


def _rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def test_repair_splits_rejoins_ordinals_and_abbreviations(build_candidates):
    pieces = [
        "Am 3.",
        "März tagt die Kreisschulbehörde.",
        "Der Rat sprach 12 Mio.",
        "Franken für die Schule.",
        "Das gilt z.",
        "B. für den Hort.",
        "Die Tagesschule startet im August 2026.",
        "Danach folgt die nächste.",
        "Gemäss Art.",
        "5 gilt das für jede Schule.",
        "Wir brauchen Projekte dieser Art.",
        "Die Schule profitiert.",
    ]
    assert build_candidates.repair_splits(pieces) == [
        "Am 3. März tagt die Kreisschulbehörde.",
        "Der Rat sprach 12 Mio. Franken für die Schule.",
        "Das gilt z. B. für den Hort.",
        "Die Tagesschule startet im August 2026.",  # a year ends a sentence
        "Danach folgt die nächste.",
        "Gemäss Art. 5 gilt das für jede Schule.",
        "Wir brauchen Projekte dieser Art.",  # a noun ends a sentence
        "Die Schule profitiert.",
    ]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["die", "volksschulen", "wachsen"], True),
        (["neue", "kindergärten"], True),
        (["die", "hochschule", "plant"], False),
        (["hochschule", "und", "volksschule"], True),
        (["die", "eltern", "sind", "zufrieden"], False),
    ],
)
def test_school_reference_uses_substrings_and_exclude_list(build_candidates, tokens, expected):
    keywords, exclude = build_candidates.load_keywords(KEYWORDS)
    assert build_candidates.has_school_reference(tokens, keywords, exclude) is expected


def test_extract_applies_length_and_keyword_rules(build_candidates):
    cands, stats = _candidates(build_candidates)
    texts = {c.text for c in cands}
    by_id = {c.text_id: c for c in cands}
    assert stats.documents == 5
    assert stats.too_long == 1
    assert "Die Schule ist zu." not in texts  # 4 tokens
    assert "Die Hochschule plant einen neuen Studiengang." not in texts
    assert "Die Eltern zeigen sich sehr zufrieden mit dem Angebot." not in texts
    assert by_id["syn-news-001-s004"].text.startswith("Am 3. März hat die Kreisschulbehörde")
    assert all(5 <= len(build_candidates.word_tokens(c.text)) <= 60 for c in cands)
    assert not any(c.text_id.endswith("-t") for c in cands)  # titles are opt-in


def test_titles_are_opt_in_and_tagged(build_candidates):
    cands, _ = _candidates(build_candidates, include_titles=True)
    titles = [c for c in cands if c.text_id.endswith("-t")]
    assert [c.text for c in titles] == ["Neue Tagesschule im Schulkreis Nord"]
    assert titles[0].hints == ["headline"]


def test_hints_are_marked_unconfirmed(build_candidates):
    cands, _ = _candidates(build_candidates)
    by_id = {c.text_id: c for c in cands}
    assert by_id["syn-council-002-s002"].hints == ["negation?", "dialect?"]
    assert by_id["syn-news-001-s003"].hints == ["negation?"]
    assert by_id["syn-council-001-s001"].hints == []


def test_deduplicate_drops_exact_and_near_duplicates_first_wins(build_candidates):
    cands, stats = _candidates(build_candidates)
    kept = build_candidates.deduplicate(cands, threshold=0.8, stats=stats)
    ids = [c.text_id for c in kept]
    assert stats.exact_duplicates == 1 and stats.near_duplicates == 1
    assert "syn-news-001-s001" in ids
    assert "syn-news-002-s001" not in ids  # same agency sentence
    assert "syn-news-003-s001" not in ids  # "… eine weitere neue Tagesschule"
    # a stricter threshold keeps the near duplicate
    assert len(build_candidates.deduplicate(cands, threshold=0.95)) == len(kept) + 1


def test_select_balances_sources_and_is_deterministic(build_candidates):
    mk = build_candidates.Candidate
    cands = [mk(f"a{i}", f"text a{i}", "big", frozenset({f"a{i}"})) for i in range(10)]
    cands += [mk(f"b{i}", f"text b{i}", "small", frozenset({f"b{i}"})) for i in range(2)]
    picked = build_candidates.select(cands, target=6, seed=7)
    assert len(picked) == 6
    assert sum(c.source == "small" for c in picked) == 2  # small source fully used
    again = build_candidates.select(cands, target=6, seed=7)
    assert [c.text_id for c in picked] == [c.text_id for c in again]
    assert len(build_candidates.select(cands, target=500)) == 12


def test_main_writes_annotation_template(build_candidates, tmp_path, capsys):
    out = tmp_path / "cand.csv"
    assert build_candidates.main([str(SOURCES), "-o", str(out)]) == 0
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel shows umlauts
    rows = _rows(out)
    assert list(rows[0]) == ["text_id", "text", "source", "a1", "a2", "tags", "note"]
    assert len(rows) == 12
    assert all(r["a1"] == "" and r["a2"] == "" and r["note"] == "" for r in rows)
    assert len({r["text_id"] for r in rows}) == len(rows)
    report = capsys.readouterr()
    assert "selected             12" in report.out
    assert "300-500" in report.err


def test_main_semicolon_delimiter_and_target(build_candidates, tmp_path):
    out = tmp_path / "cand.csv"
    args = [str(SOURCES), "-o", str(out), "--delimiter", ";", "--target", "4"]
    assert build_candidates.main(args) == 0
    rows = _rows(out, delimiter=";")
    assert len(rows) == 4
    assert {r["source"] for r in rows} >= {"synthetisch-medien", "synthetisch-gemeinderat"}


def test_person_field_is_a_policy_violation(build_candidates, tmp_path, capsys):
    src = tmp_path / "src.jsonl"
    doc = {"id": "d1", "source": "x", "text": "Die Volksschule wächst stark.", "author": "X"}
    src.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    out = tmp_path / "cand.csv"
    assert build_candidates.main([str(src), "-o", str(out)]) == 2
    assert "POLICY VIOLATION" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"id": "d1", "source": "x"}'], "missing field"),
        (['{"id": "d1", "source": "x", "text": "a"}'] * 2, "duplicate id"),
        (["not json"], "invalid JSON"),
    ],
)
def test_malformed_input_exits_1(build_candidates, tmp_path, capsys, lines, message):
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert build_candidates.main([str(src), "-o", str(tmp_path / "c.csv")]) == 1
    assert message in capsys.readouterr().err


def test_annotated_template_is_not_overwritten(build_candidates, tmp_path):
    out = tmp_path / "cand.csv"
    assert build_candidates.main([str(SOURCES), "-o", str(out)]) == 0
    assert build_candidates.main([str(SOURCES), "-o", str(out)]) == 0  # still empty
    rows = _rows(out)
    rows[0]["a1"] = "positive"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert build_candidates.main([str(SOURCES), "-o", str(out)]) == 1
    assert _rows(out)[0]["a1"] == "positive"
    assert build_candidates.main([str(SOURCES), "-o", str(out), "--force"]) == 0
