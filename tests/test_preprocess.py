from __future__ import annotations

import pytest

from sentimentkit.preprocess import split_sentences


def _texts(text: str) -> list[str]:
    return [s.text for s in split_sentences(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Am 3. März tagt die Kreisschulbehörde. Danach folgt die nächste Sitzung.",
            ["Am 3. März tagt die Kreisschulbehörde.", "Danach folgt die nächste Sitzung."],
        ),
        (
            "Der Rat sprach 12 Mio. Franken für die Schule. Das Geld reicht.",
            ["Der Rat sprach 12 Mio. Franken für die Schule.", "Das Geld reicht."],
        ),
        ("Das gilt z. B. für den Hort.", ["Das gilt z. B. für den Hort."]),
        (
            "Die Kinder fahren nach St. Gallen. Sie kommen am Abend zurück.",
            ["Die Kinder fahren nach St. Gallen.", "Sie kommen am Abend zurück."],
        ),
        (
            "Gemäss Art. 5 gilt das für jede Schule. Die Frist läuft.",
            ["Gemäss Art. 5 gilt das für jede Schule.", "Die Frist läuft."],
        ),
    ],
)
def test_ordinals_and_abbreviations_stay_inside_the_sentence(text, expected):
    assert _texts(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a year ends a sentence
        (
            "Die Tagesschule startet im August 2026. Danach folgt die nächste.",
            ["Die Tagesschule startet im August 2026.", "Danach folgt die nächste."],
        ),
        # "Art." as a noun ends a sentence when no number follows
        (
            "Wir brauchen Projekte dieser Art. Die Schule profitiert.",
            ["Wir brauchen Projekte dieser Art.", "Die Schule profitiert."],
        ),
        # "ist." is not the abbreviation "St."
        ("Die Betreuung ist. Das reicht.", ["Die Betreuung ist.", "Das reicht."]),
        ("Gut! Schlecht? Neutral.", ["Gut!", "Schlecht?", "Neutral."]),
    ],
)
def test_genuine_sentence_ends_still_split(text, expected):
    assert _texts(text) == expected


def test_rejoined_sentence_keeps_offsets_and_original_whitespace():
    text = "  Vorwort. Am 3.\nMärz tagt die Behörde. Ende."
    sents = split_sentences(text)
    assert [s.text for s in sents] == ["Vorwort.", "Am 3.\nMärz tagt die Behörde.", "Ende."]
    for s in sents:
        assert text[s.start : s.end] == s.text
