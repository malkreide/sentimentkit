from __future__ import annotations

import pytest

from sentimentkit.models import Polarity


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Die Lehrperson war sehr freundlich.", Polarity.POSITIVE),
        ("Der Ablauf war chaotisch.", Polarity.NEGATIVE),
        ("Die Schule beginnt um acht Uhr.", Polarity.NEUTRAL),
        # negation before the sentiment word
        ("Die Betreuung war nicht gut.", Polarity.NEGATIVE),
        ("Wir hatten kein Problem.", Polarity.POSITIVE),
        # Satzklammer: negation after the sentiment word, same clause
        ("Ich bin mit der Kommunikation überhaupt nicht zufrieden.", Polarity.NEGATIVE),
        (
            "Die Schulleitung hat uns, obwohl wir nachfragten, nicht ernst genommen.",
            Polarity.NEGATIVE,
        ),
        # 'nie' after a noun does NOT negate the noun
        ("Das Servicequalitätsproblem wurde nie gelöst.", Polarity.NEGATIVE),
        # compound via suffix
        ("Die Lernfreude ist gestiegen.", Polarity.POSITIVE),
        # contrast: the part after 'aber' wins
        ("Der Unterricht ist gut, aber die Aufsicht ist mangelhaft.", Polarity.NEGATIVE),
        ("Am Anfang gab es Probleme, aber jetzt läuft alles hervorragend.", Polarity.POSITIVE),
        # 'kaum' acts as a negator
        ("Es war kaum ein Fortschritt erkennbar.", Polarity.NEGATIVE),
    ],
)
def test_polarity(lexicon, text, expected):
    d = lexicon.decide(text)
    assert d.polarity is expected, d.evidence


def test_negation_evidence_marks_direction(lexicon):
    before = lexicon.decide("Die Betreuung war nicht gut.")
    after = lexicon.decide("Ich bin damit nicht zufrieden.")
    assert any(e.startswith("NEG<") for e in before.evidence)
    # 'nicht' precedes 'zufrieden' here too — Satzklammer only when it follows
    satz = lexicon.decide("Ich habe das Angebot leider nicht genutzt.")
    assert satz.polarity in (Polarity.NEGATIVE, Polarity.NEUTRAL)
    assert after.polarity is Polarity.NEGATIVE


def test_irony_is_flagged_not_guessed(lexicon):
    d = lexicon.decide("Toll, schon wieder eine Verspätung beim Schulbus.")
    # the baseline cannot read irony; it must not confidently say positive
    assert not (d.polarity is Polarity.POSITIVE and d.confidence > 0.6)
    assert "mixed" in d.evidence or d.polarity is Polarity.UNCLEAR


def test_dialect_lowers_confidence(lexicon):
    d = lexicon.decide("D Lehrerin isch mega nett gsi und hät üs guet gholfe.")
    assert d.polarity is Polarity.POSITIVE
    assert "DIALECT?" in d.evidence
    assert d.confidence < 0.95


def test_aspect_absent_returns_not_mentioned(lexicon):
    d = lexicon.decide(
        "Die Lehrperson erklärt sehr klar.", aspect="mensa", aspect_terms=("mensa", "essen")
    )
    assert d.polarity is Polarity.NOT_MENTIONED
    assert d.distribution["not_mentioned"] == 1.0


def test_aspect_is_scoped_to_clause(lexicon):
    d = lexicon.decide(
        "Der Unterricht ist gut, der Mittagstisch dagegen ein Chaos.",
        aspect="mensa",
        aspect_terms=("mittagstisch",),
    )
    assert d.polarity is Polarity.NEGATIVE


def test_distribution_sums_to_one(lexicon):
    d = lexicon.decide("Die Räume sind etwas unklar beschildert, aber die Lehrperson ist super.")
    assert abs(sum(d.distribution.values()) - 1.0) < 1e-3
    assert -1.0 <= d.intensity <= 1.0
    assert 0.0 <= d.confidence <= 1.0


def test_model_version_reflects_lexicon(lexicon):
    assert lexicon.model_version.startswith("seed:")
    assert lexicon.model_version.endswith("de_seed.tsv")
