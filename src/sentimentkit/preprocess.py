"""German-aware preprocessing that needs no model.

Deliberately small: sentence split, tokenisation, a stop-word based language
guess, a Swiss-German marker check and a lexicon-driven decompounder. The
research behind the German specifics (compounds, Satzklammer negation,
dialect) is summarised in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÄÖÜ«\"(0-9])")
_TOKEN = re.compile(r"[A-Za-zÄÖÜäöüßÀ-ÿ']+|[!?]", re.UNICODE)

STOPWORDS = {
    "de": {"und", "der", "die", "das", "nicht", "ist", "mit", "ein", "eine", "wir", "sich", "auch"},
    "fr": {"et", "le", "la", "les", "pas", "est", "une", "des", "nous", "avec", "pour"},
    "it": {"e", "il", "la", "non", "che", "una", "per", "con", "sono", "del", "della"},
    "en": {"and", "the", "is", "not", "with", "we", "are", "this", "that", "for", "of"},
}

# Markers that are (nearly) exclusive to written Swiss German.
SWISS_GERMAN_MARKERS = {
    "nöd",
    "nid",
    "isch",
    "sind",
    "gsi",
    "hät",
    "chli",
    "chönd",
    "z'",
    "mir",
    "au",
    "ned",
    "uf",
    "nüt",
    "öppis",
    "gäll",
    "wüki",
    "mega",
    "schaffe",
    "chunt",
}


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    out: list[Sentence] = []
    pos = 0
    for piece in _SENT_SPLIT.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        start = text.find(piece, pos)
        if start < 0:
            start = pos
        end = start + len(piece)
        out.append(Sentence(piece, start, end))
        pos = end
    return out


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def guess_language(tokens: Iterable[str]) -> tuple[str, bool]:
    """Return (lang, dialect_suspected). Stop-word overlap, nothing fancy.

    Unknown texts default to 'de' because that is the toolkit's primary
    domain; the caller may override via TextItem.lang.
    """
    toks = set(tokens)
    scores = {lang: len(toks & words) for lang, words in STOPWORDS.items()}
    lang = max(scores, key=scores.get) if any(scores.values()) else "de"
    swiss = len(toks & SWISS_GERMAN_MARKERS)
    dialect = swiss >= 2 or (lang == "de" and swiss >= 1 and scores["de"] <= 1)
    return ("gsw" if dialect else lang), dialect


def decompound(token: str, known: Iterable[str], min_len: int = 4) -> str | None:
    """Return the longest known suffix of a compound, or None.

    'servicequalitätsproblem' -> 'problem' when 'problem' is in `known`.
    A real decompounder (e.g. CharSplit) is Stage-2 territory; this suffix
    match already catches the common 'X…problem', 'X…mangel', 'X…freude'.
    """
    best: str | None = None
    for w in known:
        if len(w) >= min_len and token != w and token.endswith(w):
            # allow a Fugen-s before the suffix: 'qualitätsproblem'
            if best is None or len(w) > len(best):
                best = w
    return best
