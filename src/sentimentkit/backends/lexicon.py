"""Lexicon + rules baseline (Stage 0).

Transparent by construction: every decision lists the tokens and modifiers
that fired. It handles the four German traps the research names — negation
before *and* after the sentiment word (Satzklammer), intensifiers/diminishers,
contrast ("…, aber …") and compounds via suffix matching — and it is
deliberately weak on irony and dialect, which it flags rather than guesses.

It is the bar every later backend has to clear on the gold set.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from ..models import Polarity
from ..preprocess import aspect_clauses, decompound, guess_language, split_sentences
from .base import ASPECT_OPTIONS, DEFAULT_OPTIONS, Backend, Decision, not_mentioned

LEXICON_DIR = Path(__file__).resolve().parents[3] / "lexicon"
_TOK = re.compile(r"[A-Za-zÄÖÜäöüßÀ-ÿ']+|[,;:.!?]", re.UNICODE)
_CLAUSE_END = {",", ";", ":", ".", "!", "?"}

POLAR_THRESHOLD = 0.15


class LexiconBackend(Backend):
    name = "lexicon"
    stage = 0

    def __init__(
        self,
        lexicon_path: Path | None = None,
        modifiers_path: Path | None = None,
    ) -> None:
        self.lexicon_path = lexicon_path or LEXICON_DIR / "de_seed.tsv"
        self.modifiers_path = modifiers_path or LEXICON_DIR / "de_modifiers.yaml"
        self.lexicon = self._load_lexicon(self.lexicon_path)
        mods = yaml.safe_load(self.modifiers_path.read_text(encoding="utf-8")) or {}
        self.negators: set[str] = set(mods.get("negators", []))
        self.satzklammer_negators: set[str] = set(mods.get("satzklammer_negators", ["nicht"]))
        self.negation_immune: set[str] = set(mods.get("negation_immune", []))
        self.window_before: int = int(mods.get("window_before", 3))
        self.intensifiers: dict[str, float] = dict(mods.get("intensifiers", {}))
        self.diminishers: dict[str, float] = dict(mods.get("diminishers", {}))
        self.contrast: set[str] = set(mods.get("contrast", []))
        self.w_after = float(mods.get("contrast_weight_after", 1.5))
        self.w_before = float(mods.get("contrast_weight_before", 0.5))
        self._version = f"seed:{len(self.lexicon)}:{self.lexicon_path.name}"
        self._max_ngram = max((len(k.split()) for k in self.lexicon), default=1)

    # ------------------------------------------------------------------ io
    @staticmethod
    def _load_lexicon(path: Path) -> dict[str, float]:
        lex: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            term, _, value = line.partition("\t")
            try:
                lex[term.strip().lower()] = float(value)
            except ValueError:
                continue
        return lex

    @property
    def model_version(self) -> str:
        return self._version

    # --------------------------------------------------------------- scoring
    def _score_sentence(self, sentence: str) -> tuple[float, int, list[str], bool, bool]:
        """Return (weighted sum, hits, evidence, saw_positive, saw_negative)."""
        toks = [t.lower() for t in _TOK.findall(sentence)]
        n = len(toks)
        # contrast weighting: position of the first contrast marker, if any
        contrast_at = next((i for i, t in enumerate(toks) if t in self.contrast), None)

        total = 0.0
        hits = 0
        evidence: list[str] = []
        saw_pos = saw_neg = False

        skip_until = -1
        for i, tok in enumerate(toks):
            if i < skip_until or tok in _CLAUSE_END or tok in self.negators:
                continue
            value: float | None = None
            label = tok
            # longest multiword entry first ("ernst genommen", "sehr gut")
            for n_gram in range(min(self._max_ngram, n - i), 1, -1):
                phrase = " ".join(toks[i : i + n_gram])
                if phrase in self.lexicon:
                    value, label, skip_until = self.lexicon[phrase], phrase, i + n_gram
                    break
            if value is None:
                value = self.lexicon.get(tok)
            if value is None:
                suffix = decompound(tok, self.lexicon.keys())
                if suffix is None:
                    continue
                value = self.lexicon[suffix]
                label = f"{tok}<{suffix}"
            hits += 1

            # (a) negation before, within the window, not crossing a clause end
            negated = False
            immune = label in self.negation_immune
            for j in range(i - 1, max(-1, i - self.window_before - 1), -1):
                if immune:
                    break
                if toks[j] in _CLAUSE_END:
                    break  # walk backwards; stop at the clause boundary
                if toks[j] in self.negators:
                    negated = True
                    evidence.append(f"NEG<{toks[j]}")
                    break
            # (b) Satzklammer: negation after, until the clause ends
            if not negated and not immune:
                for j in range(i + 1, n):
                    if toks[j] in _CLAUSE_END:
                        break
                    if toks[j] in self.satzklammer_negators:
                        negated = True
                        evidence.append(f"NEG>{toks[j]}")
                        break

            factor = 1.0
            for j in range(max(0, i - 2), i):
                if toks[j] in self.intensifiers:
                    factor *= self.intensifiers[toks[j]]
                    evidence.append(f"INT:{toks[j]}")
                elif toks[j] in self.diminishers:
                    factor *= self.diminishers[toks[j]]
                    evidence.append(f"DIM:{toks[j]}")

            if contrast_at is not None:
                factor *= self.w_after if i > contrast_at else self.w_before
                if i == contrast_at + 1 or (i < contrast_at and "CONTRAST" not in evidence):
                    evidence.append("CONTRAST")

            score = value * factor
            if negated:
                score = -score * 0.8  # 'nicht gut' is weaker than 'schlecht'
            total += score
            evidence.append(f"{label}({score:+.2f})")
            saw_pos |= score > 0
            saw_neg |= score < 0

        return total, hits, evidence, saw_pos, saw_neg

    def decide(
        self,
        text: str,
        *,
        aspect: str = "overall",
        aspect_terms: tuple[str, ...] = (),
        lang: str | None = None,
    ) -> Decision:
        units = [s.text for s in split_sentences(text)] or [text]
        if aspect != "overall" and aspect_terms:
            units = aspect_clauses(text, aspect_terms)
            if not units:
                return not_mentioned(aspect)

        total = 0.0
        hits = 0
        evidence: list[str] = []
        saw_pos = saw_neg = False
        for u in units:
            s, h, ev, p, q = self._score_sentence(u)
            total += s
            hits += h
            evidence.extend(ev)
            saw_pos |= p
            saw_neg |= q

        all_tokens = [t.lower() for t in _TOK.findall(text)]
        detected, dialect = guess_language(all_tokens)
        lang = lang or detected
        if dialect:
            evidence.append("DIALECT?")

        intensity = max(-1.0, min(1.0, total / max(1.0, math.sqrt(hits)))) if hits else 0.0

        if hits == 0:
            polarity, confidence = Polarity.NEUTRAL, 0.6  # neutral by absence: weak
            evidence.append("no-lexicon-hit")
        elif intensity > POLAR_THRESHOLD:
            polarity = Polarity.POSITIVE
            confidence = min(0.95, 0.5 + 0.15 * hits + 0.25 * abs(intensity))
        elif intensity < -POLAR_THRESHOLD:
            polarity = Polarity.NEGATIVE
            confidence = min(0.95, 0.5 + 0.15 * hits + 0.25 * abs(intensity))
        elif saw_pos and saw_neg:
            polarity, confidence = Polarity.UNCLEAR, 0.4
            evidence.append("mixed")
        else:
            polarity, confidence = Polarity.NEUTRAL, 0.55

        if dialect:
            confidence *= 0.8

        pos = max(0.0, intensity)
        neg = max(0.0, -intensity)
        neu = max(0.0, 1.0 - abs(intensity))
        unc = 0.15 if (saw_pos and saw_neg) else 0.05
        raw = {
            Polarity.POSITIVE.value: pos,
            Polarity.NEUTRAL.value: neu,
            Polarity.NEGATIVE.value: neg,
            Polarity.UNCLEAR.value: unc,
        }
        z = sum(raw.values()) or 1.0
        dist = {k: round(v / z, 4) for k, v in raw.items()}
        if aspect != "overall":
            dist[Polarity.NOT_MENTIONED.value] = 0.0
        options = ASPECT_OPTIONS if aspect != "overall" else DEFAULT_OPTIONS
        assert polarity in options
        return Decision(polarity, round(intensity, 4), round(confidence, 4), dist, evidence)
