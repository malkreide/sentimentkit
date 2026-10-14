#!/usr/bin/env python3
"""Build stage-1 gold-set candidates from public source documents.

    python scripts/build_gold_candidates.py SOURCES.jsonl [--target 400] [-o CSV]

Input is JSONL, one document per line: ``{"id", "source", "date", "title",
"text"}`` (date and title may be empty). Typical sources are media articles
(news-monitor-mcp) and council speeches (zurich-opendata-mcp), mapped to these
fields by a human. Lines starting with ``#`` are comments.

Rules only, no model and no network:

1. split each text with ``preprocess.split_sentences`` and re-join the false
   splits German news text produces ("am 3. März", "12 Mio. Franken", "z. B.");
2. keep sentences with 5..60 word tokens (``preprocess.tokenize``; numbers do
   not count);
3. keep sentences with a school reference (``fixtures/keywords_schule.yaml``);
4. drop exact duplicates and near-duplicates (token-set Jaccard >= 0.8), the
   first occurrence wins — agency copy appears in several outlets;
5. sample down to ``--target`` round-robin across sources, so no single outlet
   dominates, and shuffle (seeded) so annotators see sentences without their
   article context.

Output is the annotation template ``fixtures/gold_v1_candidates.csv`` with the
columns text_id, text, source, a1, a2, tags, note. ``a1``/``a2`` stay empty for
two human annotators (docs/ANNOTATION.md). ``tags`` carries rule-based hints
with a trailing ``?`` (``negation?``, ``dialect?``); annotators confirm a hint
by removing the ``?``, unconfirmed hints never reach the gold set.

The file is written as UTF-8 with BOM so that Excel shows umlauts correctly.
An existing template that already holds annotations is never overwritten
without ``--force``.

Exit codes: 0 ok · 1 error · 2 policy violation (person-level fields in the
input, see policy.yaml).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sentimentkit.backends.lexicon import LEXICON_DIR
from sentimentkit.models import TextItem
from sentimentkit.policy import PolicyViolation, load_policy
from sentimentkit.preprocess import guess_language, split_sentences, tokenize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYWORDS = ROOT / "fixtures" / "keywords_schule.yaml"
DEFAULT_OUTPUT = ROOT / "fixtures" / "gold_v1_candidates.csv"

COLUMNS = ["text_id", "text", "source", "a1", "a2", "tags", "note"]
REQUIRED_FIELDS = ("id", "source", "text")
STAGE1_RANGE = (300, 500)

# A piece ending like this was cut by split_sentences in the middle of a
# sentence: a day or class ordinal ("3. März") or a common abbreviation
# followed by a capitalised word ("12 Mio. Franken", "z. B. Die", "St. Gallen").
_FALSE_SPLIT = re.compile(
    r"(?:(?<!\d)\d{1,2}|\b(?:z|bzw|ca|dr|prof|st|vgl|evtl|inkl|sog|resp|mio|mrd|jh))\.$",
    re.IGNORECASE,
)
# Only cut when a number follows ("Art. 5 URG"): "Projekte dieser Art." is a
# genuine sentence end.
_FALSE_SPLIT_BEFORE_NUMBER = re.compile(r"\b(?:art|abs|nr|ziff)\.$", re.IGNORECASE)
_WS = re.compile(r"\s+")


class InputError(ValueError):
    """Malformed input; the CLI maps it to exit code 1."""


@dataclass
class Candidate:
    text_id: str
    text: str
    source: str
    tokens: frozenset[str] = field(repr=False)
    hints: list[str] = field(default_factory=list)


@dataclass
class Stats:
    documents: int = 0
    sentences: int = 0
    too_short: int = 0
    too_long: int = 0
    no_school_reference: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    kept: int = 0
    selected: int = 0


# --- input -------------------------------------------------------------------
def load_keywords(path: str | Path) -> tuple[list[str], list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    keywords = [str(k).lower() for k in data.get("keywords") or []]
    exclude = [str(k).lower() for k in data.get("exclude") or []]
    if not keywords:
        raise InputError(f"{path}: no 'keywords' list")
    return keywords, exclude


def read_documents(path: str | Path) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise InputError(f"{path}:{n}: expected a JSON object")
        missing = [k for k in REQUIRED_FIELDS if not str(row.get(k) or "").strip()]
        if missing:
            raise InputError(f"{path}:{n}: missing field(s) {missing}")
        doc_id = str(row["id"])
        if doc_id in seen:
            raise InputError(f"{path}:{n}: duplicate id {doc_id!r}")
        seen.add(doc_id)
        docs.append(row)
    return docs


def check_policy(docs: Sequence[dict], policy_path: str | Path | None = None) -> None:
    """Person-level fields (author, name, …) must be removed before stage 1.

    Reuses the policy's own item check: everything except id, text and
    source is treated as item meta, exactly as in a regular run."""
    items = [
        TextItem(
            text_id=str(d["id"]),
            text=str(d["text"]),
            source=str(d["source"]),
            meta={k: v for k, v in d.items() if k not in REQUIRED_FIELDS},
        )
        for d in docs
    ]
    load_policy(policy_path).check_items(items)


# --- rules -------------------------------------------------------------------
def repair_splits(pieces: Iterable[str]) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        if out and (
            _FALSE_SPLIT.search(out[-1])
            or (_FALSE_SPLIT_BEFORE_NUMBER.search(out[-1]) and piece[:1].isdigit())
        ):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def word_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in {"!", "?"}]


def has_school_reference(
    tokens: Iterable[str], keywords: Sequence[str], exclude: Sequence[str]
) -> bool:
    for tok in tokens:
        if any(x in tok for x in exclude):
            continue
        if any(k in tok for k in keywords):
            return True
    return False


def hint_tags(tokens: Sequence[str], negators: set[str]) -> list[str]:
    """Rule-based slice hints. Never final: the trailing '?' marks them as
    unconfirmed until an annotator removes it."""
    hints = []
    if negators & set(tokens):
        hints.append("negation?")
    if guess_language(tokens)[1]:
        hints.append("dialect?")
    return hints


def load_negators() -> set[str]:
    data = yaml.safe_load((LEXICON_DIR / "de_modifiers.yaml").read_text(encoding="utf-8"))
    return {str(n).lower() for n in data.get("negators", [])}


def extract_candidates(
    docs: Sequence[dict],
    keywords: Sequence[str],
    exclude: Sequence[str],
    *,
    min_tokens: int = 5,
    max_tokens: int = 60,
    include_titles: bool = False,
    stats: Stats | None = None,
) -> list[Candidate]:
    stats = stats if stats is not None else Stats()
    negators = load_negators()
    out: list[Candidate] = []
    for doc in docs:
        stats.documents += 1
        doc_id, source = str(doc["id"]), str(doc["source"])
        units: list[tuple[str, str, list[str]]] = []
        title = _WS.sub(" ", str(doc.get("title") or "")).strip()
        if include_titles and title:
            units.append((f"{doc_id}-t", title, ["headline"]))
        pieces = repair_splits(s.text for s in split_sentences(str(doc["text"])))
        for i, piece in enumerate(pieces, 1):
            units.append((f"{doc_id}-s{i:03d}", _WS.sub(" ", piece).strip(), []))
        for text_id, text, base_tags in units:
            stats.sentences += 1
            toks = word_tokens(text)
            if len(toks) < min_tokens:
                stats.too_short += 1
                continue
            if len(toks) > max_tokens:
                stats.too_long += 1
                continue
            if not has_school_reference(toks, keywords, exclude):
                stats.no_school_reference += 1
                continue
            out.append(
                Candidate(
                    text_id=text_id,
                    text=text,
                    source=source,
                    tokens=frozenset(toks),
                    hints=base_tags + hint_tags(toks, negators),
                )
            )
    return out


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def deduplicate(
    cands: Sequence[Candidate], *, threshold: float = 0.8, stats: Stats | None = None
) -> list[Candidate]:
    stats = stats if stats is not None else Stats()
    kept: list[Candidate] = []
    exact: set[str] = set()
    for c in cands:
        key = " ".join(word_tokens(c.text))
        if key in exact:
            stats.exact_duplicates += 1
            continue
        # Jaccard can only reach the threshold if the smaller set is at least
        # `threshold` times the larger one — skip the rest without intersecting.
        n = len(c.tokens)
        if any(
            min(n, len(k.tokens)) >= threshold * max(n, len(k.tokens))
            and jaccard(c.tokens, k.tokens) >= threshold
            for k in kept
        ):
            stats.near_duplicates += 1
            continue
        exact.add(key)
        kept.append(c)
    stats.kept = len(kept)
    return kept


def select(cands: Sequence[Candidate], *, target: int, seed: int = 1) -> list[Candidate]:
    """Round-robin across sources up to `target`, then a seeded shuffle."""
    rng = random.Random(seed)
    by_source: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_source[c.source].append(c)
    queues = []
    for src in sorted(by_source):
        q = list(by_source[src])
        rng.shuffle(q)
        queues.append(q)
    picked: list[Candidate] = []
    while len(picked) < target and any(queues):
        for q in queues:
            if q and len(picked) < target:
                picked.append(q.pop())
    rng.shuffle(picked)
    return picked


# --- output ------------------------------------------------------------------
def _detect_delimiter(header: str) -> str:
    return max([",", ";", "\t"], key=header.count)


def has_annotations(path: Path) -> bool:
    """True if an existing template already carries labels in a1/a2/…"""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return False
    reader = csv.DictReader(io.StringIO(text), delimiter=_detect_delimiter(text.splitlines()[0]))
    label_cols = [c for c in reader.fieldnames or [] if re.fullmatch(r"a\d+", c or "")]
    return any((row.get(c) or "").strip() for row in reader for c in label_cols)


def write_csv(path: str | Path, cands: Sequence[Candidate], *, delimiter: str = ",") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(COLUMNS)
        for c in cands:
            w.writerow([c.text_id, c.text, c.source, "", "", " ".join(c.hints), ""])


# --- CLI ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build stage-1 gold-set candidates (annotation CSV) from source documents."
    )
    ap.add_argument("input", type=Path, help="JSONL with id, source, date, title, text")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    ap.add_argument("--target", type=int, default=400, help="sentences to select (300-500)")
    ap.add_argument("--min-tokens", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--near-dup", type=float, default=0.8, help="Jaccard threshold")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-titles", action="store_true", help="titles as candidates too")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter, e.g. ';' for Excel de-CH")
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an annotated template")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target < 1:
        print("error: --target must be >= 1", file=sys.stderr)
        return 1
    try:
        keywords, exclude = load_keywords(args.keywords)
        docs = read_documents(args.input)
        check_policy(docs, args.policy)
    except PolicyViolation as exc:
        print(f"POLICY VIOLATION: {exc}", file=sys.stderr)
        print("Remove person-level fields from the input; see policy.yaml.", file=sys.stderr)
        return 2
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if has_annotations(args.output) and not args.force:
        print(
            f"error: {args.output} already contains annotations; use another --output or --force",
            file=sys.stderr,
        )
        return 1

    stats = Stats()
    cands = extract_candidates(
        docs,
        keywords,
        exclude,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        include_titles=args.include_titles,
        stats=stats,
    )
    cands = deduplicate(cands, threshold=args.near_dup, stats=stats)
    picked = select(cands, target=args.target, seed=args.seed)
    stats.selected = len(picked)
    if not picked:
        print("error: no candidate sentence left after filtering", file=sys.stderr)
        return 1
    write_csv(args.output, picked, delimiter=args.delimiter)

    print(f"documents            {stats.documents}")
    print(f"sentences            {stats.sentences}")
    print(f"  too short (<{args.min_tokens})     -{stats.too_short}")
    print(f"  too long (>{args.max_tokens})     -{stats.too_long}")
    print(f"  no school ref.     -{stats.no_school_reference}")
    print(f"  exact duplicates   -{stats.exact_duplicates}")
    print(f"  near duplicates    -{stats.near_duplicates}")
    print(f"candidates           {stats.kept}")
    print(f"selected             {stats.selected} -> {args.output}")
    hints: dict[str, int] = defaultdict(int)
    for c in picked:
        for h in c.hints:
            hints[h] += 1
    for h in sorted(hints):
        print(f"  hint {h:14} {hints[h]}")
    lo, hi = STAGE1_RANGE
    if not lo <= stats.selected <= hi:
        print(
            f"note: stage 1 aims at {lo}-{hi} sentences; got {stats.selected}. "
            "Collect more source documents or adjust --target.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
