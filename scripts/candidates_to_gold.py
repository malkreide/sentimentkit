#!/usr/bin/env python3
"""Turn annotated candidate CSVs into the gold-set JSONL.

    python scripts/candidates_to_gold.py ANNOTATED.csv [MORE.csv …] [-o fixtures/gold_v1.jsonl]

Each annotator may work on an own copy of ``fixtures/gold_v1_candidates.csv``
and fill only their column — that keeps the labels independent. All files are
merged by ``text_id``; the same cell filled differently in two files is an
error, never a silent choice.

Per row:

* labels come from every column ``a1``, ``a2``, ``a3`` …; accepted are the five
  gold labels and a few aliases (``pos``, ``neg``, ``positiv``, ``unklar`` …);
* rows with fewer than two labels are skipped (not yet annotated), rows tagged
  ``exclude`` are dropped (personal name, broken sentence, …);
* ``gold`` is the strict majority of the labels, ``unclear`` when there is
  none — with two annotators: agreement or ``unclear``. Any disagreement adds
  the tag ``dissent`` so the eval can slice contested instances;
* tags ending in ``?`` are unconfirmed hints from the candidate builder and
  are dropped;
* ``aspect`` (default ``schule``) and ``aspect_terms`` (the keyword list that
  selected the sentence) make the instance a typed question, so
  ``not_mentioned`` is a reachable answer.

Every line is validated against ``schema/gold.schema.json`` before anything is
written. The summary reports Cohen's kappa a1 vs. a2 (target >= 0.7) and the
tag slices the stage-1 gate counts.

Exit codes: 0 ok · 1 error · 2 policy violation (person-level column).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import jsonschema
import yaml

from sentimentkit.eval.metrics import cohen_kappa
from sentimentkit.models import Polarity, TextItem
from sentimentkit.policy import PolicyViolation, load_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "fixtures" / "gold_v1.jsonl"
DEFAULT_KEYWORDS = ROOT / "fixtures" / "keywords_schule.yaml"
DEFAULT_SCHEMA = ROOT / "schema" / "gold.schema.json"

LABELS = {p.value for p in Polarity}
ALIASES = {
    "pos": "positive",
    "positiv": "positive",
    "neu": "neutral",
    "neg": "negative",
    "negativ": "negative",
    "unc": "unclear",
    "unklar": "unclear",
    "nm": "not_mentioned",
    "not mentioned": "not_mentioned",
    "nicht erwähnt": "not_mentioned",
    "nicht_erwähnt": "not_mentioned",
}
KNOWN_COLUMNS = {"text_id", "text", "source", "tags", "note"}
LABEL_COLUMN = re.compile(r"a\d+")
TAG_SPLIT = re.compile(r"[\s,;|]+")
GATE_SLICES = ("negation", "irony", "dialect", "aspect")
GATE_MIN_SLICE = 20


class InputError(ValueError):
    """Malformed or conflicting annotation input; exit code 1."""


def _detect_delimiter(header: str) -> str:
    return max([",", ";", "\t"], key=header.count)


def normalise_label(raw: str, where: str) -> str | None:
    value = raw.strip().lower()
    if not value:
        return None
    value = ALIASES.get(value, value)
    if value not in LABELS:
        raise InputError(f"{where}: unknown label {raw!r}; allowed: {', '.join(sorted(LABELS))}")
    return value


def read_annotations(path: str | Path) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        raise InputError(f"{path}: empty file")
    reader = csv.DictReader(io.StringIO(text), delimiter=_detect_delimiter(text.splitlines()[0]))
    cols = [c.strip() for c in reader.fieldnames or []]
    missing = [c for c in ("text_id", "text") if c not in cols]
    if missing:
        raise InputError(f"{path}: missing column(s) {missing}")
    rows = []
    for row in reader:
        rows.append({(k or "").strip(): (v or "") for k, v in row.items() if k is not None})
    return rows


def check_policy(rows: Sequence[dict[str, str]], policy_path: str | Path | None = None) -> None:
    """Extra columns are item meta; person-level ones abort like a regular run."""
    items = [
        TextItem(
            text_id=r.get("text_id") or "?",
            text=r.get("text", "").strip() or "?",
            meta={
                k: v
                for k, v in r.items()
                if k not in KNOWN_COLUMNS and not LABEL_COLUMN.fullmatch(k)
            },
        )
        for r in rows
    ]
    load_policy(policy_path).check_items(items)


def _comparable(key: str, value: str) -> str:
    if LABEL_COLUMN.fullmatch(key):
        return ALIASES.get(value.lower(), value.lower())
    return value


def merge(files: Sequence[tuple[str, list[dict[str, str]]]]) -> dict[str, dict[str, str]]:
    """Merge rows by text_id. Labels and text must agree where both are filled;
    tags are united, notes concatenated."""
    merged: dict[str, dict[str, str]] = {}
    for name, rows in files:
        seen: set[str] = set()
        for row in rows:
            tid = row.get("text_id", "").strip()
            if not tid:
                raise InputError(f"{name}: row without text_id")
            if tid in seen:
                raise InputError(f"{name}: duplicate text_id {tid!r}")
            seen.add(tid)
            cur = merged.setdefault(tid, {"text_id": tid})
            for key, value in row.items():
                value = value.strip()
                if key == "text_id" or not value:
                    continue
                if key == "tags":
                    have = TAG_SPLIT.split(cur.get("tags", "")) if cur.get("tags") else []
                    new = [t for t in TAG_SPLIT.split(value) if t and t not in have]
                    cur["tags"] = " ".join(have + new)
                elif key == "note":
                    notes = [n for n in cur.get("note", "").split(" | ") if n]
                    if value not in notes:
                        cur["note"] = " | ".join(notes + [value])
                elif key in cur and _comparable(key, cur[key]) != _comparable(key, value):
                    raise InputError(
                        f"{name}: {tid} column {key!r} conflicts with an earlier file "
                        f"({cur[key]!r} vs {value!r})"
                    )
                else:
                    cur[key] = value
    return merged


def majority(labels: Sequence[str]) -> str:
    label, count = Counter(labels).most_common(1)[0]
    return label if count * 2 > len(labels) else Polarity.UNCLEAR.value


def to_instance(
    row: dict[str, str], *, aspect: str, aspect_terms: Sequence[str]
) -> tuple[dict | None, str]:
    """(instance, status) with status one of ok, excluded, incomplete."""
    tid = row["text_id"]
    tags = [t for t in TAG_SPLIT.split(row.get("tags", "")) if t]
    if "exclude" in (t.lower() for t in tags):
        return None, "excluded"
    annotators: dict[str, str] = {}
    for col in sorted((c for c in row if LABEL_COLUMN.fullmatch(c)), key=lambda c: int(c[1:])):
        label = normalise_label(row[col], f"{tid} column {col}")
        if label:
            annotators[col] = label
    if len(annotators) < 2:
        return None, "incomplete"
    labels = list(annotators.values())
    tags = [t.lower() for t in tags if not t.endswith("?")]
    if len(set(labels)) > 1:
        tags.append("dissent")
    inst: dict = {"text_id": tid, "text": row.get("text", "").strip(), "gold": majority(labels)}
    if aspect != "overall":
        inst["aspect"] = aspect
        inst["aspect_terms"] = list(aspect_terms)
    inst["annotators"] = annotators
    if tags:
        inst["tags"] = list(dict.fromkeys(tags))
    if row.get("source"):
        inst["source"] = row["source"]
    if row.get("note"):
        inst["note"] = row["note"]
    return inst, "ok"


def validate(instances: Sequence[dict], schema_path: str | Path) -> None:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for inst in instances:
        errors = sorted(validator.iter_errors(inst), key=lambda e: list(e.path))
        if errors:
            raise InputError(f"{inst.get('text_id')}: schema violation: {errors[0].message}")


def agreement(instances: Sequence[dict]) -> tuple[int, float, float] | None:
    """(n, raw agreement, Cohen's kappa) for a1 vs. a2 where both labelled."""
    pairs = [
        (i["annotators"]["a1"], i["annotators"]["a2"])
        for i in instances
        if "a1" in i["annotators"] and "a2" in i["annotators"]
    ]
    if not pairs:
        return None
    a, b = zip(*pairs)
    raw = sum(x == y for x, y in pairs) / len(pairs)
    return len(pairs), raw, cohen_kappa(a, b)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Annotated candidate CSV(s) -> gold-set JSONL.")
    ap.add_argument("inputs", type=Path, nargs="+", help="annotated CSV file(s)")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--aspect", default="schule", help="'overall' omits aspect fields")
    ap.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS, help="aspect_terms")
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--min-kappa", type=float, default=0.7)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = [(str(p), read_annotations(p)) for p in args.inputs]
        for _, rows in files:
            check_policy(rows, args.policy)
        terms: list[str] = []
        if args.aspect != "overall":
            data = yaml.safe_load(args.keywords.read_text(encoding="utf-8")) or {}
            terms = [str(k).lower() for k in data.get("keywords") or []]
        status: Counter[str] = Counter()
        instances = []
        for row in merge(files).values():
            inst, state = to_instance(row, aspect=args.aspect, aspect_terms=terms)
            status[state] += 1
            if inst is not None:
                instances.append(inst)
        if not instances:
            raise InputError("no row with at least two labels")
        validate(instances, args.schema)
    except PolicyViolation as exc:
        print(f"POLICY VIOLATION: {exc}", file=sys.stderr)
        return 2
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for inst in instances:
            fh.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"rows            {sum(status.values())}")
    print(f"  excluded      -{status['excluded']}")
    print(f"  incomplete    -{status['incomplete']}")
    print(f"gold instances  {len(instances)} -> {args.output}")
    for label, n in Counter(i["gold"] for i in instances).most_common():
        print(f"  {label:14} {n}")
    agree = agreement(instances)
    if agree:
        n, raw, kappa = agree
        verdict = "ok" if kappa >= args.min_kappa else f"BELOW TARGET {args.min_kappa}"
        print(f"a1 vs a2        n={n} agreement={raw:.3f} kappa={kappa:.3f} ({verdict})")
    tags = Counter(t for i in instances for t in i.get("tags", []))
    for t in GATE_SLICES:
        mark = "ok" if tags[t] >= GATE_MIN_SLICE else f"< {GATE_MIN_SLICE}"
        print(f"  slice {t:9} {tags[t]:4} ({mark})")
    print(f"  slice dissent   {tags['dissent']:4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
