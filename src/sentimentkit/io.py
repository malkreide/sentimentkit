"""Readers and writers. JSONL is the canonical exchange format; CSV/XLSX
exports from survey tools are the main real-world input."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from .models import Judgement, RunMeta, TextItem


def read_items(path: str | Path, *, text_column: str = "text") -> list[TextItem]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return [TextItem.model_validate(json.loads(line)) for line in _lines(p)]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return [TextItem.model_validate(row) for row in data]
    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        with p.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            items = []
            for i, row in enumerate(reader):
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                meta = {k: v for k, v in row.items() if k not in {text_column, "text_id", "lang"}}
                items.append(
                    TextItem(
                        text_id=row.get("text_id") or f"row{i + 1}",
                        text=text,
                        lang=row.get("lang") or None,
                        source=p.name,
                        meta=meta,
                    )
                )
            return items
    if suffix in {".md", ".txt"}:
        # one item per non-empty paragraph
        paras = [b.strip() for b in p.read_text(encoding="utf-8").split("\n\n") if b.strip()]
        return [TextItem(text_id=f"p{i + 1}", text=t, source=p.name) for i, t in enumerate(paras)]
    raise ValueError(f"unsupported input format: {suffix}")


def _lines(p: Path) -> Iterable[str]:
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield line


def write_judgements(path: str | Path, judgements: Iterable[Judgement]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for j in judgements:
            fh.write(j.model_dump_json(exclude_none=True) + "\n")


def write_run_meta(path: str | Path, meta: RunMeta) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_judgements(path: str | Path) -> list[Judgement]:
    return [Judgement.model_validate(json.loads(line)) for line in _lines(Path(path))]


def write_review_queue(path: str | Path, judgements: Iterable[Judgement]) -> int:
    """CSV the reviewer fills in: human_label column. Returns the row count."""
    rows = [j for j in judgements if j.needs_review]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "judgement_id",
                "text_id",
                "aspect",
                "sentence",
                "polarity",
                "confidence",
                "evidence",
                "human_label",
            ]
        )
        for j in rows:
            w.writerow(
                [
                    j.judgement_id,
                    j.text_id,
                    j.aspect,
                    j.sentence or "",
                    j.polarity.value,
                    j.confidence,
                    " ".join(j.evidence),
                    "",
                ]
            )
    return len(rows)
