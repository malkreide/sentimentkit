"""Command line interface.

    sentimentkit backends
    sentimentkit analyze INPUT --purpose media_resonance [--backend lexicon] [--aspects aspects.yaml] -o out/
    sentimentkit eval GOLD.jsonl [--backend lexicon] [--json]
    sentimentkit policy-check [--policy policy.yaml]
    sentimentkit validate judgements.jsonl

Exit codes: 0 ok · 1 error · 2 policy violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from . import __version__
from .backends import available, get_backend
from .eval import evaluate, load_gold
from .io import read_items, write_judgements, write_review_queue, write_run_meta
from .models import Judgement
from .pipeline import run as run_pipeline
from .policy import PolicyViolation, load_policy
from .report import markdown_report

EXIT_POLICY = 2


@click.group()
@click.version_option(__version__, prog_name="sentimentkit")
def main() -> None:
    """Sentiment toolkit with swappable backends, evidence atoms and policy-as-code."""


@main.command()
def backends() -> None:
    """List backends and the roadmap stage in which they become usable."""
    for name, b in available().items():
        state = "available" if b.stage == 0 else f"stage {b.stage}"
        note = getattr(b, "note", "")
        click.echo(f"{name:10} {state:12} {note}")


@main.command("policy-check")
@click.option("--policy", "policy_path", type=click.Path(exists=True, dir_okay=False), default=None)
def policy_check(policy_path: str | None) -> None:
    """Load and print the policy (and its hash)."""
    pol = load_policy(policy_path)
    click.echo(f"policy v{pol.version} · hash {pol.hash}")
    click.echo(f"allowed purposes : {', '.join(pol.allowed_purposes)}")
    click.echo(f"forbidden dims   : {', '.join(pol.forbidden_dimensions)}")
    click.echo(
        f"aggregation floor: {pol.aggregation_floor} · review threshold: {pol.review_threshold}"
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--purpose", required=True, help="Declared purpose (must be allowed by policy).")
@click.option("--backend", "backend_name", default="lexicon", show_default=True)
@click.option(
    "--aspects",
    "aspects_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="YAML: {aspect: [term, …]} — one typed question per aspect.",
)
@click.option("--policy", "policy_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--anonymized", is_flag=True, help="Assert that the input is anonymised.")
@click.option("--context", default=None, help="Deployment context, checked against policy.")
@click.option("--text-column", default="text", show_default=True)
@click.option("-o", "--out", "out_dir", type=click.Path(file_okay=False), default="out")
def analyze(
    input_path: str,
    purpose: str,
    backend_name: str,
    aspects_path: str | None,
    policy_path: str | None,
    anonymized: bool,
    context: str | None,
    text_column: str,
    out_dir: str,
) -> None:
    """Judge every text in INPUT and write judgements, run meta, review queue and report."""
    pol = load_policy(policy_path)
    aspects = None
    if aspects_path:
        raw = yaml.safe_load(Path(aspects_path).read_text(encoding="utf-8")) or {}
        aspects = {str(k): tuple(str(t) for t in v) for k, v in raw.items()}
    try:
        items = read_items(input_path, text_column=text_column)
        backend = get_backend(backend_name)
        result = run_pipeline(
            items,
            policy=pol,
            purpose=purpose,
            backend=backend,
            aspects=aspects,
            anonymized=anonymized,
            context=context,
        )
    except PolicyViolation as exc:
        click.echo(f"POLICY VIOLATION: {exc}", err=True)
        sys.exit(EXIT_POLICY)
    except NotImplementedError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    out = Path(out_dir)
    write_judgements(out / "judgements.jsonl", result.judgements)
    write_run_meta(out / "run.json", result.meta)
    n_review = write_review_queue(out / "review_queue.csv", result.judgements)
    (out / "report.md").write_text(
        markdown_report(result.meta, result.judgements, pol), encoding="utf-8"
    )
    click.echo(
        f"run {result.meta.run_id}: {result.meta.n_items} items → {result.meta.n_judgements} judgements, "
        f"{n_review} for review · {out}/"
    )


@main.command("eval")
@click.argument("gold_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--backend", "backend_name", default="lexicon", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print the full result as JSON.")
@click.option("--no-alt-test", is_flag=True)
def eval_cmd(gold_path: str, backend_name: str, as_json: bool, no_alt_test: bool) -> None:
    """Evaluate a backend on a gold set (JSONL)."""
    try:
        backend = get_backend(backend_name)
    except NotImplementedError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    gold = load_gold(gold_path)
    res = evaluate(backend, gold, with_alt_test=not no_alt_test)
    if as_json:
        click.echo(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return
    click.echo(f"backend {res.backend} ({res.model_version}) · n={res.n}")
    click.echo(
        f"accuracy {res.accuracy:.3f} · macro-F1 {res.macro_f1:.3f} · "
        f"kappa {res.kappa_vs_gold:.3f} · ECE {res.ece:.3f}"
    )
    for tag, row in res.by_tag.items():
        click.echo(f"  {tag:14} n={row['n']:<3} acc={row['accuracy']:.2f}")
    if res.alt_test:
        a = res.alt_test
        click.echo(
            f"alt-test: omega={a['omega']:.2f} ({'PASS' if a['passed'] else 'FAIL'}) "
            f"eps={a['epsilon']} alpha={a['alpha']}"
        )
    if res.failures:
        click.echo(f"failures: {len(res.failures)} (use --json for details)")


@main.command()
@click.argument("judgements_path", type=click.Path(exists=True, dir_okay=False))
def validate(judgements_path: str) -> None:
    """Validate a judgements file against the JSON schema."""
    import jsonschema

    schema_path = Path(__file__).resolve().parents[2] / "schema" / "judgement.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    n = 0
    for line in Path(judgements_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        jsonschema.validate(row, schema)
        Judgement.model_validate(row)
        n += 1
    click.echo(f"{n} judgements valid")


if __name__ == "__main__":  # pragma: no cover
    main()
