"""Command line interface.

    sentimentkit backends
    sentimentkit analyze INPUT --purpose media_resonance [--backend lexicon] [--aspects aspects.yaml] -o out/
    sentimentkit eval GOLD.jsonl [--backend lexicon | --compare lexicon,encoder] [--json]
    sentimentkit policy-check [--policy policy.yaml]
    sentimentkit validate judgements.jsonl

Exit codes: 0 ok · 1 error · 2 policy violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import click
import yaml
from click.core import ParameterSource

from . import __version__
from .backends import Backend, BackendUnavailable, NotAvailable, available, get_backend
from .eval import MIN_GAIN, EvalResult, beats_baseline, evaluate, load_gold
from .io import read_items, write_judgements, write_review_queue, write_run_meta
from .models import Judgement
from .pipeline import run as run_pipeline
from .policy import PolicyViolation, load_policy
from .report import markdown_report

EXIT_POLICY = 2


def _fail(message: str) -> NoReturn:
    click.echo(f"ERROR: {message}", err=True)
    sys.exit(1)


def _backend(name: str) -> Backend:
    try:
        return get_backend(name)
    except (KeyError, NotImplementedError) as exc:
        _fail(str(exc.args[0]))


@click.group()
@click.version_option(__version__, prog_name="sentimentkit")
def main() -> None:
    """Sentiment toolkit with swappable backends, evidence atoms and policy-as-code."""


@main.command()
def backends() -> None:
    """List backends: available, missing an optional extra, or scheduled for a later stage."""
    for name, b in available().items():
        if isinstance(b, NotAvailable):
            state = f"stage {b.stage}"
        else:
            missing = b.missing_extra()
            state = f"needs [{missing}]" if missing else "available"
        note = getattr(b, "note", "")
        click.echo(f"{name:10} {state:16} {note}".rstrip())


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
        backend = _backend(backend_name)
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
    except BackendUnavailable as exc:
        _fail(str(exc))

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
@click.option(
    "--compare",
    default=None,
    metavar="BASELINE,CANDIDATE[,...]",
    help="Measure several backends on the same gold set; the first is the baseline "
    "for eval.beats_baseline().",
)
@click.option("--json", "as_json", is_flag=True, help="Print the full result as JSON.")
@click.option("--no-alt-test", is_flag=True)
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    gold_path: str,
    backend_name: str,
    compare: str | None,
    as_json: bool,
    no_alt_test: bool,
) -> None:
    """Evaluate a backend on a gold set (JSONL), or compare several with --compare."""
    names = [backend_name]
    if compare is not None:
        if ctx.get_parameter_source("backend_name") is ParameterSource.COMMANDLINE:
            _fail("--compare and --backend exclude each other")
        names = [n.strip() for n in compare.split(",") if n.strip()]
        if len(names) < 2 or len(set(names)) != len(names):
            _fail("--compare needs two or more different backends, e.g. lexicon,encoder")
    chosen = [_backend(n) for n in names]
    gold = load_gold(gold_path)
    try:
        results = [evaluate(b, gold, with_alt_test=not no_alt_test) for b in chosen]
    except BackendUnavailable as exc:
        _fail(str(exc))
    if compare is None:
        _print_result(results[0], as_json)
    else:
        _print_comparison(gold_path, results, as_json)


def _print_result(res: EvalResult, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return
    click.echo(f"backend {res.backend} ({res.model_version}) · n={res.n}")
    click.echo(
        f"accuracy {res.accuracy:.3f} · macro-F1 {res.macro_f1:.3f} · "
        f"kappa {res.kappa_vs_gold:.3f} · ECE {res.ece:.3f}"
    )
    click.echo(_calibration_line(res))
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


def _calibration_line(res: EvalResult) -> str:
    c = res.calibration
    if c is None:
        return "calibration: n/a — temperature scaling needs a softmax distribution"
    return (
        f"calibration: ECE {c['ece_before']:.3f} → {c['ece_after']:.3f} after temperature "
        f"scaling (T={c['temperature']:g}, {c['folds']}-fold cross-fitted, "
        f"{c['n_scaled']} of {res.n} judgements rescaled)"
    )


def _print_comparison(gold_path: str, results: list[EvalResult], as_json: bool) -> None:
    baseline, candidates = results[0], results[1:]
    verdicts = {c.backend: beats_baseline(c, baseline) for c in candidates}
    if as_json:
        payload = {
            "gold": gold_path,
            "baseline": baseline.backend,
            "min_gain": MIN_GAIN,
            "beats_baseline": verdicts,
            "results": [r.to_dict() for r in results],
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"gold {gold_path} · n={baseline.n} · baseline {baseline.backend}")
    click.echo(
        f"{'backend':10} {'acc':>6} {'macro-F1':>8} {'kappa':>6} {'ECE':>6} {'ECE-TS':>6}"
        "  alt-test  model"
    )
    for r in results:
        alt = (
            f"{r.alt_test['omega']:.2f} {'PASS' if r.alt_test['passed'] else 'FAIL'}"
            if r.alt_test
            else "—"
        )
        ece_ts = f"{r.calibration['ece_after']:6.3f}" if r.calibration else f"{'—':>6}"
        click.echo(
            f"{r.backend:10} {r.accuracy:6.3f} {r.macro_f1:8.3f} {r.kappa_vs_gold:6.3f} "
            f"{r.ece:6.3f} {ece_ts}  {alt:8}  {r.model_version}"
        )
    if baseline.by_tag:
        click.echo("tag slices (accuracy):")
        click.echo(f"  {'tag':14} {'n':>3}  " + "  ".join(f"{r.backend:>8}" for r in results))
        for tag, row in baseline.by_tag.items():
            accs = "  ".join(f"{r.by_tag[tag]['accuracy']:8.2f}" for r in results)
            click.echo(f"  {tag:14} {row['n']:>3}  {accs}")
    for r in results:
        if r.calibration:
            click.echo(f"{r.backend}: {_calibration_line(r)}")
    for c in candidates:
        gain = c.macro_f1 - baseline.macro_f1
        click.echo(
            f"eval.beats_baseline({c.backend}, {baseline.backend}): "
            f"{'yes' if verdicts[c.backend] else 'no'} — macro-F1 {gain:+.3f}, "
            f"gate +{MIN_GAIN:.2f}"
        )


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
