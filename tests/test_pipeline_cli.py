from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sentimentkit.cli import main
from sentimentkit.eval import evaluate, load_gold
from sentimentkit.io import read_items
from sentimentkit.models import Polarity, TextItem
from sentimentkit.pipeline import run
from sentimentkit.report import aggregate, markdown_report

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"


def test_run_produces_one_judgement_per_item_and_aspect(policy, lexicon):
    items = [TextItem(text="Die Betreuung ist super."), TextItem(text="Der Bus kommt zu spät.")]
    res = run(
        items,
        policy=policy,
        purpose="evaluation",
        backend=lexicon,
        aspects={"betreuung": ("betreuung",)},
    )
    assert res.meta.n_items == 2
    assert res.meta.n_judgements == 4
    assert res.meta.policy_hash == policy.hash
    assert res.meta.model_version == lexicon.model_version
    by = {(j.text_id, j.aspect): j for j in res.judgements}
    assert by[(items[0].text_id, "betreuung")].polarity is Polarity.POSITIVE
    assert by[(items[1].text_id, "betreuung")].polarity is Polarity.NOT_MENTIONED


def test_aggregate_enforces_floor(policy, lexicon):
    items = [TextItem(text="gut") for _ in range(policy.aggregation_floor - 1)]
    res = run(items, policy=policy, purpose="evaluation", backend=lexicon)
    import pytest

    from sentimentkit.policy import PolicyViolation

    with pytest.raises(PolicyViolation):
        aggregate(res.judgements, policy=policy)
    report = markdown_report(res.meta, res.judgements, policy)
    assert "unterdrückt" in report


def test_gold_fixture_baseline_gate(lexicon):
    gold = load_gold(FIX / "de_synthetic_gold.jsonl")
    res = evaluate(lexicon, gold)
    # the ROADMAP stage-0 gate: baseline runs and documents its floor
    assert res.n == 30
    assert res.accuracy >= 0.85
    assert res.by_tag["irony"]["accuracy"] == 0.0  # known, documented weakness
    assert res.alt_test is not None


def test_cli_analyze_and_validate(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out"
    r = runner.invoke(
        main,
        [
            "analyze",
            str(FIX / "survey_synthetic.csv"),
            "--purpose",
            "survey_open_answers",
            "--aspects",
            str(FIX / "aspects_schule.yaml"),
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out / "judgements.jsonl").exists() and (out / "report.md").exists()
    meta = json.loads((out / "run.json").read_text())
    assert meta["n_items"] == 24
    v = runner.invoke(main, ["validate", str(out / "judgements.jsonl")])
    assert v.exit_code == 0 and "valid" in v.output


def test_cli_policy_violation_exit_code(tmp_path):
    runner = CliRunner()
    r = runner.invoke(
        main,
        [
            "analyze",
            str(FIX / "survey_synthetic.csv"),
            "--purpose",
            "marketing_profiling",
            "-o",
            str(tmp_path / "o"),
        ],
    )
    assert r.exit_code == 2
    assert "POLICY VIOLATION" in r.output


def test_cli_unavailable_backend(tmp_path):
    runner = CliRunner()
    r = runner.invoke(
        main,
        [
            "analyze",
            str(FIX / "survey_synthetic.csv"),
            "--purpose",
            "media_resonance",
            "--backend",
            "llm-cloud",
            "-o",
            str(tmp_path / "o"),
        ],
    )
    assert r.exit_code == 1
    assert "stage 2" in r.output


def test_read_items_csv_keeps_meta():
    items = read_items(FIX / "survey_synthetic.csv")
    assert len(items) == 24
    assert set(items[0].meta) == {"schulkreis"}
