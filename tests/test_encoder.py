"""Encoder backend against a fake tokenizer and model: no download, no torch."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from click.testing import CliRunner

from sentimentkit import backends
from sentimentkit.backends import BackendUnavailable, EncoderBackend, NotAvailable, get_backend
from sentimentkit.backends.base import ASPECT_OPTIONS, DEFAULT_OPTIONS
from sentimentkit.backends.encoder import (
    DEFAULT_MODEL,
    ENV_BATCH_SIZE,
    ENV_DEVICE,
    ENV_MODEL,
    ENV_REVISION,
    MULTILINGUAL_MODEL,
    PRESETS,
    clean_germansentiment,
    clean_twitter,
    label_map,
)
from sentimentkit.cli import main
from sentimentkit.eval import evaluate, load_gold
from sentimentkit.models import Polarity, TextItem
from sentimentkit.pipeline import run

FIX = Path(__file__).resolve().parents[1] / "fixtures"
GERMAN_LABELS = {0: "positive", 1: "negative", 2: "neutral"}  # german-sentiment-bert
XLMR_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}  # twitter-xlm-roberta
COMMIT = "0123456789abcdef0123456789abcdef01234567"

POSITIVE_WORDS = ("super", "gut", "freundlich")
NEGATIVE_WORDS = ("chaos", "spät", "mangelhaft")


class FakeTensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeBatch(dict):
    def to(self, device: str) -> FakeBatch:
        self.device = device
        return self


class FakeTokenizer:
    """One token per word plus two specials, like a word-level tokenizer."""

    def __init__(self, model_max_length: int = 512) -> None:
        self.model_max_length = model_max_length
        self.calls: list[dict] = []

    def __call__(self, texts, **kwargs) -> FakeBatch:
        self.calls.append({"texts": list(texts), **kwargs})
        lengths = [min(len(t.split()) + 2, kwargs["max_length"]) for t in texts]
        width = max(lengths)
        mask = [[1] * n + [0] * (width - n) for n in lengths]
        return FakeBatch(input_ids=list(texts), attention_mask=FakeTensor(mask))


class FakeModel:
    """Logits from keywords, in the class order of `id2label`."""

    def __init__(self, id2label: dict | None = None, commit: str | None = None) -> None:
        self.config = SimpleNamespace(id2label=id2label or GERMAN_LABELS)
        if commit:
            self.config._commit_hash = commit
        self.batches: list[list[str]] = []
        self.device: str | None = None
        self.evaluated = False

    def __call__(self, *, input_ids, attention_mask) -> SimpleNamespace:
        self.batches.append(list(input_ids))
        return SimpleNamespace(logits=FakeTensor([self.logits(t) for t in input_ids]))

    def logits(self, text: str) -> list[float]:
        score = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        if any(w in text for w in POSITIVE_WORDS):
            score["positive"] = 3.0
        if any(w in text for w in NEGATIVE_WORDS):
            score["negative"] = 3.5
        return [score[self.config.id2label[i].lower()] for i in range(len(self.config.id2label))]

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self


def fake_encoder(**kwargs) -> EncoderBackend:
    tokenizer = kwargs.pop("tokenizer", None) or FakeTokenizer()
    model = kwargs.pop("model", None) or FakeModel()
    return EncoderBackend(tokenizer=tokenizer, model=model, **kwargs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (ENV_MODEL, ENV_REVISION, ENV_DEVICE, ENV_BATCH_SIZE):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_extra(monkeypatch):
    """Simulate an environment without the `encoder` extra."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)


# ------------------------------------------------------------ configuration
def test_defaults_are_german_model_pinned_on_cpu():
    b = EncoderBackend()
    assert b.model_name == DEFAULT_MODEL
    assert b.device == "cpu"
    assert b.revision == PRESETS[DEFAULT_MODEL].revision
    # a pinned commit is the version; nothing is loaded to report it
    assert b.model_version == f"{DEFAULT_MODEL}@{PRESETS[DEFAULT_MODEL].revision}"
    assert b._model is None


def test_configuration_from_environment_and_arguments(monkeypatch):
    monkeypatch.setenv(ENV_MODEL, MULTILINGUAL_MODEL)
    monkeypatch.setenv(ENV_DEVICE, "cuda")
    monkeypatch.setenv(ENV_BATCH_SIZE, "4")
    b = EncoderBackend()
    assert (b.model_name, b.device, b.batch_size) == (MULTILINGUAL_MODEL, "cuda", 4)
    assert b.revision == PRESETS[MULTILINGUAL_MODEL].revision
    explicit = EncoderBackend(DEFAULT_MODEL, revision=COMMIT, device="cpu", batch_size=2)
    assert explicit.model_version == f"{DEFAULT_MODEL}@{COMMIT}"
    assert (explicit.device, explicit.batch_size) == ("cpu", 2)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        EncoderBackend(batch_size=-1)
    with pytest.raises(ValueError):
        EncoderBackend(model=FakeModel())  # tokenizer missing


def test_model_version_resolves_a_branch_to_the_loaded_commit():
    on_main = fake_encoder(revision="main", model=FakeModel(commit=COMMIT))
    assert on_main.model_version == f"{DEFAULT_MODEL}@{COMMIT}"
    unknown = fake_encoder(model_name="some/local-model")
    assert unknown.model_version == "some/local-model@unpinned"


# ------------------------------------------------------------- decisions
def test_distribution_is_the_softmax_and_confidence_its_top():
    d = fake_encoder().decide("Die Betreuung ist super.")
    logits = {"positive": 3.0, "negative": 0.0, "neutral": 1.0}
    z = sum(math.exp(v) for v in logits.values())
    expected = {k: math.exp(v) / z for k, v in logits.items()}
    assert d.polarity is Polarity.POSITIVE
    assert set(d.distribution) == {o.value for o in DEFAULT_OPTIONS}
    assert d.distribution["unclear"] == 0.0  # the escape option stays in the answer set
    for k, p in expected.items():
        assert d.distribution[k] == pytest.approx(p, abs=1e-4)
    assert d.confidence == pytest.approx(max(expected.values()), abs=1e-4)
    assert d.intensity == pytest.approx(expected["positive"] - expected["negative"], abs=1e-4)
    assert d.evidence[0].startswith("positive(")


def test_label_order_comes_from_the_model_config():
    german = fake_encoder().decide("Der Ablauf war ein Chaos.")
    xlmr = fake_encoder(model=FakeModel(XLMR_LABELS)).decide("Der Ablauf war ein Chaos.")
    assert german.polarity is xlmr.polarity is Polarity.NEGATIVE
    assert german.distribution == xlmr.distribution


def test_unmapped_labels_fail_loudly():
    with pytest.raises(ValueError, match="LABEL_0"):
        label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValueError):
        label_map({0: "positive", 1: "positive"})


def test_decide_many_batches_and_keeps_input_order():
    tokenizer, model = FakeTokenizer(), FakeModel()
    b = fake_encoder(tokenizer=tokenizer, model=model, batch_size=2)
    texts = [
        "Der Bus kommt jeden Morgen viel zu spät an.",
        "Super.",
        "Die Schule beginnt um acht Uhr.",
        "Die Lehrperson ist sehr freundlich.",
        "Chaos.",
    ]
    decisions = b.decide_many(texts)
    assert [len(batch) for batch in model.batches] == [2, 2, 1]
    assert [d.polarity for d in decisions] == [
        Polarity.NEGATIVE,
        Polarity.POSITIVE,
        Polarity.NEUTRAL,
        Polarity.POSITIVE,
        Polarity.NEGATIVE,
    ]
    call = tokenizer.calls[0]
    assert call["padding"] is True and call["truncation"] is True
    assert call["max_length"] == 512 and call["return_tensors"] == "pt"


def test_input_is_cleaned_like_the_training_data():
    tokenizer, model = FakeTokenizer(), FakeModel()
    fake_encoder(tokenizer=tokenizer, model=model).decide("Am 3. Mai: super! https://x.ch @schule")
    assert model.batches == [["am drei mai super"]]
    assert clean_germansentiment("Klasse 5b,\nGrösse") == "klasse fünfb grösse"
    assert clean_twitter("@schule toll https://x.ch") == "@user toll http"


def test_truncation_is_flagged():
    b = fake_encoder(tokenizer=FakeTokenizer(model_max_length=6))
    assert b.max_length == 6
    assert "TRUNCATED?" in b.decide("Die Betreuung ist wirklich ganz super gewesen.").evidence
    assert "TRUNCATED?" not in b.decide("Super.").evidence


def test_dialect_and_language_are_flagged_not_corrected():
    b = fake_encoder()
    assert "DIALECT?" in b.decide("D Lehrerin isch mega nett gsi und hät üs guet gholfe.").evidence
    assert "LANG?:fr" in b.decide("Le bus est en retard.", lang="fr").evidence
    xlmr = fake_encoder(model_name=MULTILINGUAL_MODEL, model=FakeModel(XLMR_LABELS))
    assert not any(e.startswith("LANG?") for e in xlmr.decide("Le bus.", lang="fr").evidence)


# ----------------------------------------------------------------- aspects
def test_absent_aspect_is_not_mentioned_without_a_forward_pass():
    model = FakeModel()
    d = fake_encoder(model=model).decide(
        "Die Lehrperson erklärt sehr klar.", aspect="mensa", aspect_terms=("mensa", "essen")
    )
    assert d.polarity is Polarity.NOT_MENTIONED
    assert d.distribution["not_mentioned"] == 1.0
    assert model.batches == []


def test_aspect_is_scoped_to_the_clause_that_mentions_it():
    model = FakeModel()
    d = fake_encoder(model=model).decide(
        "Der Unterricht ist super, der Mittagstisch dagegen ein Chaos.",
        aspect="mensa",
        aspect_terms=("mittagstisch",),
    )
    assert model.batches == [["der mittagstisch dagegen ein chaos"]]
    assert d.polarity is Polarity.NEGATIVE
    assert set(d.distribution) == {o.value for o in ASPECT_OPTIONS}
    assert "aspect:mensa:1-clause" in d.evidence


# -------------------------------------------------------- loading & extra
def test_missing_extra_is_reported_not_crashed(no_extra):
    b = EncoderBackend()
    assert b.missing_extra() == "encoder"
    with pytest.raises(BackendUnavailable, match=r'pip install -e "\.\[encoder\]"'):
        b.decide("Die Betreuung ist super.")


def test_loader_uses_revision_device_and_the_cache_commit(monkeypatch):
    loaded: list[tuple[str, str, dict]] = []
    model = FakeModel()

    def loader(kind, result):
        def from_pretrained(name, **kwargs):
            loaded.append((kind, name, kwargs))
            return result

        return SimpleNamespace(from_pretrained=from_pretrained)

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = loader("tokenizer", FakeTokenizer())
    transformers.AutoModelForSequenceClassification = loader("model", model)
    hub = ModuleType("transformers.utils.hub")
    hub.cached_file = lambda name, filename, **kw: f"/hf/models--x/snapshots/{COMMIT}/{filename}"
    hub.extract_commit_hash = lambda path, _: path.split("/snapshots/")[1].split("/")[0]
    torch = ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    for name, module in {
        "transformers": transformers,
        "transformers.utils": ModuleType("transformers.utils"),
        "transformers.utils.hub": hub,
        "torch": torch,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    b = EncoderBackend(revision="main", device="cpu")
    assert b.model_version == f"{DEFAULT_MODEL}@{COMMIT}"
    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"  # no background Hub conversion
    assert loaded == [
        ("tokenizer", DEFAULT_MODEL, {"revision": "main"}),
        ("model", DEFAULT_MODEL, {"revision": "main"}),
    ]
    assert model.device == "cpu" and model.evaluated
    assert b.decide("Super.").polarity is Polarity.POSITIVE


# ------------------------------------------------------------- integration
def test_registry_serves_the_encoder():
    assert isinstance(get_backend("encoder"), EncoderBackend)
    assert not isinstance(backends.available()["encoder"], NotAvailable)


def test_pipeline_batches_per_aspect_and_keeps_order(policy):
    model = FakeModel()
    items = [TextItem(text="Die Betreuung ist super."), TextItem(text="Der Bus kommt zu spät.")]
    res = run(
        items,
        policy=policy,
        purpose="evaluation",
        backend=fake_encoder(model=model),
        aspects={"betreuung": ("betreuung",)},
    )
    assert [(j.text_id, j.aspect) for j in res.judgements] == [
        (items[0].text_id, "overall"),
        (items[0].text_id, "betreuung"),
        (items[1].text_id, "overall"),
        (items[1].text_id, "betreuung"),
    ]
    assert len(model.batches) == 2  # one forward pass for 'overall', one for 'betreuung'
    assert res.judgements[3].polarity is Polarity.NOT_MENTIONED
    assert res.meta.model_version == res.judgements[0].model_version


def test_evaluate_runs_on_the_gold_set():
    res = evaluate(fake_encoder(), load_gold(FIX / "de_synthetic_gold.jsonl"))
    assert res.n == 30 and res.backend == "encoder"


def test_cli_backends_lists_encoder_state(no_extra):
    r = CliRunner().invoke(main, ["backends"])
    assert r.exit_code == 0
    line = next(x for x in r.output.splitlines() if x.startswith("encoder"))
    assert "needs [encoder]" in line and DEFAULT_MODEL in line


def test_cli_analyze_without_extra_exits_1(no_extra, tmp_path):
    r = CliRunner().invoke(
        main,
        [
            "analyze",
            str(FIX / "survey_synthetic.csv"),
            "--purpose",
            "survey_open_answers",
            "--backend",
            "encoder",
            "-o",
            str(tmp_path / "o"),
        ],
    )
    assert r.exit_code == 1
    assert "[encoder]" in r.output


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "encoder", fake_encoder)


def test_cli_eval_compare_prints_gate(fake_registry):
    gold = str(FIX / "de_synthetic_gold.jsonl")
    r = CliRunner().invoke(main, ["eval", gold, "--compare", "lexicon,encoder"])
    assert r.exit_code == 0, r.output
    assert "baseline lexicon" in r.output
    assert any(x.startswith("lexicon") for x in r.output.splitlines())
    assert any(x.startswith("encoder") for x in r.output.splitlines())
    assert "irony" in r.output
    assert "eval.beats_baseline(encoder, lexicon):" in r.output


def test_cli_eval_compare_json(fake_registry):
    gold = str(FIX / "de_synthetic_gold.jsonl")
    r = CliRunner().invoke(main, ["eval", gold, "--compare", "lexicon,encoder", "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["baseline"] == "lexicon" and payload["min_gain"] == 0.05
    assert [x["backend"] for x in payload["results"]] == ["lexicon", "encoder"]
    assert isinstance(payload["beats_baseline"]["encoder"], bool)
    assert payload["results"][0]["n"] == payload["results"][1]["n"] == 30


@pytest.mark.parametrize(
    "args",
    [
        ["--compare", "lexicon"],
        ["--compare", "lexicon,lexicon"],
        ["--compare", "lexicon,encoder", "--backend", "lexicon"],
        ["--compare", "lexicon,encodr"],
    ],
)
def test_cli_eval_compare_rejects_bad_input(fake_registry, args):
    gold = str(FIX / "de_synthetic_gold.jsonl")
    r = CliRunner().invoke(main, ["eval", gold, *args])
    assert r.exit_code == 1
    assert "ERROR" in r.output
