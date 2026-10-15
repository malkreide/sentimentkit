"""llm-score against a fake tokenizer and a fake HTTP client: no server, no
httpx, no transformers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import jsonschema
import pytest
from click.testing import CliRunner

from sentimentkit import backends
from sentimentkit.backends import BackendUnavailable, LLMScoreBackend, get_backend
from sentimentkit.backends.base import ASPECT_OPTIONS, DEFAULT_OPTIONS
from sentimentkit.backends.llm_score import (
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    ENV_BASE_URL,
    ENV_CALIBRATION_T,
    ENV_MODEL,
    ENV_SCORER,
    ENV_TIMEOUT,
    ENV_TOKENIZER,
    ENV_TOKENIZER_REVISION,
    MAX_TEXT_CHARS,
    OLLAMA_TOP_LOGPROBS,
    PROMPT_VERSION,
    SYSTEM,
    label_token_ids,
    prompt_fingerprint,
    question,
    restricted_softmax,
)
from sentimentkit.cli import main
from sentimentkit.eval import evaluate, load_gold
from sentimentkit.models import Polarity, TextItem, stable_hash
from sentimentkit.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
SERVED = "Qwen/Qwen3-4B"
DIGEST = "0123456789abcdef0123"


class CharTokenizer:
    """One token per character (its code point), and a Qwen-like chat template."""

    def __init__(self) -> None:
        self.template_kwargs: list[dict] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_kwargs.append(kwargs)
        body = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        return body + "<|assistant|>\n"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]


class MergingTokenizer(CharTokenizer):
    """Glues an answer letter to the newline before it: labels are not one token."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text]
        if len(ids) >= 2 and text[-1] in "ABCDE" and text[-2] == "\n":
            ids[-2:] = [1000 + ids[-1]]
        return ids


def answer_probs(prompt: str) -> dict[str, float]:
    """The fake model: raw next-token probabilities of A–E from keywords."""
    text = prompt.split("<<<")[1].split(">>>")[0]
    if "E) " in prompt and "nebenbei" in text:  # named, but not about the aspect
        return {"A": 0.01, "B": 0.02, "C": 0.01, "D": 0.01, "E": 0.9}
    if "super" in text:
        return {"A": 0.8, "B": 0.1, "C": 0.02, "D": 0.03, "E": 0.0}
    if "Chaos" in text:
        return {"A": 0.02, "B": 0.05, "C": 0.85, "D": 0.04, "E": 0.0}
    if "Toll, schon wieder" in text:
        return {"A": 0.3, "B": 0.05, "C": 0.3, "D": 0.3, "E": 0.0}
    if "Rauschen" in text:  # the model would rather answer something else
        return {"A": 0.05, "B": 0.1, "C": 0.05, "D": 0.0, "E": 0.0}
    return {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.02, "E": 0.0}


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict:
        return self._body


class FakeServer:
    """SGLang (/get_model_info, /v1/score) or Ollama (/api/tags, /api/generate)."""

    def __init__(self, kind: str = "sglang", *, down=False, routes=None) -> None:
        self.kind = kind
        self.down = down  # False, or the exception a request raises
        self.routes = routes or {}
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self._check()
        path = url.split("//", 1)[1].split("/", 1)[1]
        self.gets.append("/" + path)
        if "/" + path in self.routes:
            return self.routes["/" + path]
        if self.kind == "sglang" and path == "model_info":
            info = {"model_path": SERVED, "is_generation": True, "weight_version": "default"}
            return FakeResponse(200, info)
        if self.kind == "ollama" and path == "api/tags":
            return FakeResponse(200, {"models": [{"name": "qwen3:4b", "digest": DIGEST}]})
        return FakeResponse(404)

    def post(self, url: str, json: dict) -> FakeResponse:
        self._check()
        path = "/" + url.split("//", 1)[1].split("/", 1)[1]
        self.posts.append((path, json))
        if path in self.routes:
            return self.routes[path]
        if self.kind == "sglang" and path == "/v1/score":
            labels = [chr(i) for i in json["label_token_ids"]]
            prompts = ["".join(chr(i) for i in ids) for ids in json["items"]]
            return FakeResponse(
                200, {"scores": [[answer_probs(p)[lab] for lab in labels] for p in prompts]}
            )
        if self.kind == "ollama" and path == "/api/generate":
            probs = answer_probs(json["prompt"])
            top = [{"token": t, "logprob": math.log(p)} for t, p in probs.items() if p > 0]
            top.append({"token": "Die", "logprob": math.log(0.01)})
            top.sort(key=lambda e: e["logprob"], reverse=True)
            first = {**top[0], "top_logprobs": top[: json["top_logprobs"]]}
            return FakeResponse(200, {"response": first["token"], "logprobs": [first]})
        return FakeResponse(404)

    def _check(self) -> None:
        if self.down:
            raise self.down


def fake_llm(kind: str = "sglang", **kwargs) -> LLMScoreBackend:
    server = kwargs.pop("server", None) or FakeServer(kind)
    tokenizer = kwargs.pop("tokenizer", None) or CharTokenizer()
    return LLMScoreBackend(scorer=kind, tokenizer=tokenizer, client=server, **kwargs)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        ENV_SCORER,
        ENV_BASE_URL,
        ENV_MODEL,
        ENV_TOKENIZER,
        ENV_TOKENIZER_REVISION,
        ENV_CALIBRATION_T,
        ENV_TIMEOUT,
    ):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------ configuration
def test_defaults_are_sglang_with_qwen3_and_a_pinned_tokenizer():
    b = LLMScoreBackend()
    assert (b.scorer, b.base_url, b.model) == ("sglang", "http://localhost:30000", SERVED)
    assert (b.tokenizer_name, b.tokenizer_revision) == (
        DEFAULT_TOKENIZER,
        DEFAULT_TOKENIZER_REVISION,
    )
    assert b.calibration_t == 1.0
    ollama = LLMScoreBackend(scorer="ollama")
    assert (ollama.base_url, ollama.model) == ("http://localhost:11434", "qwen3:4b")


def test_configuration_from_environment(monkeypatch):
    monkeypatch.setenv(ENV_SCORER, "ollama")
    monkeypatch.setenv(ENV_BASE_URL, "http://gpu-host:11434/")
    monkeypatch.setenv(ENV_MODEL, "qwen3:8b")
    monkeypatch.setenv(ENV_TOKENIZER, "Qwen/Qwen3-8B")
    monkeypatch.setenv(ENV_CALIBRATION_T, "1.8")
    monkeypatch.setenv(ENV_TIMEOUT, "600")
    b = LLMScoreBackend()
    assert (b.scorer, b.base_url, b.model) == ("ollama", "http://gpu-host:11434", "qwen3:8b")
    assert b.tokenizer_name == "Qwen/Qwen3-8B" and b.tokenizer_revision is None
    assert b.calibration_t == 1.8 and b.timeout == 600.0


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="scorer"):
        LLMScoreBackend(scorer="vllm")
    with pytest.raises(ValueError, match="temperature"):
        LLMScoreBackend(calibration_t=-1)


# ------------------------------------------------------------------ prompt
def test_typed_question_offers_a_fixed_answer_set():
    overall = question("Der Bus ist zu spät.", "overall", ())
    assert all(f"{x}) " in overall for x in "ABCD") and "E) " not in overall
    aspect = question("Die Mensa ist gut.", "mensa", ("mensa", "essen"))
    assert "«mensa» (z. B. mensa, essen)" in aspect
    assert "E) nicht erwähnt" in aspect
    assert aspect.index("<<<") < aspect.index("Die Mensa ist gut.") < aspect.index(">>>")


def test_prompt_wording_is_versioned():
    # Changed the wording? Bump PROMPT_VERSION and update this hash: runs with
    # different prompts must not share a model_version.
    assert (PROMPT_VERSION, prompt_fingerprint()) == ("v1", "2c2d853c0e0e490c")


def test_prompt_uses_the_chat_template_without_thinking():
    tokenizer = CharTokenizer()
    b = fake_llm(tokenizer=tokenizer)
    prompt = b.render("Text", "overall", ())
    assert prompt.startswith(f"<|system|>{SYSTEM}") and prompt.endswith("<|assistant|>\n")
    assert tokenizer.template_kwargs[-1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_each_label_must_be_one_distinct_token():
    prompt_ids, ids = label_token_ids(CharTokenizer(), "prompt\n", "ABCDE")
    assert ids == [65, 66, 67, 68, 69] and len(prompt_ids) == 7
    with pytest.raises(ValueError, match="not one distinct token"):
        label_token_ids(MergingTokenizer(), "prompt\n", "ABCDE")


def test_restricted_softmax_and_temperature():
    assert restricted_softmax([0.8, 0.2, 0.0], 1.0) == pytest.approx([0.8, 0.2, 0.0])
    flat = restricted_softmax([0.8, 0.2, 0.0], 2.0)
    assert flat[0] == pytest.approx(math.sqrt(0.8) / (math.sqrt(0.8) + math.sqrt(0.2)))
    assert flat[2] == 0.0


# ---------------------------------------------------------------- SGLang
def test_sglang_scores_token_ids_in_batches():
    server = FakeServer("sglang")
    b = fake_llm(server=server, batch_size=2)
    texts = ["Die Betreuung ist super.", "Der Ablauf war ein Chaos.", "Die Schule beginnt um acht."]
    decisions = b.decide_many(texts)
    scores = [body for path, body in server.posts if path == "/v1/score"]
    assert [len(body["items"]) for body in scores] == [2, 1]
    first = scores[0]
    assert first["label_token_ids"] == [65, 66, 67, 68]  # A–D for the overall question
    assert first["apply_softmax"] is False and first["query"] == [] and first["model"] == SERVED
    assert "".join(chr(i) for i in first["items"][0]) == b.render(texts[0], "overall", ())
    assert [d.polarity for d in decisions] == [
        Polarity.POSITIVE,
        Polarity.NEGATIVE,
        Polarity.NEUTRAL,
    ]


def test_distribution_is_the_restricted_softmax_with_provenance():
    b = fake_llm()
    d = b.decide("Die Betreuung ist super.")
    raw = {"positive": 0.8, "neutral": 0.1, "negative": 0.02, "unclear": 0.03}
    z = sum(raw.values())
    assert set(d.distribution) == {o.value for o in DEFAULT_OPTIONS}
    for k, p in raw.items():
        assert d.distribution[k] == pytest.approx(p / z, abs=1e-4)
    assert d.confidence == max(d.distribution.values())
    assert d.intensity == pytest.approx((0.8 - 0.02) / z, abs=1e-4)
    assert "label-mass:0.95" in d.evidence
    assert d.prompt_hash == stable_hash(b.render("Die Betreuung ist super.", "overall", ()))
    assert b.model_version == f"sglang:{SERVED}+tok:{DEFAULT_TOKENIZER_REVISION[:12]}+prompt:v1"


def test_calibration_temperature_flattens_and_is_in_the_version():
    plain = fake_llm().decide("Die Betreuung ist super.")
    scaled_backend = fake_llm(calibration_t=2.0)
    scaled = scaled_backend.decide("Die Betreuung ist super.")
    assert scaled.polarity is plain.polarity
    assert scaled.confidence < plain.confidence
    assert scaled_backend.model_version.endswith("+prompt:v1+T2")


def test_aspect_question_can_answer_not_mentioned():
    b = fake_llm()
    aside, present = b.decide_many(
        ["Die Mensa wird nebenbei genannt.", "Die Mensa ist super."],
        aspect="mensa",
        aspect_terms=("mensa",),
    )
    assert set(aside.distribution) == {o.value for o in ASPECT_OPTIONS}
    assert aside.polarity is Polarity.NOT_MENTIONED and aside.prompt_hash
    assert present.polarity is Polarity.POSITIVE


def test_texts_without_the_aspect_are_not_asked():
    server = FakeServer()
    absent = fake_llm(server=server).decide(
        "Die Betreuung im Hort ist chaotisch.", aspect="schulweg", aspect_terms=("schulweg", "bus")
    )
    assert absent.polarity is Polarity.NOT_MENTIONED
    assert absent.evidence == ["aspect:schulweg:absent"] and absent.prompt_hash is None
    assert server.posts == [] and server.gets == []  # no question, not even a connection


def test_low_and_missing_label_mass_are_flagged():
    b = fake_llm()
    low = b.decide("Rauschen im Kanal.")
    assert "LOW-LABEL-MASS?" in low.evidence
    nothing = fake_llm(
        server=FakeServer(
            routes={"/v1/score": FakeResponse(200, {"scores": [[0.0, 0.0, 0.0, 0.0]]})}
        )
    ).decide("Egal.")
    assert nothing.polarity is Polarity.UNCLEAR and nothing.confidence == 0.0
    assert "NO-LABEL?" in nothing.evidence


def test_long_text_is_cut_and_flagged():
    b = fake_llm()
    d = b.decide("Die Betreuung ist super. " + "x" * MAX_TEXT_CHARS)
    assert "TRUNCATED?" in d.evidence


def test_sglang_identity_comes_from_the_server():
    versioned = FakeServer(
        routes={
            "/model_info": FakeResponse(200, {"model_path": "/models/q3", "weight_version": "w2"})
        }
    )
    assert fake_llm(server=versioned).model_version.startswith("sglang:/models/q3@w2+tok:")
    old = FakeServer(
        routes={
            "/model_info": FakeResponse(404),
            "/get_model_info": FakeResponse(200, {"model_path": "Qwen/Qwen3-8B"}),
        }
    )
    assert fake_llm(server=old).model_version.startswith("sglang:Qwen/Qwen3-8B+tok:")
    classifier = FakeServer(routes={"/model_info": FakeResponse(200, {"is_generation": False})})
    with pytest.raises(BackendUnavailable, match="causal LM"):
        fake_llm(server=classifier).decide("Text")


def test_server_errors_are_reported():
    with pytest.raises(BackendUnavailable, match="no scoring server"):
        fake_llm(server=FakeServer(down=ConnectionError("refused"))).decide("Text")
    with pytest.raises(BackendUnavailable, match="did not answer within 5 s"):
        fake_llm(server=FakeServer(down=TimeoutError("read")), timeout=5).decide("Text")
    no_score = FakeServer(routes={"/v1/score": FakeResponse(404)})
    with pytest.raises(BackendUnavailable, match="SENTIMENTKIT_LLM_SCORER=ollama"):
        fake_llm(server=no_score).decide("Text")
    broken = FakeServer(routes={"/v1/score": FakeResponse(500, {"error": "boom"})})
    with pytest.raises(RuntimeError, match="500"):
        fake_llm(server=broken).decide("Text")


# ---------------------------------------------------------------- Ollama
def test_ollama_reads_first_token_logprobs_with_a_neutral_sampler():
    server = FakeServer("ollama")
    b = fake_llm("ollama", server=server)
    d = b.decide("Der Ablauf war ein Chaos.")
    path, body = server.posts[-1]
    assert path == "/api/generate"
    assert body["raw"] is True and body["logprobs"] is True and body["stream"] is False
    assert body["top_logprobs"] == OLLAMA_TOP_LOGPROBS
    assert body["prompt"] == b.render("Der Ablauf war ein Chaos.", "overall", ())
    options = body["options"]
    assert options["num_predict"] == 1
    assert options["temperature"] == 1.0 and options["repeat_penalty"] == 1.0
    assert d.polarity is Polarity.NEGATIVE
    assert b.model_version.startswith(f"ollama:qwen3:4b@{DIGEST[:12]}+tok:")


def test_ollama_and_sglang_agree_on_the_same_model():
    text = "Toll, schon wieder eine Verspätung."
    assert (
        fake_llm("ollama").decide(text).distribution == fake_llm("sglang").decide(text).distribution
    )


def test_ollama_model_must_be_pulled():
    server = FakeServer("ollama", routes={"/api/tags": FakeResponse(200, {"models": []})})
    with pytest.raises(BackendUnavailable, match="ollama pull qwen3:4b"):
        fake_llm("ollama", server=server).decide("Text")


def test_ollama_without_logprobs_is_reported():
    server = FakeServer("ollama", routes={"/api/generate": FakeResponse(200, {"response": "A"})})
    with pytest.raises(BackendUnavailable, match="no logprobs"):
        fake_llm("ollama", server=server).decide("Text")


# ------------------------------------------------------------ extra, CLI
def test_missing_extra_is_reported(monkeypatch):
    for module in ("httpx", "transformers", "jinja2"):
        monkeypatch.setitem(sys.modules, module, None)
    b = LLMScoreBackend()
    assert b.missing_extra() == "llm"
    with pytest.raises(BackendUnavailable, match=r'pip install -e "\.\[llm\]"'):
        b.decide("Text")


def test_registry_serves_llm_score():
    assert isinstance(get_backend("llm-score"), LLMScoreBackend)


def test_pipeline_atoms_carry_prompt_hash_and_validate(policy):
    items = [TextItem(text="Die Mensa ist super."), TextItem(text="Der Bus kommt zu spät.")]
    res = run(
        items,
        policy=policy,
        purpose="evaluation",
        backend=fake_llm(),
        aspects={"mensa": ("mensa",)},
    )
    schema = json.loads((ROOT / "schema" / "judgement.schema.json").read_text(encoding="utf-8"))
    hashes = set()
    for j in res.judgements:
        jsonschema.validate(json.loads(j.model_dump_json(exclude_none=True)), schema)
        assert j.model_version == res.meta.model_version
        hashes.add(j.prompt_hash)
    # one rendered prompt per question; the bus text never names the Mensa
    assert len(hashes - {None}) == 3
    absent = next(j for j in res.judgements if j.prompt_hash is None)
    assert (absent.aspect, absent.polarity) == ("mensa", Polarity.NOT_MENTIONED)


def test_evaluate_reports_calibration():
    res = evaluate(fake_llm(), load_gold(FIX / "de_synthetic_gold.jsonl"))
    assert res.calibration is not None
    assert res.calibration["temperature"] > 0 and 0 <= res.calibration["ece_after"] <= 1


def test_cli_compare_prints_ece_before_and_after(monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "llm-score", fake_llm)
    gold = str(FIX / "de_synthetic_gold.jsonl")
    r = CliRunner().invoke(main, ["eval", gold, "--compare", "lexicon,llm-score"])
    assert r.exit_code == 0, r.output
    assert "ECE-TS" in r.output
    assert "llm-score: calibration: ECE" in r.output
    assert "eval.beats_baseline(llm-score, lexicon):" in r.output
