"""Fixed-answer scoring with a local LLM (Stage 2b).

One typed question per text and aspect, answered by the probability of
single-token labels at the first answer position, never by generated text:

    A positive · B neutral · C negative · D unclear · E not_mentioned (aspects only)

A text in which no clause names one of the aspect's terms is answered
`not_mentioned` without a question, as in the lexicon and encoder backends:
measured on Qwen3-4B, the model otherwise carries a text's overall tone over
to aspects the text never names. A text that names the aspect gets the full
question, E included, on the whole text.

The prompt is rendered with the model's own chat template, with thinking
switched off so that the answer is the first token. Each label must add
exactly one distinct token after the prompt. The label probabilities come
from SGLang's `/v1/score` (default) or from Ollama's `/api/generate` logprobs
(fallback for machines without a GPU, e.g. a Raspberry Pi). The softmax
restricted to the labels is the distribution; `label-mass` in the evidence
says how much of the model's probability the labels held at all.

`httpx`, `transformers` (tokenizer only, no torch) and `jinja2` come with the
optional extra `llm`. Texts go to the configured server: point it only at a
server you run.

Configuration, by constructor argument or environment variable:

    SENTIMENTKIT_LLM_SCORER              sglang | ollama          default: sglang
    SENTIMENTKIT_LLM_BASE_URL            scoring server           default: localhost:30000 | :11434
    SENTIMENTKIT_LLM_MODEL               model on that server     default: Qwen/Qwen3-4B | qwen3:4b
    SENTIMENTKIT_LLM_TOKENIZER           Hugging Face id or path  default: Qwen/Qwen3-4B
    SENTIMENTKIT_LLM_TOKENIZER_REVISION  commit, tag or branch    default: pinned commit
    SENTIMENTKIT_LLM_CALIBRATION_T       temperature from `sentimentkit eval`  default: 1
    SENTIMENTKIT_LLM_TIMEOUT             seconds per request      default: 120
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Sequence
from typing import Any, NamedTuple

from ..models import Polarity, stable_hash
from ..preprocess import aspect_clauses
from .base import (
    ASPECT_OPTIONS,
    DEFAULT_OPTIONS,
    Backend,
    BackendUnavailable,
    Decision,
    not_mentioned,
)

ENV_SCORER = "SENTIMENTKIT_LLM_SCORER"
ENV_BASE_URL = "SENTIMENTKIT_LLM_BASE_URL"
ENV_MODEL = "SENTIMENTKIT_LLM_MODEL"
ENV_TOKENIZER = "SENTIMENTKIT_LLM_TOKENIZER"
ENV_TOKENIZER_REVISION = "SENTIMENTKIT_LLM_TOKENIZER_REVISION"
ENV_CALIBRATION_T = "SENTIMENTKIT_LLM_CALIBRATION_T"
ENV_TIMEOUT = "SENTIMENTKIT_LLM_TIMEOUT"

INSTALL_HINT = 'pip install -e ".[llm]"'


class Scorer(NamedTuple):
    base_url: str
    model: str


SCORERS = {
    "sglang": Scorer("http://localhost:30000", "Qwen/Qwen3-4B"),
    "ollama": Scorer("http://localhost:11434", "qwen3:4b"),
}
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
DEFAULT_TOKENIZER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# A change to SYSTEM, OPTION_TEXT or question() is a new prompt: bump the
# version, so that runs with different wording never share a model_version.
PROMPT_VERSION = "v1"
LABELS = ("A", "B", "C", "D", "E")
SYSTEM = (
    "Du bestimmst die Tonalität von Texten zu einem Thema. Du beurteilst den Text, "
    "nicht die Person, die ihn geschrieben hat. Antworte nur mit dem Buchstaben einer Option."
)
OPTION_TEXT = {
    Polarity.POSITIVE: "positiv",
    Polarity.NEUTRAL: "neutral",
    Polarity.NEGATIVE: "negativ",
    Polarity.UNCLEAR: "unklar – gemischt, ironisch oder nicht entscheidbar",
    Polarity.NOT_MENTIONED: "nicht erwähnt – der Text sagt nichts zu diesem Thema",
}
MAX_TEXT_CHARS = 6000  # longer texts are cut and flagged TRUNCATED?
OLLAMA_TOP_LOGPROBS = 20
LOW_LABEL_MASS = 0.5


def question(text: str, aspect: str, aspect_terms: Sequence[str]) -> str:
    """The typed question: the text as data, then a fixed answer set."""
    if aspect == "overall":
        ask, options = "Welche Tonalität hat der Text insgesamt?", DEFAULT_OPTIONS
    else:
        examples = f" (z. B. {', '.join(aspect_terms)})" if aspect_terms else ""
        ask = f"Welche Tonalität hat der Text gegenüber dem Thema «{aspect}»{examples}?"
        options = ASPECT_OPTIONS
    lines = [f"{label}) {OPTION_TEXT[o]}" for label, o in zip(LABELS, options)]
    return "\n".join(
        [
            "Text:",
            "<<<",
            text,
            ">>>",
            "",
            f"Frage: {ask}",
            *lines,
            "",
            "Antworte nur mit dem Buchstaben.",
        ]
    )


def prompt_fingerprint() -> str:
    """Hash of the wording, independent of any text; pinned in the tests."""
    return stable_hash(
        [SYSTEM, question("{text}", "overall", ()), question("{text}", "{aspect}", ("{term}",))]
    )


def label_token_ids(
    tokenizer: Any, prompt: str, labels: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Token ids of the prompt and of each label after it. Every label must
    add exactly one distinct token, else the model cannot answer in one step."""
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    ids: list[int] = []
    for label in labels:
        full = list(tokenizer.encode(prompt + label, add_special_tokens=False))
        if len(full) != len(prompt_ids) + 1 or full[:-1] != prompt_ids or full[-1] in ids:
            raise ValueError(
                f"answer label {label!r} is not one distinct token after the chat prompt; "
                "this tokenizer cannot be scored with fixed answers"
            )
        ids.append(full[-1])
    return prompt_ids, ids


def restricted_softmax(probs: Sequence[float], t: float) -> list[float]:
    """Softmax of log(p) / t over the labels; labels with p = 0 stay at 0."""
    logs = [math.log(p) / t if p > 0 else -math.inf for p in probs]
    top = max(logs)
    weights = [math.exp(v - top) if v > -math.inf else 0.0 for v in logs]
    z = sum(weights)
    return [w / z for w in weights]


class LLMScoreBackend(Backend):
    name = "llm-score"
    stage = 2
    softmax_distribution = True

    def __init__(
        self,
        model: str | None = None,
        *,
        scorer: str | None = None,
        base_url: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        calibration_t: float | None = None,
        batch_size: int = 16,
        timeout: float | None = None,
        tokenizer: Any = None,
        client: Any = None,
    ) -> None:
        """`tokenizer` and `client` take already built objects (tests, callers
        that share them); without them both are created on first use."""
        env = os.environ.get
        self.scorer = (scorer or env(ENV_SCORER) or "sglang").lower()
        if self.scorer not in SCORERS:
            raise ValueError(f"unknown scorer {self.scorer!r}; known: {', '.join(SCORERS)}")
        defaults = SCORERS[self.scorer]
        self.base_url = (base_url or env(ENV_BASE_URL) or defaults.base_url).rstrip("/")
        self.model = model or env(ENV_MODEL) or defaults.model
        self.tokenizer_name = tokenizer_name or env(ENV_TOKENIZER) or DEFAULT_TOKENIZER
        self.tokenizer_revision = (
            tokenizer_revision
            or env(ENV_TOKENIZER_REVISION)
            or (DEFAULT_TOKENIZER_REVISION if self.tokenizer_name == DEFAULT_TOKENIZER else None)
        )
        self.calibration_t = float(calibration_t or env(ENV_CALIBRATION_T) or 1.0)
        if not self.calibration_t > 0:
            raise ValueError("the calibration temperature must be greater than 0")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.timeout = float(timeout or env(ENV_TIMEOUT) or 120.0)
        self._tokenizer = tokenizer
        self._client = client
        self._timeout_errors: tuple[type[BaseException], ...] = (TimeoutError,)
        self._transport_errors: tuple[type[BaseException], ...] = (OSError,)
        self._served: str | None = None  # the model as the server names it

    # -------------------------------------------------------------- set-up
    @property
    def note(self) -> str:
        return f"{self.scorer} · {self.model} · {self.base_url}"

    def missing_extra(self) -> str | None:
        needed = []
        if self._client is None:
            needed.append("httpx")
        if self._tokenizer is None:
            needed += ["transformers", "jinja2"]
        found = all(importlib.util.find_spec(m) for m in needed)
        return None if found else "llm"

    def _connect(self) -> None:
        if self._tokenizer is None or self._client is None:
            try:
                import httpx
                import jinja2  # noqa: F401 — transformers renders chat templates with it
                from transformers import AutoTokenizer
                from transformers.utils.hub import cached_file, extract_commit_hash
            except ImportError as exc:
                raise BackendUnavailable(
                    f"backend 'llm-score' needs the optional extra: {INSTALL_HINT}"
                ) from exc
            if self._tokenizer is None:
                kwargs = {"revision": self.tokenizer_revision} if self.tokenizer_revision else {}
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **kwargs)
                config = cached_file(self.tokenizer_name, "tokenizer_config.json", **kwargs)
                self.tokenizer_revision = (
                    extract_commit_hash(config, None) or self.tokenizer_revision
                )
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
                self._timeout_errors = (httpx.TimeoutException,)
                self._transport_errors = (httpx.TransportError,)
        if self._served is None:
            self._served = self._identify()

    def _call(self, path: str, payload: dict | None = None) -> Any:
        """GET (no payload) or POST to the server; None on 404."""
        url = self.base_url + path
        try:
            if payload is None:
                response = self._client.get(url)
            else:
                response = self._client.post(url, json=payload)
        except self._timeout_errors as exc:
            raise BackendUnavailable(
                f"{self.base_url} did not answer within {self.timeout:g} s; "
                f"raise {ENV_TIMEOUT} or lower the batch size"
            ) from exc
        except self._transport_errors as exc:
            raise BackendUnavailable(f"no scoring server at {self.base_url}: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(f"{url} answered {response.status_code}: {response.text[:300]}")
        return response.json()

    def _identify(self) -> str:
        if self.scorer == "sglang":
            # /model_info, before SGLang renamed it /get_model_info
            info = self._call("/model_info") or self._call("/get_model_info")
            if not info:
                return self.model
            if info.get("is_generation") is False:
                raise BackendUnavailable(
                    f"{self.base_url} serves a classification model; llm-score needs a causal LM"
                )
            served = str(info.get("model_path") or self.model)
            weights = info.get("weight_version")
            return f"{served}@{weights}" if weights and weights != "default" else served
        tags = self._call("/api/tags") or {}
        wanted = {self.model, f"{self.model}:latest"}
        for entry in tags.get("models", []):
            if entry.get("name") in wanted or entry.get("model") in wanted:
                return f"{self.model}@{str(entry.get('digest', ''))[:12]}"
        raise BackendUnavailable(
            f"model {self.model!r} is not on the Ollama server {self.base_url}: ollama pull {self.model}"
        )

    @property
    def model_version(self) -> str:
        """`<scorer>:<served model>+tok:<commit>+prompt:<version>[+T<t>]`."""
        self._connect()
        version = (
            f"{self.scorer}:{self._served}+tok:{(self.tokenizer_revision or 'unpinned')[:12]}"
            f"+prompt:{PROMPT_VERSION}"
        )
        return version + (f"+T{self.calibration_t:g}" if self.calibration_t != 1.0 else "")

    # ------------------------------------------------------------ deciding
    def render(self, text: str, aspect: str, aspect_terms: Sequence[str]) -> str:
        """The prompt exactly as the model sees it."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question(text, aspect, aspect_terms)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def decide(
        self,
        text: str,
        *,
        aspect: str = "overall",
        aspect_terms: tuple[str, ...] = (),
        lang: str | None = None,
    ) -> Decision:
        return self.decide_many([text], aspect=aspect, aspect_terms=aspect_terms, lang=lang)[0]

    def decide_many(
        self,
        texts: list[str],
        *,
        aspect: str = "overall",
        aspect_terms: tuple[str, ...] = (),
        lang: str | None = None,
    ) -> list[Decision]:
        answers: dict[int, Decision] = {}
        asked: list[int] = []
        for i, text in enumerate(texts):
            if aspect != "overall" and aspect_terms and not aspect_clauses(text, aspect_terms):
                answers[i] = not_mentioned(aspect)
            else:
                asked.append(i)
        if asked:
            self._connect()
            options = ASPECT_OPTIONS if aspect != "overall" else DEFAULT_OPTIONS
            labels = LABELS[: len(options)]
            prompts = [self.render(texts[i][:MAX_TEXT_CHARS], aspect, aspect_terms) for i in asked]
            encoded = [label_token_ids(self._tokenizer, p, labels) for p in prompts]
            if any(label_ids != encoded[0][1] for _, label_ids in encoded):
                raise ValueError("the answer labels map to different tokens across prompts")
            if self.scorer == "sglang":
                scores = self._score_sglang([ids for ids, _ in encoded], encoded[0][1])
            else:
                scores = [self._score_ollama(p, labels) for p in prompts]
            for i, probs, prompt in zip(asked, scores, prompts):
                truncated = len(texts[i]) > MAX_TEXT_CHARS
                answers[i] = self._decision(probs, options, prompt, truncated)
        return [answers[i] for i in range(len(texts))]

    def _score_sglang(self, prompt_ids: list[list[int]], label_ids: list[int]) -> list[list[float]]:
        """Token ids go to the server, so the model reads exactly the tokens
        whose label continuation was checked."""
        out: list[list[float]] = []
        for start in range(0, len(prompt_ids), self.batch_size):
            batch = prompt_ids[start : start + self.batch_size]
            payload = {
                "model": self.model,
                "query": [],
                "items": batch,
                "label_token_ids": label_ids,
                "apply_softmax": False,
            }
            body = self._call("/v1/score", payload)
            if body is None:
                raise BackendUnavailable(
                    f"{self.base_url} has no /v1/score; for Ollama set {ENV_SCORER}=ollama"
                )
            rows = body.get("scores", [])
            if len(rows) != len(batch) or any(len(r) != len(label_ids) for r in rows):
                raise RuntimeError(f"/v1/score returned {len(rows)} rows for {len(batch)} prompts")
            out += [[float(p) for p in r] for r in rows]
        return out

    def _score_ollama(self, prompt: str, labels: Sequence[str]) -> list[float]:
        """Logprobs of the first token. Neutral sampler options: temperature 1
        and no repeat penalty, so the label ratios are the model's whether
        Ollama reports them before or after the sampler (A–E occur in the
        prompt, a repeat penalty would push them down)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": OLLAMA_TOP_LOGPROBS,
            "options": {
                "num_predict": 1,
                "temperature": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "seed": 0,
                "num_ctx": 4096,
            },
        }
        body = self._call("/api/generate", payload)
        if body is None:
            raise BackendUnavailable(f"{self.base_url} has no /api/generate; is it Ollama?")
        first = (body.get("logprobs") or [None])[0]
        if not first:
            raise BackendUnavailable(
                f"Ollama at {self.base_url} returned no logprobs; it needs a release with logprobs support"
            )
        logprobs = {e["token"]: e["logprob"] for e in first.get("top_logprobs", [])}
        logprobs.setdefault(first["token"], first["logprob"])
        return [math.exp(logprobs[label]) if label in logprobs else 0.0 for label in labels]

    def _decision(
        self, probs: list[float], options: Sequence[Polarity], prompt: str, truncated: bool
    ) -> Decision:
        mass = sum(probs)
        flags = ["TRUNCATED?"] if truncated else []
        prompt_hash = stable_hash(prompt)
        if mass <= 0:
            dist = {o.value: round(1 / len(options), 4) for o in options}
            cues = ["label-mass:0.00", "NO-LABEL?", *flags]
            return Decision(Polarity.UNCLEAR, 0.0, 0.0, dist, cues, prompt_hash)
        q = restricted_softmax(probs, self.calibration_t)
        ranked = sorted(zip(options, q), key=lambda oq: oq[1], reverse=True)
        p = {o.value: v for o, v in ranked}
        top, confidence = ranked[0]
        intensity = p[Polarity.POSITIVE.value] - p[Polarity.NEGATIVE.value]
        cues = [f"{o.value}({v:.2f})" for o, v in ranked] + [f"label-mass:{mass:.2f}"]
        if mass < LOW_LABEL_MASS:
            cues.append("LOW-LABEL-MASS?")
        dist = {o.value: round(p[o.value], 4) for o in options}
        return Decision(
            top, round(intensity, 4), round(confidence, 4), dist, cues + flags, prompt_hash
        )
