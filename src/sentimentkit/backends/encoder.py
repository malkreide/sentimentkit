"""Encoder backend (Stage 2): a fine-tuned transformer classifier, run locally.

Default model is `oliverguhr/german-sentiment-bert` (German, MIT); for French
and Italian text `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Both are
three-way classifiers. The softmax over the logits is the decision's
distribution, its top probability the confidence. No text leaves the machine:
only the weights come from the Hugging Face Hub, once, or from a local
directory or the Hugging Face cache.

`transformers` and `torch` come with the optional extra (`pip install -e
".[encoder]"`). Nothing heavy is imported before the first decision, so the
registry and `sentimentkit backends` work without the extra.

Configuration, by constructor argument or environment variable:

    SENTIMENTKIT_ENCODER_MODEL       hub id or local path   default: german-sentiment-bert
    SENTIMENTKIT_ENCODER_REVISION    commit, tag or branch  default: pinned commit (presets)
    SENTIMENTKIT_ENCODER_DEVICE      cpu, cuda, mps         default: cpu
    SENTIMENTKIT_ENCODER_BATCH_SIZE  texts per forward pass default: 16
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import Polarity
from ..preprocess import aspect_clauses, guess_language, tokenize
from .base import (
    ASPECT_OPTIONS,
    DEFAULT_OPTIONS,
    Backend,
    BackendUnavailable,
    Decision,
    not_mentioned,
)

DEFAULT_MODEL = "oliverguhr/german-sentiment-bert"
MULTILINGUAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

ENV_MODEL = "SENTIMENTKIT_ENCODER_MODEL"
ENV_REVISION = "SENTIMENTKIT_ENCODER_REVISION"
ENV_DEVICE = "SENTIMENTKIT_ENCODER_DEVICE"
ENV_BATCH_SIZE = "SENTIMENTKIT_ENCODER_BATCH_SIZE"

INSTALL_HINT = 'pip install -e ".[encoder]"'

_COMMIT = re.compile(r"[0-9a-f]{40}")
_LABELS = {
    "positive": Polarity.POSITIVE,
    "pos": Polarity.POSITIVE,
    "neutral": Polarity.NEUTRAL,
    "neu": Polarity.NEUTRAL,
    "negative": Polarity.NEGATIVE,
    "neg": Polarity.NEGATIVE,
}

# Each model expects its input cleaned the way its training data was.
_URL = re.compile(r"https*\S+")
_MENTION = re.compile(r"@\S+")
_NON_LETTER = re.compile(r"[^A-Za-züöäÖÜÄß ]")
_DIGITS = str.maketrans(
    {
        "0": " null",
        "1": " eins",
        "2": " zwei",
        "3": " drei",
        "4": " vier",
        "5": " fünf",
        "6": " sechs",
        "7": " sieben",
        "8": " acht",
        "9": " neun",
    }
)


def clean_germansentiment(text: str) -> str:
    """The `germansentiment` package's `clean_text`: no URLs or mentions,
    digits spelled out, letters only, lower case."""
    text = _MENTION.sub("", _URL.sub("", text.replace("\n", " ")))
    text = _NON_LETTER.sub("", text.translate(_DIGITS))
    return " ".join(text.split()).lower()


def clean_twitter(text: str) -> str:
    """The cardiffnlp model card's `preprocess`: user and link placeholders."""
    words = text.split(" ")
    return " ".join(
        "@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w
        for w in words
    )


@dataclass(frozen=True)
class Preset:
    revision: str  # pinned commit, so that a re-upload on the Hub cannot change results
    languages: tuple[str, ...]
    clean: Callable[[str], str]


PRESETS: dict[str, Preset] = {
    DEFAULT_MODEL: Preset(
        "b1177ff59e305c966836ba2825d3dc2efc53f125", ("de",), clean_germansentiment
    ),
    MULTILINGUAL_MODEL: Preset(
        "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8",
        ("ar", "de", "en", "es", "fr", "hi", "it", "pt"),
        clean_twitter,
    ),
}


def label_map(id2label: Mapping[Any, str]) -> tuple[Polarity, ...]:
    """Polarity per class index, from the model config. Fails loudly instead
    of guessing what `LABEL_0` means."""
    labels: list[Polarity] = []
    for i in range(len(id2label)):
        raw = id2label.get(i, id2label.get(str(i)))
        polarity = _LABELS.get(str(raw).strip().lower())
        if polarity is None:
            raise ValueError(
                f"model class {i} is labelled {raw!r}; expected positive / neutral / negative"
            )
        labels.append(polarity)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError(f"model labels {labels} are not a sentiment classification")
    return tuple(labels)


def softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


class EncoderBackend(Backend):
    name = "encoder"
    stage = 2

    def __init__(
        self,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int = 512,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        """`tokenizer` and `model` take already loaded objects (tests, callers
        that share one model); without them the model loads on first use."""
        self.model_name = model_name or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.preset = PRESETS.get(self.model_name)
        self.revision = (
            revision
            or os.environ.get(ENV_REVISION)
            or (self.preset.revision if self.preset else None)
        )
        self.device = device or os.environ.get(ENV_DEVICE) or "cpu"
        self.batch_size = int(batch_size or os.environ.get(ENV_BATCH_SIZE) or 16)
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.max_length = max_length
        self._clean = self.preset.clean if self.preset else (lambda t: t)
        self._tokenizer: Any = None
        self._model: Any = None
        self._labels: tuple[Polarity, ...] = ()
        self._commit: str | None = None
        self._no_grad: Callable[[], Any] = contextlib.nullcontext
        if (tokenizer is None) != (model is None):
            raise ValueError("pass tokenizer and model together")
        if model is not None:
            self._attach(tokenizer, model)

    # ------------------------------------------------------------- loading
    @property
    def note(self) -> str:
        revision = (self.revision or "main")[:12]
        return f"{self.model_name}@{revision} · {self.device}"

    def missing_extra(self) -> str | None:
        if self._model is not None:
            return None
        found = all(importlib.util.find_spec(m) for m in ("transformers", "torch"))
        return None if found else "encoder"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401 — transformers without torch fails only later
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers.utils.hub import cached_file, extract_commit_hash
        except ImportError as exc:
            raise BackendUnavailable(
                f"backend 'encoder' needs the optional extra: {INSTALL_HINT}"
            ) from exc
        # For a repo with pickled weights only (the XLM-R preset), transformers would
        # otherwise ask a Hub service to convert them and fetch the result from an
        # open pull request in a background thread: egress nobody asked for.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        kwargs = {"revision": self.revision} if self.revision else {}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kwargs)
        model.to(self.device)
        model.eval()
        self._attach(tokenizer, model)
        if self._commit is None:  # transformers 5 no longer keeps it on the config
            # …/snapshots/<commit>/config.json in the Hugging Face cache; None for a local directory
            config_path = cached_file(self.model_name, "config.json", **kwargs)
            self._commit = extract_commit_hash(config_path, None)

    def _attach(self, tokenizer: Any, model: Any) -> None:
        self._labels = label_map(model.config.id2label)
        self._commit = getattr(model.config, "_commit_hash", None)
        limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(limit, int) and limit < self.max_length:
            self.max_length = limit
        try:
            import torch

            self._no_grad = torch.inference_mode
        except ImportError:  # an injected model that is not a torch module
            self._no_grad = contextlib.nullcontext
        self._tokenizer, self._model = tokenizer, model

    @property
    def model_version(self) -> str:
        """`<model>@<commit>`. A branch or tag is resolved to the commit the
        weights were actually loaded from, which loads the model."""
        if self.revision and _COMMIT.fullmatch(self.revision):
            return f"{self.model_name}@{self.revision}"
        self._load()
        return f"{self.model_name}@{self._commit or self.revision or 'unpinned'}"

    # ------------------------------------------------------------ deciding
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
        pending: list[tuple[int, str, list[str]]] = []  # index, model input, evidence
        for i, text in enumerate(texts):
            unit, evidence = text, self._flags(text, lang)
            if aspect != "overall" and aspect_terms:
                clauses = aspect_clauses(text, aspect_terms)
                if not clauses:
                    answers[i] = not_mentioned(aspect)
                    continue
                unit = " ".join(clauses)
                evidence.append(f"aspect:{aspect}:{len(clauses)}-clause")
            pending.append((i, self._clean(unit), evidence))
        if pending:
            self._load()
            results = self._infer([unit for _, unit, _ in pending])
            for (i, _, evidence), (probs, truncated) in zip(pending, results):
                if truncated:
                    evidence.append("TRUNCATED?")
                answers[i] = self._decision(probs, evidence, aspect)
        return [answers[i] for i in range(len(texts))]

    def _flags(self, text: str, lang: str | None) -> list[str]:
        detected, dialect = guess_language(tokenize(text))
        lang = lang or detected
        if dialect or lang == "gsw":
            return ["DIALECT?"]
        if self.preset and lang not in self.preset.languages:
            return [f"LANG?:{lang}"]
        return []

    def _infer(self, units: list[str]) -> list[tuple[list[float], bool]]:
        """Softmax and truncation flag per unit, `batch_size` units per
        forward pass. Sorting by length first keeps padding short."""
        out: list[tuple[list[float], bool]] = [([], False)] * len(units)
        order = sorted(range(len(units)), key=lambda i: len(units[i]))
        for start in range(0, len(order), self.batch_size):
            idxs = order[start : start + self.batch_size]
            batch = self._tokenizer(
                [units[i] for i in idxs],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self._no_grad():
                logits = self._model(**batch).logits.tolist()
            lengths = [sum(row) for row in batch["attention_mask"].tolist()]
            for i, row, n in zip(idxs, logits, lengths):
                out[i] = (softmax(row), n >= self.max_length)
        return out

    def _decision(self, probs: list[float], evidence: list[str], aspect: str) -> Decision:
        ranked = sorted(zip(self._labels, probs), key=lambda lp: lp[1], reverse=True)
        p = {label.value: prob for label, prob in ranked}
        options = ASPECT_OPTIONS if aspect != "overall" else DEFAULT_OPTIONS
        dist = {o.value: round(p.get(o.value, 0.0), 4) for o in options}
        intensity = p.get(Polarity.POSITIVE.value, 0.0) - p.get(Polarity.NEGATIVE.value, 0.0)
        top, confidence = ranked[0]
        cues = [f"{label.value}({prob:.2f})" for label, prob in ranked]
        return Decision(top, round(intensity, 4), round(confidence, 4), dist, cues + evidence)
