# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LLM scoring backend `backends/llm_score.py` (stage 2b): fixed-answer scoring, no free text. One typed question per text and aspect with single-token labels A–E (positive, neutral, negative, unclear, not_mentioned; the overall question offers A–D), rendered with the model's chat template with thinking switched off; every label must add exactly one distinct token after the prompt. A text that names none of an aspect's terms is answered `not_mentioned` without a question, as in the other backends: without that rule Qwen3-4B carried a text's overall tone over to aspects it never names. Label probabilities from SGLang `/v1/score` (default, token ids sent so the model reads exactly the checked tokens) or Ollama `/api/generate` logprobs (fallback, neutral sampler options). Restricted softmax over the labels as `distribution`, `label-mass`, `LOW-LABEL-MASS?`, `NO-LABEL?` and `TRUNCATED?` in the evidence, `prompt_hash` of the rendered prompt in every atom. Configuration via `SENTIMENTKIT_LLM_*` (scorer, base URL, model, tokenizer and its revision, calibration temperature, timeout); default Qwen3-4B with a pinned tokenizer. `model_version` names scorer, served model (SGLang `/model_info` incl. weight version, Ollama digest), tokenizer commit, prompt version and calibration temperature; a classification model behind SGLang is refused
- Calibration in `sentimentkit eval`: ECE before and after temperature scaling for backends whose distribution is a softmax (encoder, llm-score), with the fitted T. T is fitted on the log-loss of the top label being right; the ECE after scaling is 5-fold cross-fitted. `eval/calibration.py`, pure Python. The `--compare` table gets an `ECE-TS` column
- Encoder backend `backends/encoder.py` (stage 2a): fine-tuned transformer classifier, local, CPU by default. Default model `oliverguhr/german-sentiment-bert`, alternative `cardiffnlp/twitter-xlm-roberta-base-sentiment` for French and Italian; model, revision, device and batch size via `SENTIMENTKIT_ENCODER_*` or constructor arguments. Softmax as `distribution`, top probability as `confidence`, P(positive) − P(negative) as `intensity`; `model_version` is `<model>@<commit>`, both presets pinned to a commit, branches and tags resolved to the loaded commit. Real batch inference in `decide_many` (length-sorted batches), per-preset input cleaning as in training, clause scoping and `not_mentioned` as in the lexicon backend, evidence flags `DIALECT?`, `LANG?:<lang>`, `TRUNCATED?`. `transformers`/`torch` are imported only on first use; a missing extra ends with exit code 1 and the install command. transformers' background safetensors conversion is switched off (it would contact the Hub and download from an open pull request)
- `sentimentkit eval GOLD --compare lexicon,encoder`: several backends on the same gold set, metrics and tag slices side by side, and `eval.beats_baseline()` against the first one; `--json` for the full results
- `backends.base.decide_grouped()`: answers mixed questions with one `decide_many` call per aspect and language, used by `eval` and `analyze`
- Tests for the encoder backend with a mocked tokenizer and model (no download, no torch in CI) and for `--compare`; README section on the encoder backend with fixture figures and caveats (EN/DE)
- `scripts/build_gold_candidates.py`: stage-1 annotation template from public source documents (JSONL with id, source, date, title, text) — sentence split via `preprocess.split_sentences` with repair of German false splits ("3. März", "12 Mio. Franken"), school-reference filter from `fixtures/keywords_schule.yaml` (keywords and exclude list), 5–60 tokens, exact and near-duplicate removal (token-set Jaccard), seeded round-robin sampling across sources (`--target`), unconfirmed rule-based tag hints (`negation?`, `dialect?`), policy check on person-level fields (exit 2), UTF-8 with BOM for Excel, no overwrite of an annotated template without `--force`
- `scripts/candidates_to_gold.py`: merges independently annotated CSV copies by `text_id` into the gold-set JSONL — majority label or `unclear` on dissent (optional `a3` adjudicates), `dissent` tag, label aliases, aspect `schule` with the keyword list as `aspect_terms`, schema validation, Cohen's kappa a1 vs. a2 and the stage-1 tag slices
- `docs/ANNOTATION.md`: annotation guide — labels with one example each, rules for irony, dialect, mixed sentences and negation, tag vocabulary, privacy, Excel workflow, kappa target ≥ 0.7, rights question for media text
- `fixtures/sources_synthetic.jsonl` (5 synthetic source documents) and `fixtures/keywords_schule.yaml`
- Tests for both scripts; README section on the stage-1 workflow (EN/DE)

### Changed
- Registry: `llm-score` is a real backend instead of the `NotAvailable` placeholder; only `llm-cloud` remains scheduled
- Optional extra `llm`: `transformers` (tokenizer only, no torch) and `jinja2` next to `httpx`
- `Backend.softmax_distribution` marks backends whose distribution is a softmax with the polarity as its argmax (encoder, llm-score)
- Registry: `encoder` is a real backend instead of the `NotAvailable` placeholder; `sentimentkit backends` shows `available`, `needs [encoder]` or the roadmap stage
- Optional extra `encoder`: `torch>=2.6` (transformers loads pickled weights only from 2.6 on, CVE-2025-32434), plus `sentencepiece` and `protobuf` for the XLM-R tokenizer
- Aspect scoping moved to `preprocess.aspect_clauses()` and the `not_mentioned` answer to `backends.base.not_mentioned()`, shared by all backends; the lexicon eval figures are unchanged
- `eval` and `analyze` batch their questions per aspect; unknown backends end with a one-line error instead of a traceback
- `schema/gold.schema.json`: optional `source` field, so every gold instance keeps its provenance

### Fixed
- `preprocess.split_sentences` no longer ends a sentence after a German day or class ordinal ("am 3. März") or a common abbreviation ("12 Mio. Franken", "z. B.", "St. Gallen"); "Art.", "Abs.", "Nr." and "Ziff." only stay inside the sentence when a number follows. Offsets stay exact. The lexicon backend and `scripts/build_gold_candidates.py` (whose local repair step is gone) use the shared rule; the stage-0 eval numbers are unchanged

## [0.1.0] - 2026-10-14

### Added
- Stage-0 skeleton: Python package, CLI (`analyze`, `eval`, `backends`, `policy-check`, `validate`)
- Evidence-atom data model and JSON schemas (judgement, run, gold set)
- `policy.yaml` with checks before the backend and before aggregation; exit code 2 on violation
- Lexicon backend for German: negation (before and Satzklammer), compounds via suffix match, multiword entries, intensifiers/diminishers, contrast, negation-immune adverbs, Swiss-German flagging
- Backend registry with stage-2 placeholders (`encoder`, `llm-score`, `llm-cloud`)
- Eval harness: accuracy, macro-F1, Cohen's kappa, ECE, tag slices, Alternative Annotator Test (pure Python)
- Human gate and review queue; Markdown report with k-anonymity floor
- Synthetic German gold set (30), synthetic survey CSV (24), aspects example
- Claude skill `sentiment-scan`
- Bilingual README, ROADMAP, ARCHITECTURE, CI with pinned ruff and format gate
