# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scripts/build_gold_candidates.py`: stage-1 annotation template from public source documents (JSONL with id, source, date, title, text) — sentence split via `preprocess.split_sentences` with repair of German false splits ("3. März", "12 Mio. Franken"), school-reference filter from `fixtures/keywords_schule.yaml` (keywords and exclude list), 5–60 tokens, exact and near-duplicate removal (token-set Jaccard), seeded round-robin sampling across sources (`--target`), unconfirmed rule-based tag hints (`negation?`, `dialect?`), policy check on person-level fields (exit 2), UTF-8 with BOM for Excel, no overwrite of an annotated template without `--force`
- `scripts/candidates_to_gold.py`: merges independently annotated CSV copies by `text_id` into the gold-set JSONL — majority label or `unclear` on dissent (optional `a3` adjudicates), `dissent` tag, label aliases, aspect `schule` with the keyword list as `aspect_terms`, schema validation, Cohen's kappa a1 vs. a2 and the stage-1 tag slices
- `docs/ANNOTATION.md`: annotation guide — labels with one example each, rules for irony, dialect, mixed sentences and negation, tag vocabulary, privacy, Excel workflow, kappa target ≥ 0.7, rights question for media text
- `fixtures/sources_synthetic.jsonl` (5 synthetic source documents) and `fixtures/keywords_schule.yaml`
- Tests for both scripts; README section on the stage-1 workflow (EN/DE)

### Changed
- `schema/gold.schema.json`: optional `source` field, so every gold instance keeps its provenance

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
